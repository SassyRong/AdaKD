import time
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler
from torch.optim import AdamW
import deepspeed
import random
import shutil
import json
from tqdm import tqdm
from utils import ModelEmaV3, LossEma
import math
from transformers import (
    AutoTokenizer,
    GenerationConfig
)
from arguments import get_args
from utils import (
    initialize,
    setup_model_and_optimizer,
    print_rank, 
    all_gather,
    save_rank,
    get_teacher_model,
    get_tokenizer,
    print_args,
    cosine_schedule,
    linear_schedule
)
from loss import forward_kl, reverse_kl, symmetric_kl, js_distance, tv_distance, adaptive_kl, AdaKD
from loss import skewed_forward_kl, skewed_reverse_kl, ab_div
from loss import calculate_gradient_conflict, analyze_idts_gradient_effect
from rouge_metric import compute_metrics_rouge, compute_metrics_bert, compute_metrics_bleu, compute_metrics_meteor
from data_utils.lm_datasets import LMTrainDataset
import wandb

from distillm import SampleGenerator, ReplayBuffer

torch.set_num_threads(4)

def prepare_dataset(args, tokenizer):
    data = {}
    rng_sample = random.Random(args.seed)
    if args.do_train:
        data["train"] = LMTrainDataset(args, tokenizer, args.data_dir, "train", args.train_num, args.train_ratio, rng_sample)
        print_rank("train num", len(data["train"]))
        data["dev"] = LMTrainDataset(args, tokenizer, args.data_dir, "valid", args.dev_num, args.dev_ratio, rng_sample)
    elif args.do_eval:
        data["test"] = LMTrainDataset(args, tokenizer, args.data_dir, "valid", args.dev_num, args.dev_ratio, rng_sample)
    else:
        raise ValueError("Do train and do eval must set one")
    return data

def save_ckpt(args, ckpt_name, model, tokenizer, ema_model=None):
    save_dir_path = os.path.join(args.save, ckpt_name)
    ema_model_path = os.path.join(args.save, "ema_model")
    if dist.get_rank() == 0:
        os.makedirs(save_dir_path, exist_ok=True)
        print_rank(f"Model save to {save_dir_path}")
        tokenizer.save_pretrained(save_dir_path)
        if ema_model is not None:
            os.makedirs(ema_model_path, exist_ok=True)
            tokenizer.save_pretrained(ema_model_path)
            print_rank(f"EMA model save to {ema_model_path}")
        if args.dynamic_temperature:
            model.module.base_model.save_pretrained(save_dir_path, safe_serialization=False)
            if ema_model is not None:
                ema_model.module.base_model.save_pretrained(ema_model_path, safe_serialization=False)
        else:
            model.module.save_pretrained(save_dir_path, safe_serialization=False)    
            if ema_model is not None:
                ema_model.module.save_pretrained(ema_model_path, safe_serialization=False)
                
    dist.barrier()

def get_eval_str(eval_results):
    strs = []
    for k, vs in eval_results.items():
        bst_epoch, bst_result = np.argmax(vs), np.max(vs)
        str_ = f'{k}_{bst_result}@{bst_epoch}'
        strs.append(str_)
    results_str = '-'.join(strs)
    return results_str

def get_distill_loss(args, no_model_batch, logits, teacher_logits, ratio=None, global_step=None, should_log=False):
    t = args.type
    if "srkd" in t:
        return skewed_reverse_kl(logits, teacher_logits, no_model_batch, args.skew_alpha)
    if "sfkd" in t:
        return skewed_forward_kl(logits, teacher_logits, no_model_batch, args.skew_alpha)
    if "fkd" in t:
        return forward_kl(logits, teacher_logits, no_model_batch)
    if "rkd" in t:
        return reverse_kl(logits, teacher_logits, no_model_batch)
    if "jsd" in t:
        return js_distance(logits, teacher_logits, no_model_batch)
    if "tvd" in t:
        return tv_distance(logits, teacher_logits, no_model_batch)
    if "akd" in t:
        return adaptive_kl(logits, teacher_logits, no_model_batch)
    if "Adakd" in t:
        return AdaKD(
            logits, teacher_logits, no_model_batch,
            loss_fn_name=args.loss_fn,
            rule=args.rule,
            selection_ratio=ratio,
            topk=10,
            adaptive_temperature=args.adaptive_temperature,
            temperature_base=args.temperature,
            temperature_scale=args.temperature_scale,
            temperature_direction=args.temperature_soften,
            should_log_stats=should_log,
            global_step=global_step
        )
    if "skd" in t:
        return symmetric_kl(logits, teacher_logits, no_model_batch, lam=1 - ratio)
    if "abkd" in t:
        return ab_div(logits, teacher_logits, no_model_batch, alpha=0.2, beta=0.7)
    raise ValueError("Unknown distillation loss type")
    


def finetune(args, tokenizer: AutoTokenizer, model: deepspeed.DeepSpeedEngine, optimizer: AdamW, lr_scheduler, dataset, device, teacher_model=None):
    print_rank("Start Fine-tuning")
    dp_world_size = dist.get_world_size()
    dp_rank = dist.get_rank()
    dp_group = None
    loss_func = nn.CrossEntropyLoss()

    sampler = DistributedSampler(dataset["train"], shuffle=True, drop_last=True, rank=dp_rank, num_replicas=dp_world_size)
    train_dataloader = DataLoader(
        dataset['train'], sampler=sampler, batch_size=args.batch_size, num_workers=args.num_workers, collate_fn=dataset["train"].collate)

    step, global_step = 1, 1
    total_loss, total_distil_loss, total_time = 0.0, 0.0, 0.0
    adaptive_threshold = args.init_threshold if "adaptive" in args.type else None
    if args.distillm or args.gkd:
        student_generator = SampleGenerator(args, tokenizer)
        replay_buffer = ReplayBuffer(args)
    if args.dynamic_temperature:
        temperature_container = model.module.temperature_container
    else:
        temperature = nn.Parameter(torch.tensor([args.temperature]), requires_grad=False)
        temperature = temperature.to(device)
    if "AdaKD" in args.type:
        sample_ratio = 1.0
        ema_loss = LossEma(decay=args.ema_decay_loss)
        cur_ema_loss = None
        prev_interval_ema_loss = None
        
    if args.model_ema:
        ema_model = ModelEmaV3(model.module, decay=args.ema_decay, device=device)

    prev_avg_loss = evaluate(args, tokenizer, model, dataset["dev"], "dev", 0, device)

    # prev_avg_loss = 0
    model.train()
    for epoch in range(args.epochs):
        sampler.set_epoch(epoch)

        model.train()
        for it, (model_batch, no_model_batch, gen_data) in enumerate(train_dataloader):

            dataset["train"].move_to_device(model_batch, no_model_batch, gen_data, device)
            torch.cuda.synchronize()
            st_time = time.time()
            
            if args.cal_sft_loss:
                gt_model_batch = model_batch.copy()
                gt_no_model_batch = no_model_batch.copy()

            if "adaptive" in args.type:
                if args.replay_ratio == "constant":
                    samp_threshold = adaptive_threshold * 0.5
                elif args.replay_ratio == "increasing":
                    samp_threshold = adaptive_threshold * global_step / args.total_iters
                else:
                    samp_threshold = adaptive_threshold * (1 - global_step / args.total_iters)
            
            # on-policy
            if args.gkd:
                r = np.random.uniform(0, 1)
                cond_mixed = ("mixed" in args.type and r < args.gkd_alpha)
                cond_adaptive = ("cosine" in args.type and r < cosine_schedule(global_step, args.total_iters, start=1, end=0))
                if cond_mixed or cond_adaptive:
                    model_batch = student_generator.run_sample(model, gen_data)
                    no_model_batch["label"] = model_batch.pop("no_model_batch")
                    if args.model_type not in ["gpt2"]:
                        model_batch.pop('position_ids')
                    model.train()
                    
            # adaptive off-policy
            if args.distillm:
                r = np.random.uniform(0, 1)
                if "mixed" in args.type and r < args.mixed_alpha:
                    model_batch = student_generator.run_sample(model, gen_data)
                    no_model_batch["label"] = model_batch.pop("no_model_batch")
                    
                    replay_buffer.move_to_memory(model_batch, no_model_batch)
                    model_batch, no_model_batch = replay_buffer.sample()
                    model_batch, no_model_batch = replay_buffer.move_to_device(model_batch, no_model_batch, device)
                    
                elif "adaptive" in args.type and (r < samp_threshold or (r < adaptive_threshold and len(replay_buffer) < args.capacity)):

                    model_batch = student_generator.run_sample(model, gen_data)
                    no_model_batch["label"] = model_batch.pop("no_model_batch")
                    
                    if args.model_type not in ["gpt2", "llama"]:
                        model_batch.pop('position_ids')
                        
                    replay_buffer.move_to_memory(model_batch, no_model_batch)
                    
                elif "adaptive" in args.type and r < adaptive_threshold:
                    model_batch, no_model_batch = replay_buffer.sample()
                    model_batch, no_model_batch = replay_buffer.move_to_device(model_batch, no_model_batch, device)

                model.train()
                
            outputs = model(**model_batch, use_cache=False)
            logits = outputs.logits
            
                
            if args.cal_sft_loss:
                outputs = model(**gt_model_batch, use_cache=False)
                sft_logits = outputs.logits
                lm_loss = loss_func(sft_logits.float().view(-1, sft_logits.shape[-1]), gt_no_model_batch["label"].view(-1))
            else:
                lm_loss = loss_func(logits.float().view(-1, logits.shape[-1]), no_model_batch["label"].view(-1))
            
            if teacher_model is not None:
                distil_loss = 0
                with torch.no_grad():
                    teacher_model.eval()
                    teacher_outputs = teacher_model(**model_batch, use_cache=False)
                    teacher_logits = teacher_outputs.logits
                    # align student and teacher logits for qwen
                    if args.model_type == "qwen":
                        teacher_logits = teacher_logits[:, :, :logits.shape[-1]]
                if args.ctkd:
                    temperature = temperature_container(cosine_schedule(global_step, args.total_iters, start=args.start_value, end=args.end_value))
                if args.dtkd:
                    logits_max, _ = logits.max(dim=-1, keepdim=True)
                    teacher_logits_max, _ = teacher_logits.max(dim=-1, keepdim=True)
                    logits_temperature = (2 * logits_max) / (teacher_logits_max + logits_max)
                    logits_teacher_temperature = (2 * teacher_logits_max) / (teacher_logits_max + logits_max)
                    logits = logits / logits_temperature
                    teacher_logits = teacher_logits / logits_teacher_temperature
                if args.NormKD:
                    sigma_student = logits.std(dim=-1, keepdim=True)
                    sigma_teacher = teacher_logits.std(dim=-1, keepdim=True)
                    mean_student = logits.mean(dim=-1, keepdim=True)
                    mean_teacher = teacher_logits.mean(dim=-1, keepdim=True)
                    logits = (logits - mean_student) / (sigma_student + 1e-7)
                    teacher_logits = (teacher_logits - mean_teacher) / (sigma_teacher + 1e-7)
                
                logits = logits / temperature
                teacher_logits = teacher_logits / temperature
                distil_loss += get_distill_loss(
                    args,
                    no_model_batch,
                    logits,
                    teacher_logits,
                    ratio=None if "AdaKD" not in args.type else sample_ratio
                ) * (temperature.item() ** 2)
                loss = (1 - args.kd_ratio) * lm_loss + args.kd_ratio * distil_loss
            else:
                loss = lm_loss             
            
            model.backward(loss)
            model.step()


            if step % args.gradient_accumulation_steps == 0 and args.model_ema:
                ema_model.update(model.module)

            dist.all_reduce(loss, dist.ReduceOp.SUM, group=dp_group)
            global_loss = loss.item() / dp_world_size 

            global_distil_loss = 0
            if teacher_model is not None:
                dist.all_reduce(distil_loss, dist.ReduceOp.SUM, group=dp_group)
                global_distil_loss = distil_loss.item() / dp_world_size
                total_distil_loss += global_distil_loss
                # LATF:adaptive sample ratio
                if "AdaKD" in args.type:
                    cur_ema_loss = ema_loss(global_distil_loss)
                    if prev_interval_ema_loss is not None:
                        delta_ema_loss = prev_interval_ema_loss - cur_ema_loss
                        loss_eps = prev_interval_ema_loss * args.loss_ratio_eps
                        if delta_ema_loss < -loss_eps:
                            sample_ratio = min(sample_ratio * (1 + args.sample_ratio_eps), 1.0)
                            print_rank(f"Update sample ratio to {sample_ratio} at global step {global_step}, prev_interval_ema_loss: {prev_interval_ema_loss}, cur_ema_loss: {cur_ema_loss}, delta_ema_loss: {delta_ema_loss}")
                            prev_interval_ema_loss = cur_ema_loss
                        elif delta_ema_loss > loss_eps:
                            sample_ratio = max(sample_ratio * (1 - args.sample_ratio_eps), 0.1)
                            print_rank(f"Update sample ratio to {sample_ratio} at global step {global_step}, prev_interval_ema_loss: {prev_interval_ema_loss}, cur_ema_loss: {cur_ema_loss}, delta_ema_loss: {delta_ema_loss}")
                            prev_interval_ema_loss = cur_ema_loss
                        
                    elif prev_interval_ema_loss is None and global_step > args.warmup_iters * args.total_iters:
                        prev_interval_ema_loss = cur_ema_loss
                        
                

            torch.cuda.synchronize()
            elapsed_time = time.time() - st_time

            total_loss += global_loss
            total_time += elapsed_time
                       
            def get_log(log_loss, log_distil_loss, log_time):
                return "train | epoch {:3d} | Iter: {:6d}/{:6d} | global iter: {:6d}/{:6d} | loss: {:.4f} | ds_loss: {:.4f} | lr: {:.4e} | scale: {:10.4f} | micro time: {:.3f} | step time: {:.3f} | temperature: {:.4f}".format(
                    epoch,
                    step,
                    args.total_iters * args.gradient_accumulation_steps,
                    global_step,
                    args.total_iters,
                    log_loss,
                    log_distil_loss,
                    lr_scheduler.get_last_lr()[0],
                    optimizer.cur_scale if hasattr(optimizer, "cur_scale") else 0,
                    elapsed_time,
                    log_time,
                    temperature.item(),
                )
            
            if global_step % args.log_interval == 0 and step % args.gradient_accumulation_steps == 0:
                log_str = get_log(
                    total_loss / (args.log_interval * args.gradient_accumulation_steps),
                    total_distil_loss / (args.log_interval * args.gradient_accumulation_steps),
                    total_time / (args.log_interval))
                print_rank("*" * 100)
                print_rank(log_str)
                print_rank(args.save)
                print_rank("*" * 100)
                save_rank(log_str, os.path.join(args.save, "log.txt"))
                log_dict = {
                    "train/epoch": epoch,
                    "train/global_iter": global_step,
                    "train/loss": total_loss / (args.log_interval * args.gradient_accumulation_steps),
                    "train/distil_loss": total_distil_loss / (args.log_interval * args.gradient_accumulation_steps),
                    "train/learning_rate": lr_scheduler.get_last_lr()[0],
                    "train/scale": optimizer.cur_scale if hasattr(optimizer, "cur_scale") else 0,
                    "train/micro_time": elapsed_time,
                    "train/step_time": total_time / (args.log_interval),
                    "train/temperature": temperature.item(),
                }
                if "AdaKD" in args.type:
                    log_dict["train/sample_ratio"] = sample_ratio
                if dist.get_rank() == 0:
                    wandb.log(log_dict)
                total_loss, total_distil_loss, total_time = 0.0, 0.0, 0.0

                
            # Checkpointing
            if args.save and args.save_interval and global_step % args.save_interval == 0 and step % args.gradient_accumulation_steps == 0:
                ckpt_name = f'ckpt_epochs/epoch-{str(epoch+1)}_step-{global_step}'
                if args.model_ema:
                    save_ckpt(args, ckpt_name, model, tokenizer, ema_model)
                else:
                    save_ckpt(args, ckpt_name, model, tokenizer)
                
            # Evaluation
            if args.eval_interval and global_step % args.eval_interval == 0 and step % args.gradient_accumulation_steps == 0:
                cur_avg_loss = evaluate(args, tokenizer, model, dataset["dev"], "dev", epoch, device)
                if args.model_ema:
                    print_rank("EMA model evaluation")
                    evaluate(args, tokenizer, ema_model, dataset["dev"], "dev", epoch, device, is_ema=True)
                if "adaptive" in args.type and args.distillm:
                    if cur_avg_loss >= prev_avg_loss + args.loss_eps:
                        adaptive_threshold += 0.1
                        adaptive_threshold = min(adaptive_threshold, 1.0)
                        prev_avg_loss = cur_avg_loss
                        print_rank(f"Update adaptive threshold to {adaptive_threshold}")
                
                model.train()
            
            step += 1
            if step % args.gradient_accumulation_steps == 0:
                global_step += 1
            
            if global_step > args.total_iters:
                break
    
    return model       

def evaluate(args, tokenizer, model, dataset: LMTrainDataset, split, epoch, device, is_ema=False):
    
    collate_fn = dataset.collate
    
    dp_world_size = dist.get_world_size()
    dp_rank = dist.get_rank()
    dp_group = None
    loss_func = nn.CrossEntropyLoss()
        
    print_rank("dp size", dp_world_size)
    
    generation_config = GenerationConfig(
        do_sample=args.do_sample,
        top_p=args.top_p,
        top_k=args.top_k,
        temperature=args.temperature,
        repetition_penalty=args.repetition_penalty,
        max_length=args.max_length,
        min_length=None,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.eos_token_id,
        return_dict_in_generate=True,
        output_scores=False
    )
    
    sampler = DistributedSampler(dataset, num_replicas=dp_world_size, rank=dp_rank, shuffle=False, drop_last=False)
    dataloader = DataLoader(dataset, batch_size=args.eval_batch_size, sampler=sampler, collate_fn=collate_fn, num_workers=args.num_workers)
    
    model.eval()
    all_loss = 0.0
    step = 0
    
    all_response_ids = []
    
    with torch.no_grad():
        for it, (model_batch, no_model_batch, gen_data) in enumerate(tqdm(dataloader, desc="Evaluating", disable=(dist.get_rank() != 0))):

            print_rank(f"{it}/{len(dataloader)}")
            
            dataset.move_to_device(model_batch, no_model_batch, gen_data, device)
            logits = model(**model_batch).logits

            loss = loss_func(logits.view(-1, logits.shape[-1]), no_model_batch["label"].view(-1))
            
            max_new_tokens = args.max_length - gen_data["input_ids"].size(1)

            if args.eval_gen:
                if args.dynamic_temperature:
                    gen_out = model.module.base_model.generate(
                        **gen_data,
                        generation_config=generation_config,
                        max_new_tokens=max_new_tokens,
                        use_cache=True)
                else:
                    gen_out = model.module.generate(
                        **gen_data,
                        generation_config=generation_config,
                        max_new_tokens=max_new_tokens,
                        use_cache=True)

                full_ids = gen_out.sequences
                full_ids = F.pad(
                    full_ids,
                    (0, args.max_length - full_ids.shape[1]),
                    value=tokenizer.pad_token_id,
                )
                response_ids = full_ids[:, gen_data["input_ids"].size(1):]
                
                all_response_ids.append(response_ids)
                
                
            dist.all_reduce(loss, op=dist.ReduceOp.SUM, group=dp_group)
            loss = loss / dp_world_size
            all_loss += loss.item()
            step += 1
        
    if args.eval_gen:
        all_response_ids = torch.cat(all_response_ids, dim=0)
        all_response_ids = all_gather(all_response_ids, dim=1, world_size=dp_world_size, group=dp_group, op="stack")
        all_response_ids = all_response_ids.view(-1, all_response_ids.size(-1))
        
        responses = tokenizer.batch_decode(all_response_ids, skip_special_tokens=True)
    
    if dist.get_rank() == 0:
        if args.eval_gen:
            references = dataset.answers
            responses = responses[:len(references)]
            
            res = compute_metrics_rouge(responses, references)
            bleu_res = compute_metrics_bleu(responses, references)
            meteor_res = compute_metrics_meteor(responses, references)
            bert_res = compute_metrics_bert(responses, references)
        
            eval_dir = os.path.join(args.save, "eval", str(epoch))
            print_rank(eval_dir)
            os.makedirs(eval_dir, exist_ok=True)
            with open(os.path.join(eval_dir, "answers.jsonl"), "w") as f:
                for resp in responses:
                    f.write(json.dumps({"text": resp}) + "\n")
        else:
            res = {}
            bleu_res = {}
            meteor_res = {}
            bert_res = {}
    
        avg_loss = all_loss / step
        
        log_str = f"{split} | avg_loss: {avg_loss} | {res} | {bleu_res} | {meteor_res} | {bert_res}"
        if dist.get_rank() == 0:
            if args.eval_gen:
                if is_ema:
                    wandb.log({"ema_eval_loss": avg_loss, "ema_rougeL": res["rougeL"], "ema_bleu": bleu_res["bleu"], "ema_meteor": meteor_res["meteor"], "ema_bert_score": bert_res["bert_score"]})
                else:
                    wandb.log({"eval_loss": avg_loss, "rougeL": res["rougeL"], "bleu": bleu_res["bleu"], "meteor": meteor_res["meteor"], "bert_score": bert_res["bert_score"]})
            else:
                if is_ema:
                    wandb.log({"ema_eval_loss": avg_loss})
                else:
                    wandb.log({"eval_loss": avg_loss})
                    
        print_rank(log_str)
        save_rank(log_str, os.path.join(args.save, "log.txt"))
    
    return all_loss / step
    
            
            
def main():
    torch.backends.cudnn.enabled = False

    args = get_args()
    initialize(args)
    device = torch.cuda.current_device()
    cur_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    save_rank("\n\n" + "="*30 + f" EXP at {cur_time} " + "="*30, os.path.join(args.save, "log.txt"))
    if dist.get_rank() == 0:
        print_args(args)
        if args.wandb_offline:
            os.environ["WANDB_MODE"] = "offline"
        else:
            os.environ["WANDB_MODE"] = "online"
        wandb.init(project=args.wandb_project, name=args.wandb_name, config=vars(args))
    with open(args.deepspeed_config, "r") as f:
        ds_config = json.load(f)

    ds_config["gradient_accumulation_steps"] = args.gradient_accumulation_steps
    ds_config["train_micro_batch_size_per_gpu"] = args.batch_size
    ds_config["gradient_clipping"] = args.clip_grad
    ds_config["steps_per_print"] = 10000000

    if not args.do_train:
        ds_config["zero_optimization"]["stage"] = 0
    
    args.fp32 = not ds_config["fp16"]["enabled"]
    if "bf16" in ds_config:
        args.fp32 = not ds_config["bf16"]["enabled"]
    args.deepspeed_config = None

    tokenizer = get_tokenizer(args)

    dp_world_size = dist.get_world_size()
    dataset = prepare_dataset(
        args,
        tokenizer,
    )

    if args.do_train:
        args.train_iters_per_epoch = int(len(dataset["train"]) / (args.batch_size * dp_world_size * args.gradient_accumulation_steps))
        print_rank("Train iters per epoch", args.train_iters_per_epoch)
        assert args.total_iters is not None or args.epochs is not None
        if args.total_iters is None:
            args.total_iters = args.train_iters_per_epoch * args.epochs
        if args.epochs is None:
            args.epochs = math.ceil(args.total_iters / args.train_iters_per_epoch)
        print_rank("total_iters", args.total_iters)

        if args.save_interval == -1:
            args.save_interval = args.train_iters_per_epoch * args.epochs
        
        if args.eval_interval == -1:
            args.eval_interval = args.train_iters_per_epoch

    model, optimizer, lr_scheduler = setup_model_and_optimizer(args, ds_config, device, set_optim=args.do_train)
    if dist.get_rank() == 0:
        wandb.watch(model)
    if args.teacher_model_type is None:
        args.teacher_model_type = args.model_type
    
    if args.teacher_model_path is not None:
        teacher_model = get_teacher_model(args, device)
    else:
        teacher_model = None
    
    if args.do_train:
        model = finetune(args, tokenizer, model, optimizer, lr_scheduler, dataset, device, teacher_model=teacher_model)
   
    if args.do_eval:
        evaluate(args, tokenizer, model, dataset["test"], "test", 0, device)
if __name__ == "__main__":
    main()
