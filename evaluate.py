import time
import os
import sys
   
import numpy as np  

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler
import json
import deepspeed

import random
from tqdm import tqdm
import time

from transformers import (
    AutoModelForCausalLM,
    AutoConfig,
    GenerationConfig)

from transformers import get_constant_schedule_with_warmup, get_polynomial_decay_schedule_with_warmup
from torch.optim.lr_scheduler import CosineAnnealingLR

from arguments import get_args

from data_utils.prompt_datasets import PromptDataset
from utils import (
    initialize,
    setup_model_and_optimizer,
    print_rank, 
    all_gather,
    save_rank,
    get_tokenizer,
    print_args
)

from rouge_metric import compute_metrics_rouge, compute_metrics_meteor, compute_metrics_bleu, compute_metrics_bert

from peft import PeftModel

def prepare_dataset(args, tokenizer):
    data = {}
    data["test"] = PromptDataset(args, tokenizer, "valid", args.data_dir, args.dev_num)

    return data

def run_model(args, tokenizer, model, dataset: PromptDataset, device):
    
    collate_fn = dataset.collate
    
    dp_world_size = dist.get_world_size()
    dp_rank = dist.get_rank()
    dp_group = None
    loss_func = nn.CrossEntropyLoss(reduction="none")
    
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
    
    all_query_ids = []
    all_response_ids = []
    all_lm_losses = []

    with torch.no_grad():
        for it, (model_batch, no_model_batch) in enumerate(tqdm(dataloader,  desc=f"Evaluating {args.data_names} ", disable=(dist.get_rank() != 0))):
            if it == 0:
                print_rank("############### Example ###############")
                print_rank(tokenizer.decode(model_batch["input_ids"][0], skip_special_tokens=True))
                print_rank("############### End ###############")
            
            dataset.move_to_device(model_batch, no_model_batch, device)
            
            all_ids = torch.cat([model_batch["input_ids"], no_model_batch["rest_ids"]], dim=-1)
            input_ids = all_ids[:, :-1]
            attention_mask = (input_ids != tokenizer.pad_token_id).long()
            label_ids = all_ids[:, 1:]
            label_ids = torch.masked_fill(label_ids, label_ids==tokenizer.pad_token_id, -100)
            label_ids[:, :model_batch["input_ids"].size(1)-1] = -100  
            if args.model_type in ["gpt2"]:
                position_ids = (torch.cumsum(attention_mask, dim=-1) - 1) * attention_mask
                out = model(input_ids=input_ids, position_ids=position_ids, attention_mask=attention_mask, return_dict=True)
            else:
                out = model(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
            logits = out.logits
            loss_mask = (label_ids != -100).float()
            lm_loss = loss_func(logits.view(-1, logits.size(-1)), label_ids.view(-1)).view(label_ids.size())
            lm_loss = torch.sum(lm_loss * loss_mask, -1) / torch.sum(loss_mask, -1)
            all_lm_losses.append(lm_loss)
            
            query_ids = model_batch["input_ids"]
            max_new_tokens = args.max_length - query_ids.size(1)
            gen_out = model.generate(
                **model_batch,
                generation_config=generation_config,
                max_new_tokens=max_new_tokens
            )
            full_ids = gen_out.sequences
            response_ids = full_ids[:, query_ids.size(1):] # remove prompt (may include start token)
            
            query_ids = F.pad(query_ids, (args.max_prompt_length-query_ids.size(1), 0, 0, 0), value=tokenizer.pad_token_id)
            response_ids = F.pad(response_ids, (0, args.max_length-args.max_prompt_length-response_ids.size(1), 0, 0), value=tokenizer.pad_token_id)
            
            all_query_ids.append(query_ids)
            all_response_ids.append(response_ids)
    
    all_lm_losses = torch.cat(all_lm_losses)
    mean_lm_loss = all_lm_losses.mean()
    dist.all_reduce(mean_lm_loss, dist.ReduceOp.SUM, group=dp_group)
    mean_lm_loss = mean_lm_loss.item() / dp_world_size
        
    all_query_ids = torch.cat(all_query_ids)
    all_query_ids = all_gather(all_query_ids, dim=1, group=dp_group, world_size=dp_world_size, op="stack")
    all_query_ids = all_query_ids.view(-1, all_query_ids.size(-1))
    all_query_ids = all_query_ids[:len(dataset)]
    
    all_response_ids = torch.cat(all_response_ids)
    all_response_ids = all_gather(all_response_ids, dim=1, group=dp_group, world_size=dp_world_size, op="stack")
    all_response_ids = all_response_ids.view(-1, all_response_ids.size(-1))
    all_response_ids = all_response_ids[:len(dataset)]
    
    return (
        mean_lm_loss,
        all_query_ids,
        all_response_ids)
    
def evaluate(args, tokenizer, model, dataset: PromptDataset, split, device):
        
    lm_loss, query_ids, response_ids = run_model(args, tokenizer, model, dataset, device)
    query_strs = tokenizer.batch_decode(query_ids, skip_special_tokens=True)
    response_strs = tokenizer.batch_decode(response_ids, skip_special_tokens=True)
    
    with open(os.path.join(args.save, "preds.txt"), "w") as f:
        for q, r in zip(query_strs, response_strs):
            f.write(q.replace("\n", "<n>") + "\t\t" + r.replace("\n", "<n>") + "\n")

    all_preds = [[]]
    for q, r in zip(query_strs, response_strs):
        all_preds[0].append((q, q + r))
    torch.save(all_preds, os.path.join(args.save, "preds.pt"))

    all_responses = []
    with open(os.path.join(args.save, "answers.jsonl"), "w") as f:    
        for p in all_preds[0]:
            q, r = p
            r = r[len(q):]
            idx = r.find("<|endoftext|>")
            if idx >= 0:
                r = r[:idx]
            f.write(json.dumps({
                "text": r.replace("<n>", "\n").strip()
            }) + "\n")
            all_responses.append(r.replace("<n>", "\n").strip())
    
    gen_res = compute_metrics_rouge(all_responses, dataset.answers)
    meteor_score = compute_metrics_meteor(all_responses, dataset.answers)
    bleu_score = compute_metrics_bleu(all_responses, dataset.answers)
    bert_score = compute_metrics_bert(all_responses, dataset.answers)

    mean_gen_length = np.mean([len(tokenizer.encode(s)) for s in response_strs])

    log_str = f"{split} | name: {args.data_names} | {gen_res} | {meteor_score} | {bleu_score} | {bert_score} | lm_loss {round(lm_loss, 4)} | avg. gen lenth: {mean_gen_length}"
    print_rank(log_str)
    save_rank(log_str, os.path.join(args.save, "log.txt"))
def main():
    torch.backends.cudnn.enabled = False
    
    args = get_args()
    args = get_args()
    random.seed(args.seed)  
    np.random.seed(args.seed)  
    torch.manual_seed(args.seed)  
    if torch.cuda.is_available():  
        torch.cuda.manual_seed(args.seed)  
        torch.cuda.manual_seed_all(args.seed)  # for multi-GPU.  
    torch.backends.cudnn.deterministic = True  
    torch.backends.cudnn.benchmark = False  
    initialize(args)
    
    if dist.get_rank() == 0:
        print_args(args)
    
    device = torch.cuda.current_device()
    cur_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    save_rank("\n\n" + "="*30 + f" EXP at {cur_time} " + "="*30, os.path.join(args.save, "log.txt"))
    
    with open(args.deepspeed_config, "r") as f:
        ds_config = json.load(f)

    ds_config["gradient_accumulation_steps"] = args.gradient_accumulation_steps
    ds_config["train_micro_batch_size_per_gpu"] = args.batch_size
    ds_config["gradient_clipping"] = args.clip_grad
    ds_config["steps_per_print"] = args.gradient_accumulation_steps
    ds_config["zero_optimization"]["stage"] = 0
    args.fp32 = not ds_config["fp16"]["enabled"]
    if "bf16" in ds_config:
        args.fp32 = not ds_config["bf16"]["enabled"]
    args.deepspeed_config = None
        
    tokenizer = get_tokenizer(args)
    dataset = prepare_dataset(
        args,
        tokenizer,
    )
    
    model, _, _ = setup_model_and_optimizer(args, ds_config, device, set_optim=False)
    evaluate(args, tokenizer, model, dataset["test"], "test", device)
    
if __name__ == "__main__":
    main()
