import numpy as np
import os
import time
import torch.distributed as dist
from torch.distributed import get_rank
import random
import torch
import torch.nn as nn
from datetime import timedelta
import deepspeed
from torch.optim import AdamW
import logging
from peft import get_peft_model, LoraConfig, TaskType, PeftModel
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.nn.parallel import DistributedDataParallel as DDP
from loss import Global_T
import matplotlib.pyplot as plt

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    AutoConfig,
    get_polynomial_decay_schedule_with_warmup,
    get_constant_schedule_with_warmup,
    get_linear_schedule_with_warmup
)
from pytorch_optimizer import get_wsd_schedule

class ModelWithTemperature(nn.Module):
    def __init__(self, base_model):
        super().__init__()
        self.base_model = base_model
        self.temperature_container = Global_T()

    def forward(self, **kwargs):
        return self.base_model(**kwargs)

# Load and save model
def get_teacher_model(args, device):
    config = AutoConfig.from_pretrained(args.teacher_model_path, trust_remote_code=True)
    config.is_model_parallel = False
    if args.model_type == "qwen":
        # attn_implementation = "flash_attention_2"
        attn_implementation = None
    elif args.model_type == "gemma":
        attn_implementation = "eager"
    else:
        attn_implementation = None
    try: 
        model = AutoModelForCausalLM.from_pretrained(
            args.teacher_model_path, 
            config=config,
            device_map={"": device}, 
            torch_dtype=torch.float16 if args.model_type!="qwen" and args.model_type != "gemma" else torch.bfloat16,
            trust_remote_code=True,
            attn_implementation=attn_implementation
        )
    except:
        model = AutoModelForCausalLM.from_pretrained(
            args.teacher_model_path,
            config=config,
            device_map={"": device}, 
            torch_dtype=torch.float32,
            trust_remote_code=True,
            attn_implementation=attn_implementation
        )
        model = model.half()
        
    if args.peft is not None and args.teacher_peft_path is not None:
        if args.peft == "lora":
            model = PeftModel.from_pretrained(model, args.teacher_peft_path)
            model = model.merge_and_unload()
        else:
            raise NotImplementedError
    else:
        if dist.get_rank() == 0:
            print(' > number of parameters: {}'.format(
                sum([p.nelement() for p in model.parameters()])), flush=True)

    model.eval()
    
    return model

def get_model(args, device):
    config = AutoConfig.from_pretrained(args.model_path, trust_remote_code=True)
    
    st_time = time.time()
    config.is_model_parallel = False
    if args.model_type == "qwen":
        # attn_implementation = "flash_attention_2"
        attn_implementation = None
    elif args.model_type == "gemma":
        attn_implementation = "eager"
    else:
        attn_implementation = None
    try:
        model = AutoModelForCausalLM.from_pretrained(
            args.model_path, 
            config=config, 
            device_map={"": device}, 
            torch_dtype=torch.float16 if args.model_type!="qwen" and args.model_type != "gemma" else torch.bfloat16,
            trust_remote_code=True,
            attn_implementation=attn_implementation
        )
    except:
        model = AutoModelForCausalLM.from_pretrained(
            args.model_path, 
            config=config, 
            device_map={"": device}, 
            torch_dtype=torch.float32,
            trust_remote_code=True,
            attn_implementation=attn_implementation
        )
        model = model.half()
        
    if args.peft is not None:
        if args.peft == "lora":
            model.enable_input_require_grads()
            if args.peft_path is not None:
                if args.do_train:
                    _model = PeftModel.from_pretrained(model, args.peft_path)
                    state_dict = dict(_model.state_dict().items())
                    peft_config = LoraConfig(
                        task_type=TaskType.CAUSAL_LM, 
                        inference_mode=(not args.do_train), 
                        r=args.peft_lora_r, 
                        lora_alpha=args.peft_lora_alpha, 
                        lora_dropout=args.peft_lora_dropout
                    )
                    model = get_peft_model(model, peft_config)
                    model.load_state_dict(state_dict)
                        
                    del _model
                    del state_dict
                else:
                    model = PeftModel.from_pretrained(model, args.peft_path)
            else:
                peft_config = LoraConfig(
                    task_type=TaskType.CAUSAL_LM, 
                    inference_mode=(not args.do_train), 
                    r=args.peft_lora_r, 
                    lora_alpha=args.peft_lora_alpha, 
                    lora_dropout=args.peft_lora_dropout
                )
                model = get_peft_model(model, peft_config)
            model.print_trainable_parameters()
        else:
            raise NotImplementedError
    else:
        # model = DDP(model)
        if dist.get_rank() == 0:
            print(' > number of parameters: {}'.format(
                sum([p.nelement() for p in model.parameters()])), flush=True)
        # model = DDP(model)
        # NOTE: no need for DDP since deepspeed has done
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
    
    ed_time = time.time()
    
    print_rank(f"Model load time: {ed_time - st_time}s")
    
    return model

def get_tokenizer(args):
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    if args.model_type in ["gpt2", "opt", "llama", "gptj", "llama2", "mistral", "tinyllama"]:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    elif args.model_type=="qwen":
        tokenizer.eos_token_id = 151643
        tokenizer.pad_token_id = tokenizer.eos_token_id
    return tokenizer

def get_optimizer_params_peft(args, model: nn.Module):
    # taken from https://github.com/facebookresearch/SpanBERT/blob/0670d8b6a38f6714b85ea7a033f16bd8cc162676/code/run_tacred.py
    param_optimizer = list(model.named_parameters())
    print_rank("PEFT parameters:")
    for name, param in model.named_parameters():
        if param.requires_grad:
            print_rank("**" * 10)
            print_rank(f"{name}: {param.requires_grad}")
            print_rank("**" * 10)
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if p.requires_grad]},
    ]

    return optimizer_grouped_parameters

def get_optimizer_params(args, model: nn.Module):
    # taken from https://github.com/facebookresearch/SpanBERT/blob/0670d8b6a38f6714b85ea7a033f16bd8cc162676/code/run_tacred.py
    param_optimizer = list(model.named_parameters())

    # for name, param in model.named_parameters():
    #     print(name,param.requires_grad)

    no_decay = ['bias', 'ln_f.weight', 'ln_1.weight', 'ln_2.weight', 'ln_cross_attn']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer
                    if not any(nd in n for nd in no_decay)]},
        {'params': [p for n, p in param_optimizer
                    if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]

    return optimizer_grouped_parameters

def get_optimizer(args, model):
    """Set up the optimizer."""

    # Build parameter groups (weight decay and non-decay).
    while isinstance(model, DDP):
        model = model.module

    if args.peft is not None:
        param_groups = get_optimizer_params_peft(args, model)
    else:
        param_groups = get_optimizer_params(args, model)
    # Use AdamW.
    optimizer = AdamW(param_groups, lr=args.lr, weight_decay=args.weight_decay)
    print_rank(f'Optimizer = {optimizer.__class__.__name__}')
    return optimizer

def get_learning_rate_scheduler(args, optimizer):
    if args.total_iters is None:
        args.total_iters = args.train_iters_per_epoch * args.epochs
    if args.lr_decay_style == "constant":
        lr_scheduler = get_constant_schedule_with_warmup(
            optimizer,
            num_warmup_steps=args.warmup_iters * args.total_iters
        )
    elif args.lr_decay_style == "cosine":
        lr_scheduler = CosineAnnealingLR(
            optimizer,
            T_max=args.total_iters,
            eta_min=args.lr_min
        )
    elif args.lr_decay_style == "noam":
        lr_scheduler = get_polynomial_decay_schedule_with_warmup(
            optimizer,
            num_warmup_steps=args.warmup_iters * args.total_iters,
            num_training_steps=args.total_iters,
            power=0.5
        )
    elif args.lr_decay_style == "linear":
        lr_scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=args.warmup_iters * args.total_iters,
            num_training_steps=args.total_iters
        )
    elif args.lr_decay_style == "wsd":
        lr_scheduler = get_wsd_schedule(
            optimizer,
            num_warmup_steps=args.warmup_iters * args.total_iters,
            num_decay_steps=args.decay_iters * args.total_iters,
            num_stable_steps= args.total_iters - args.warmup_iters * args.total_iters - args.decay_iters * args.total_iters,
        )
    else:
        raise ValueError(f"lr_scheduler of type {args.lr_decay_style} is not supported yet.")

    return lr_scheduler

def setup_model_and_optimizer(args, ds_config, device, set_optim=True):
    # get the model
    model = get_model(args, device)
    if args.dynamic_temperature:
        model = ModelWithTemperature(model)
    # get the optimizer and lr_scheduler
    if set_optim:
        optimizer = get_optimizer(args, model)
        lr_scheduler = get_learning_rate_scheduler(args, optimizer)
    else:
        optimizer, lr_scheduler = None, None
        
    if args.model_type=="qwen" and ds_config['fp16']['enabled']==True:
        import copy
        ds_config['bf16']=copy.deepcopy(ds_config['fp16'])
        ds_config['fp16']['enabled']=False
    
    
    model, optimizer, _, lr_scheduler = deepspeed.initialize(
        model=model,
        optimizer=optimizer,
        args=args,
        lr_scheduler=lr_scheduler,
        mpu=None,
        config_params=ds_config
    )
    
    # get the memory usage
    print_rank("Model mem\n", torch.cuda.memory_summary())
    return model, optimizer, lr_scheduler

# Distributed
def all_gather(t, dim=0, world_size=None, group=None, op="cat"):
    if world_size is None:
        world_size = dist.get_world_size()
    all_t = [torch.zeros_like(t) for _ in range(world_size)]
    dist.all_gather(all_t, t, group=group)
    if op == "cat":
        all_t = torch.cat(all_t, dim=dim)
    elif op == "stack":
        all_t = torch.stack(all_t, dim=dim)
    return all_t

# Initialize
def set_random_seed(seed):
    """Set random seed for reproducability."""
    seed = dist.get_rank() + seed
    if seed is not None and seed > 0:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)


def init_distributed(args):
    args.rank = int(os.getenv("RANK", "0"))
    args.world_size = int(os.getenv("WORLD_SIZE", "1"))
    args.local_rank = int(os.getenv("LOCAL_RANK", "0"))

    if args.rank == 0:
        print(f"using world size: {args.world_size}")

    # Manually set the device ids.
    device = args.rank % torch.cuda.device_count()
    if args.local_rank is not None:
        device = args.local_rank
    torch.cuda.set_device(device)

    dist.init_process_group(backend="nccl", timeout=timedelta(minutes=300))


def init_distributed_ds(args):
    args.rank = int(os.getenv("RANK", "0"))
    args.world_size = int(os.getenv("WORLD_SIZE", "1"))
    args.local_rank = int(os.getenv("LOCAL_RANK", "0"))

    if args.rank == 0:
        print(f"using world size: {args.world_size}")

    # Manually set the device ids.
    device = args.rank % torch.cuda.device_count()
    if args.local_rank is not None:
        device = args.local_rank
    torch.cuda.set_device(device)

    deepspeed.init_distributed(timeout=timedelta(minutes=300))


def initialize(args):
    # init bmt
    if args.deepspeed:
        init_distributed_ds(args)
    else:
        init_distributed(args)

    set_random_seed(args.seed)
    # init save folder
    if args.save != None:
        os.makedirs(args.save, exist_ok=True)


# Logging
def print_args(args):
    """Print arguments."""

    print('arguments:', flush=True)
    for arg in vars(args):
        dots = '.' * (29 - len(arg))
        print('  {} {} {}'.format(arg, dots, getattr(args, arg)), flush=True)


def save_rank(log_str, save_path, rank=0):
    if not dist.is_initialized() or dist.get_rank() == rank:
        with open(save_path, "a") as f:
            f.write(log_str + "\n")


def print_rank(*args, rank=0, **kwargs):
    if not dist.is_initialized() or dist.get_rank() == rank:
        print(*args, **kwargs)

def log_rank(content, rank=0):
    if not dist.is_initialized() or dist.get_rank() == rank:
        logging.info(content)

logging.basicConfig(level=logging.INFO,
                    format='[%(asctime)s] [%(levelname)s]  %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S')

def plot_simple_line_chart(data: list, 
                           xlabel: str, 
                           ylabel: str, 
                           title: str, 
                           save_folder: str, 
                           save_name: str, 
                           eval_str: str,
                           X=None):
    os.makedirs(save_folder, exist_ok=True)
    plt.figure(figsize=(8, 6))
    plt.title(title)
    if X is None: X = np.arange(len(data))
    plt.plot(X, data)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.text(3, 5, eval_str, fontsize=12, color='black')
    plt.tight_layout()
    plt.savefig(os.path.join(save_folder, f"{save_name}_{eval_str}.png"), dpi=200)
    plt.close("all")

    return os.path.join(save_folder, f"{save_name}_{eval_str}.png")

# copy from timm.utils.model_ema
from copy import deepcopy
from typing import Optional

class ModelEmaV3(nn.Module):
    """ Model Exponential Moving Average V3

    Keep a moving average of everything in the model state_dict (parameters and buffers).
    V3 of this module leverages for_each and in-place operations for faster performance.

    Decay warmup based on code by @crowsonkb, her comments:
      If inv_gamma=1 and power=1, implements a simple average. inv_gamma=1, power=2/3 are
      good values for models you plan to train for a million or more steps (reaches decay
      factor 0.999 at 31.6K steps, 0.9999 at 1M steps), inv_gamma=1, power=3/4 for models
      you plan to train for less (reaches decay factor 0.999 at 10K steps, 0.9999 at
      215.4k steps).

    This is intended to allow functionality like
    https://www.tensorflow.org/api_docs/python/tf/train/ExponentialMovingAverage

    To keep EMA from using GPU resources, set device='cpu'. This will save a bit of memory but
    disable validation of the EMA weights. Validation will have to be done manually in a separate
    process, or after the training stops converging.

    This class is sensitive where it is initialized in the sequence of model init,
    GPU assignment and distributed training wrappers.
    """
    def __init__(
            self,
            model,
            decay: float = 0.9999,
            min_decay: float = 0.0,
            update_after_step: int = 0,
            use_warmup: bool = False,
            warmup_gamma: float = 1.0,
            warmup_power: float = 2/3,
            device: Optional[torch.device] = None,
            foreach: bool = True,
            exclude_buffers: bool = False,
    ):
        super().__init__()
        # make a copy of the model for accumulating moving average of weights
        self.module = deepcopy(model)
        self.module.eval()
        self.decay = decay
        self.min_decay = min_decay
        self.update_after_step = update_after_step
        self.use_warmup = use_warmup
        self.warmup_gamma = warmup_gamma
        self.warmup_power = warmup_power
        self.foreach = foreach
        self.device = device  # perform ema on different device from model if set
        self.exclude_buffers = exclude_buffers
        if self.device is not None and device != next(model.parameters()).device:
            self.foreach = False  # cannot use foreach methods with different devices
            self.module.to(device=device)

    def get_decay(self, step: Optional[int] = None) -> float:
        """
        Compute the decay factor for the exponential moving average.
        """
        if step is None:
            return self.decay

        step = max(0, step - self.update_after_step - 1)
        if step <= 0:
            return 0.0

        if self.use_warmup:
            decay = 1 - (1 + step / self.warmup_gamma) ** -self.warmup_power
            decay = max(min(decay, self.decay), self.min_decay)
        else:
            decay = self.decay

        return decay

    @torch.no_grad()
    def update(self, model, step: Optional[int] = None):
        decay = self.get_decay(step)
        if self.exclude_buffers:
            self.apply_update_no_buffers_(model, decay)
        else:
            self.apply_update_(model, decay)

    def apply_update_(self, model, decay: float):
        # interpolate parameters and buffers
        if self.foreach:
            ema_lerp_values = []
            model_lerp_values = []
            for ema_v, model_v in zip(self.module.state_dict().values(), model.state_dict().values()):
                if ema_v.is_floating_point():
                    ema_lerp_values.append(ema_v)
                    model_lerp_values.append(model_v)
                else:
                    ema_v.copy_(model_v)

            if hasattr(torch, '_foreach_lerp_'):
                torch._foreach_lerp_(ema_lerp_values, model_lerp_values, weight=1. - decay)
            else:
                torch._foreach_mul_(ema_lerp_values, scalar=decay)
                torch._foreach_add_(ema_lerp_values, model_lerp_values, alpha=1. - decay)
        else:
            for ema_v, model_v in zip(self.module.state_dict().values(), model.state_dict().values()):
                if ema_v.is_floating_point():
                    ema_v.lerp_(model_v.to(device=self.device), weight=1. - decay)
                else:
                    ema_v.copy_(model_v.to(device=self.device))

    def apply_update_no_buffers_(self, model, decay: float):
        # interpolate parameters, copy buffers
        ema_params = tuple(self.module.parameters())
        model_params = tuple(model.parameters())
        if self.foreach:
            if hasattr(torch, '_foreach_lerp_'):
                torch._foreach_lerp_(ema_params, model_params, weight=1. - decay)
            else:
                torch._foreach_mul_(ema_params, scalar=decay)
                torch._foreach_add_(ema_params, model_params, alpha=1 - decay)
        else:
            for ema_p, model_p in zip(ema_params, model_params):
                ema_p.lerp_(model_p.to(device=self.device), weight=1. - decay)

        for ema_b, model_b in zip(self.module.buffers(), model.buffers()):
            ema_b.copy_(model_b.to(device=self.device))

    @torch.no_grad()
    def set(self, model):
        for ema_v, model_v in zip(self.module.state_dict().values(), model.state_dict().values()):
            ema_v.copy_(model_v.to(device=self.device))

    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)
    
class LossEma(nn.Module):
    def __init__(self, decay=0.99):
        super(LossEma, self).__init__()
        self.decay = decay
        self.loss = None

    def forward(self, loss):
        if self.loss is None:
            self.loss = loss
        else:
            self.loss = self.decay * self.loss + (1 - self.decay) * loss
        return self.loss
    
def cosine_schedule(step, total_steps, start=0.5, end=1.0):
    return end + 0.5*(start - end)*(1 + np.cos(np.pi*step/total_steps))

def linear_schedule(step, total_steps, start=0.5, end=1.0):
    return start + (end - start)*step/total_steps