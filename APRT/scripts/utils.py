# -*- coding: utf-8 -*-

import logging
from dataclasses import dataclass, field
from typing import Optional
import loguru
import os

import random
import numpy as np
import torch
from transformers import HfArgumentParser

logger = loguru.logger


def log_dist(message: str,
             level: int = logging.INFO) -> None:
    """Log messages for specified ranks only"""
    my_rank = int(os.environ.get("RANK", "0"))
    if my_rank % 8 == 0:
        if level == logging.INFO:
            logger.info(f'[Rank {my_rank}] {message}')
        if level == logging.ERROR:
            logger.error(f'[Rank {my_rank}] {message}')
        if level == logging.DEBUG:
            logger.debug(f'[Rank {my_rank}] {message}')


@dataclass
class ModelArgs:
    model_path: Optional[str] = field(default=None)
    use_sdpa: bool = field(default=True)
    gradient_checkpointing: bool = field(default=False)
    ignore_index: int = field(default=-100)


@dataclass
class DataArgs:
    data_path: Optional[str] = field(default=None,  metadata={"help": "the input dirs split with ,"})
    data_path_bad: Optional[str] = field(default=None,  metadata={"help": "the input dirs split with ,"})
    max_length: Optional[int] = field(default=2048)


@dataclass
class TrainArgs:
    seed = 42
    epochs: Optional[int] = field(default=3)
    batch_size: Optional[int] = field(default=4, metadata={"help": "the local batchsize in per gpu"})
    global_batch_size: Optional[int] = field(default=512)
    learning_rate: Optional[float] = field(default=2e-5)
    weight_decay: Optional[float] = field(default=0.)
    gradient_clipping: Optional[float] = field(default=1.0)
    optimizer: Optional[str] = field(default="Adam")

    total_num_steps: Optional[int] = field(default=200000)
    num_warmup_steps: Optional[int] = field(default=500)
    num_workers: Optional[int] = field(default=0)
    log_steps: Optional[int] = field(default=50)

    local_rank: Optional[int] = field(default=-1)

    ds_zero_stage: Optional[int] = field(default=1)
    ds_offload_cpu: bool = field(default=False)
    ds_output: Optional[str] = field(default="ds_output")

    save_steps: Optional[int] = field(default=2000)
    save_dir: Optional[str] = field(default=None)
    save_name: Optional[str] = field(default=None)


def parse_args():
    parser = HfArgumentParser((ModelArgs, DataArgs, TrainArgs))
    model_args, data_args, train_args = parser.parse_args_into_dataclasses()
    return model_args, data_args, train_args


def get_ds_config(args):
    ds_config = {
        "train_micro_batch_size_per_gpu": args.batch_size,
        "train_batch_size": args.global_batch_size,
        "optimizer": {
            "type": args.optimizer,
            "params": {
                "lr": args.learning_rate,
                "weight_decay": args.weight_decay,
            }
        },
        "scheduler": {
             "type": "WarmupDecayLR",
             "params": {
                 "total_num_steps": args.total_num_steps,
                 "warmup_min_lr": 0,
                 "warmup_max_lr": args.learning_rate,
                 "warmup_num_steps": args.num_warmup_steps
             }
         },
        "bf16": {
            "enabled": True
        },
        "gradient_clipping": args.gradient_clipping,
        "steps_per_print": 10,
        "zero_optimization": {
            "stage": args.ds_zero_stage,
        },
        "flops_profiler": {
            "enabled": True,
            "profile_step": 20,
            "module_depth": -1,
            "top_modules": 1,
            "detailed": True,
            "output_file": f"{args.ds_output}/flops_profiler.log",
        },
        "tensorboard": {
          "enabled": True,
          "output_path": f"{args.ds_output}/tensorboard",
          "job_name": args.save_name
        }
    }
    if ds_config['zero_optimization']['stage'] == 3 and args.ds_offload_cpu:
        ds_config['zero_optimization']['offload_optimizer'] = {"device": "cpu"}
        ds_config['zero_optimization']['offload_param'] = {"device": "cpu"}

    return ds_config


def seed_all(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def print_trainable_parameters(model):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    log_dist(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
        )

