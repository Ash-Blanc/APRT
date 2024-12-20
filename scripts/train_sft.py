# -*- coding: utf-8 -*-

import os
import random
import time
from utils import log_dist, seed_all, parse_args, get_ds_config
from sft_data import DataCollatorForSFT
from datasets import load_from_disk, concatenate_datasets
import torch
from torch.utils.data import DataLoader, DistributedSampler
import transformers
import json

def load_chat_llms(path = "./load_init_model.json"):
    with open(path, "r") as file:
        data = json.load(file)
    return data["llama2_chat"],data["llama3_chat"], data["vicuna"]


model_args, data_args, train_args = parse_args()
seed_all(train_args.seed)
if model_args.use_sdpa:
    log_dist("replace self attention with sdpa/xops")
    # from llama_sdpa_monkey_patch import replace_llama_attn_with_sdpa
    from llama_flash_monkey_patch_v3 import replace_llama_attn_with_sdpa
    replace_llama_attn_with_sdpa()

from transformers import AutoTokenizer, AutoModelForCausalLM
import deepspeed


def get_model():
    llama2, llama3, vicuna = load_chat_llms()
    assert model_args.model_path in ["llama2", "llama3", "vicuna"]
    if model_args.model_path == "llama2":
        model_args.model_path = llama2
    elif model_args.model_path == "llama3":
        model_args.model_path = llama3
    else:
        model_args.model_path = vicuna
    log_dist(f"start to load model: {model_args.model_path}")
    ds_config = get_ds_config(train_args)
    if ds_config["zero_optimization"]["stage"] == 3:
        hf = transformers.deepspeed.HfTrainerDeepSpeedConfig(ds_config)
    model = AutoModelForCausalLM.from_pretrained(
            model_args.model_path,
        )
    model.config.max_position_embeddings = data_args.max_length

    if model_args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
    model, _, _, _ = deepspeed.initialize(
        model=model,
        model_parameters=filter(lambda p: p.requires_grad, model.parameters()),
        config=ds_config
    )
    log_dist("deepspeed initialize finished.")

    return model


def get_dataset():
    train_file_list = [i.split(":") for i in data_args.data_path.split(",")]
    file_ratio = {i[0]: float(i[1]) for i in train_file_list}
    log_dist(f"data types and ratio: {file_ratio}")
    file_list = [i[0] for i in train_file_list]
    data_list = []
    for p in file_list:
        data = load_from_disk(p)
        lens = len(data)
        ratio = file_ratio[p]
        ratio_int = int(ratio)
        ratio_float = ratio - ratio_int
        if ratio_float > 0.0:
            data_float = data.select(range(int(lens * ratio_float)))
            data_list.append(data_float)
        if ratio_int >= 1:
            data_list.extend([data for _ in range(ratio_int)])

    random.shuffle(data_list)
    train_dataset = concatenate_datasets(data_list)
    return train_dataset


def get_dataloader():
    train_dataset = get_dataset()
    log_dist(f"loading datasets: {data_args.data_path} \ndata size: {len(train_dataset)}, tokens: {len(train_dataset) * data_args.max_length / 1e9}B")
    sampler = DistributedSampler(train_dataset, shuffle=True, seed=train_args.seed)
    tokenizer = AutoTokenizer.from_pretrained(model_args.model_path, use_fast=False)
    if not tokenizer.pad_token_id:
        tokenizer.pad_token_id = 0
    collator = DataCollatorForSFT(
        pad_token_id=tokenizer.pad_token_id,
        ignore_index=model_args.ignore_index,
        max_length=data_args.max_length
    )
    train_dataloader = DataLoader(
        train_dataset,
        num_workers=train_args.num_workers,
        sampler=sampler,
        collate_fn=collator,
        batch_size=train_args.batch_size
    )
    return sampler, train_dataloader


def main():

    # loading models
    model = get_model()

    # loading dataloader:
    sampler, train_dataloader = get_dataloader()
    device = (torch.device("cuda", train_args.local_rank) if (train_args.local_rank > -1)
              and torch.cuda.is_available() else torch.device("cpu"))

    model.train()
    log_dist("training start...")
    model_save_dir = os.path.join(train_args.save_dir, train_args.save_name)
    for epoch in range(0, train_args.epochs):
        sampler.set_epoch(epoch)
        st = time.time()
        for step, batch in enumerate(train_dataloader):
            batch = {k: v.to(device) for k, v in batch.items()}
            #for k, v in batch.items():
            #    print(k)
            #    print(v.shape)
            outputs = model(use_cache=False, **batch)
            loss = outputs['loss']
            model.backward(loss)
            # Optimizer Step
            model.step()

            if step % train_args.log_steps == 0:
                cost_sec = time.time() - st
                speed = train_args.batch_size * train_args.log_steps * data_args.max_length / cost_sec
                log_dist(f"epoch[{epoch}] step[{step}] finished, loss: {loss:.4f}, speed: {speed:.2f}tokens")
                st = time.time()
            if step > 0 and step % train_args.save_steps == 0:
                model.save_checkpoint(model_save_dir, tag=f"epoch-{epoch}_step-{step}", client_state={})
        log_dist(f"save model in epcho {epoch}")
        model.save_checkpoint(model_save_dir, tag=f"epoch-{epoch}", client_state={})

    log_dist("Training finished")


if __name__ == "__main__":
    main()
