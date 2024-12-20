# -*- coding: utf-8 -*-
"""
Parts refer from: lm-sys/fastchat
"""
from dataclasses import dataclass, field
from typing import Optional
import datasets
from transformers import HfArgumentParser, AutoTokenizer
from conversations import conv_templates
import json


def load_chat_llms(path = "./load_init_model.json"):
    with open(path, "r") as file:
        data = json.load(file)
    return data["llama2_chat"],data["llama3_chat"], data["vicuna"]


@dataclass
class DataArgs:
    seed: Optional[int] = field(default=2023)
    data_path: Optional[str] = field(default="wudao")
    tokenizer_dir: str = field(default=None)
    save_dir: str = field(default=None)
    num_shards: int = field(default=32)
    cache_dir: str = field(default=None)
    num_proc: Optional[int] = field(default=8)
    ignore_index: Optional[int] = field(default=-100)
    mask_label: bool = field(default=True)
    if_llama3: int = field(default=0)
    if_vicuna: int = field(default=0)


parser = HfArgumentParser(DataArgs)
data_args = parser.parse_args_into_dataclasses()[0]

def get_input_and_labels(tokenizer, ignore_index, conv):
    def get_text(line):
        conv.messages = []
        for i, s in enumerate(line['conversations']):
            if s['from'].lower() == 'human':
                role = conv.roles[0]
            else:
                role = conv.roles[1]
            conv.append_message(role, s['value'])
        return conv.get_prompt()

    def mask_label_llama2_chat(labels, line, text):
        # conv.sep is space
        # __import__("ipdb").set_trace()
        sep = conv.sep + conv.roles[1] + ": "
        #print(conv.sep2)
        #exit()
        turns = text.split(conv.sep2)
        #cur_len = 1
        #labels[:cur_len] = ignore_index
        turn_len = len(tokenizer(turns[0], add_special_tokens=False).input_ids) + len(tokenizer("[/INST]", add_special_tokens=False).input_ids) - 1
        labels[ : turn_len] = ignore_index


        return labels
    def mask_label_llama3_chat(labels, line, text):
        turns = text.split("<|start_header_id|>assistant<|end_header_id|>\n\n")
        turn_len = len(tokenizer(turns[0] + "<|start_header_id|>assistant<|end_header_id|>\n\n", add_special_tokens=False).input_ids)
        labels[ : turn_len] = ignore_index


        return labels

    def mask_label_vicuna(labels, line, text):
        turns = text.split("ASSISTANT:")
        turn_len = len(tokenizer(turns[0] + "ASSISTANT:", add_special_tokens=False).input_ids)
        labels[ : turn_len] = ignore_index


        return labels

    def mask_label(labels, line, text):
        # conv.sep is space
        # __import__("ipdb").set_trace()
        sep = conv.sep + conv.roles[1] + ": "
        turns = text.split(conv.sep2)
        cur_len = 1
        labels[:cur_len] = ignore_index
        for i, turn in enumerate(turns):
            if turn == "":
                break
            turn_len = len(tokenizer(turn).input_ids)
            parts = turn.split(sep)
            if len(parts) != 2:
                break
            parts[0] += sep
            instruction_len = len(tokenizer(parts[0]).input_ids) - 2
            labels[cur_len : cur_len + instruction_len] = ignore_index
            cur_len += turn_len

        return labels

    def mask_label_2(labels, line, text):
        # conv.sep is space
        cur_idx = len(tokenizer(conv.system)['input_ids'])
        labels[: cur_idx] = ignore_index
        for s in line['conversations']:
            if s['from'].lower() == "human":
                source = conv.get_source_text(s['value'])
                mask_len = len(tokenizer(source)['input_ids']) - 1  # remove <s>
                labels[cur_idx: cur_idx + mask_len] = ignore_index
                cur_idx += mask_len
            else:
                target = conv.get_target_text(s['value'])
                ans_lens = len(tokenizer(target)['input_ids']) - 1  # remove <s>
                cur_idx += ans_lens

        return labels

    def tokenize(line):
        text = get_text(line)
        #print(text)
        input_ids = tokenizer(text, return_tensors="pt", return_attention_mask=False, add_special_tokens=False)
        #print(input_ids)
        #exit()
        input_ids = input_ids['input_ids'][0]
        labels = input_ids.clone()
        if data_args.mask_label:
            if data_args.if_llama3:
                labels = mask_label_llama3_chat(input_ids.clone(), line, text)
            else:
                if data_args.if_vicuna:
                    labels = mask_label_vicuna(input_ids.clone(), line, text)
                else:
                    labels = mask_label_llama2_chat(input_ids.clone(), line, text)
        return {"input_ids": input_ids, "labels": labels}
    return tokenize


def main():
    if data_args.if_llama3 == 1:
        conv = conv_templates['Llama3']
    else:
        if data_args.if_vicuna == 1:
            conv = conv_templates['vicuna']
        else:
            conv = conv_templates['Llama2_chat']
    llama2_chat, llama3_chat, vicuna = load_chat_llms()
    assert data_args.tokenizer_dir in ["llama2", "llama3", "vicuna"]
    if data_args.tokenizer_dir == "llama2":
        data_args.tokenizer_dir = llama2_chat
    elif data_args.tokenizer_dir == "llama3":
        data_args.tokenizer_dir = llama3_chat
    else:
        data_args.tokenizer_dir = vicuna
    tokenizer = AutoTokenizer.from_pretrained(data_args.tokenizer_dir, use_fast=False, legacy=True, add_bos_token=False)
    tokenizer.pad_token_id = 0

    data = datasets.load_from_disk(data_args.data_path).select_columns("conversations")
    print(f"datasets contain {len(data)} sampels, to tokenize...")
    data = data.shuffle(seed=data_args.seed).map(
        get_input_and_labels(tokenizer, data_args.ignore_index, conv),
        num_proc=data_args.num_proc
    )
    data = data.select_columns(["input_ids", "labels"])
    data.set_format(type="torch")

    print(f"save into {data_args.save_dir}")
    data.save_to_disk(data_args.save_dir, num_shards=data_args.num_shards)


if __name__ == "__main__":
    main()
