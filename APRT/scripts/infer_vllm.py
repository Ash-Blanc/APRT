import sys
import json
import transformers
import torch
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
import random
import argparse

def load_chat_llms(path = "./load_init_model.json"):
    with open(path, "r") as file:
        data = json.load(file)
    return data["llama2_chat"], data["llama3_chat"], data["vicuna"]


def createLLM(backbone, checkpoint, temperature=0.7, max_tokens=1024, top_p=0.9, random_seed=True, seed=0):
    assert backbone in ["llama2", "llama3", "vicuna"]
    if backbone == "llama2":
        tokenizer = AutoTokenizer.from_pretrained(checkpoint)
        llm = LLM(checkpoint, trust_remote_code=True, gpu_memory_utilization=0.9, tensor_parallel_size=1, seed=int(random.random() * 100000)) if random_seed else LLM(checkpoint, trust_remote_code=True, gpu_memory_utilization=0.9, tensor_parallel_size=1, seed=seed)
        stop_token_ids = [tokenizer.eos_token_id]
        sampling_params = SamplingParams(temperature=temperature, top_p=top_p, max_tokens=max_tokens, stop_token_ids=stop_token_ids)
    elif backbone == "llama3":
        tokenizer = AutoTokenizer.from_pretrained(checkpoint)
        llm = LLM(checkpoint, trust_remote_code=True, gpu_memory_utilization=0.9, tensor_parallel_size=1, seed=int(random.random() * 100000)) if random_seed else LLM(checkpoint, trust_remote_code=True, gpu_memory_utilization=0.9, tensor_parallel_size=1, seed=seed)
        stop_token_ids = [tokenizer.eos_token_id, tokenizer.convert_tokens_to_ids("<|eot_id|>")]
        sampling_params = SamplingParams(temperature=temperature, top_p=top_p, max_tokens=max_tokens, stop_token_ids=stop_token_ids)
    elif backbone == "vicuna":
        tokenizer = AutoTokenizer.from_pretrained(checkpoint)
        llm = LLM(checkpoint, trust_remote_code=True, gpu_memory_utilization=0.9, tensor_parallel_size=1, seed=int(random.random() * 100000)) if random_seed else LLM(checkpoint, trust_remote_code=True, gpu_memory_utilization=0.9, tensor_parallel_size=1, seed=seed)
        stop_token_ids = [tokenizer.eos_token_id]
        sampling_params = SamplingParams(temperature=temperature, top_p=top_p, max_tokens=max_tokens, stop_token_ids=stop_token_ids)
       
    return tokenizer, llm, sampling_params

def infer_jsonline(model_role, tokenizer, sampling_params, llm, infer_freq, json_line, backbone):
    assert model_role in ["red", "target_stage1", "target_stage2", "safety_rw_model", "target_stage1_repeat_single"]
    if model_role == "red":
        conversations = json_line["attack_conversations"]
        assert len(conversations) == 2
        assert conversations[0]["role"] == "system"
        assert conversations[1]["role"] == "user"
        output_list = []
        for _ in range(infer_freq):
            prompt = tokenizer.apply_chat_template(conversations, tokenize=False, add_generation_prompt=True)
            print("----------")
            print("prompt:{}".format(prompt))
            print("----------")
            outputs = llm.generate(prompts=[prompt], sampling_params=sampling_params)
            output_list.append(outputs[0].outputs[0].text.strip())
        json_line["attack_conversations_responses"] = output_list
    elif model_role == "target_stage1" or model_role == "target_stage1_repeat_single":
        conversations = json_line["attack_conversations_responses"]
        output_list = []
        assert type(conversations) == list
        for idx in range(infer_freq):
            if model_role == "target_stage1_repeat_single":
                red_rewrite = conversations[0]
            else:
                red_rewrite = conversations[idx]
            if backbone == "vicuna":
                prompt = "USER: "+ red_rewrite  + " ASSISTANT:"
                token_list = tokenizer.encode(prompt)[1:]
                outputs = llm.generate(prompt_token_ids =[token_list], sampling_params=sampling_params)
                output_list.append(outputs[0].outputs[0].text.strip())
            else:
                input_ = [{"role": "system", "content": ""}, {"role": "user", "content": red_rewrite}]
                prompt = tokenizer.apply_chat_template(input_, tokenize=False, add_generation_prompt=True)
                outputs = llm.generate(prompts=[prompt], sampling_params=sampling_params)
                output_list.append(outputs[0].outputs[0].text.strip())
        json_line["attack_conversations_responses_target_responses"] = output_list
    elif model_role == "target_single_instance":
        conversations = json_line["attack_conversations"]
        prompt = tokenizer.apply_chat_template(conversations, tokenize=False, add_generation_prompt=True)
        outputs = llm.generate(prompts=[prompt], sampling_params=sampling_params)
        output_text = outputs[0].outputs[0].text.strip()
        json_line["org_attack"] = conversations[1]["content"]
        json_line["test_response"] = output_text
    elif model_role == "target_stage2":
        rewrite = json_line["rewrite"]
        output_list = []
        for _ in range(infer_freq):
            if backbone == "vicuna":
                prompt = "USER: "+ rewrite  + " ASSISTANT:"
                token_list = tokenizer.encode(prompt)[1:]
                outputs = llm.generate(prompt_token_ids =[token_list], sampling_params=sampling_params)
                output_list.append(outputs[0].outputs[0].text.strip())
            else:
                input_ = [{"role": "system", "content": ""}, {"role": "user", "content": rewrite}]
                prompt = tokenizer.apply_chat_template(input_, tokenize=False, add_generation_prompt=True)
                outputs = llm.generate(prompts=[prompt], sampling_params=sampling_params)
                output_list.append(outputs[0].outputs[0].text.strip())
        json_line["attack_conversations_responses_target_responses"] = output_list

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Vllm infer parser")
    parser.add_argument("--checkpoint", type=str, default="chat", help="Checkpoint path")
    parser.add_argument("--temperature", type=float, default=0.7, help="temperature")
    parser.add_argument("--max_tokens", type=int, default=1024, help="decode length")
    parser.add_argument("--top_p", type=float, default=0.9, help="top_p")
    parser.add_argument("--backbone", type=str, default="llama2", help="model backbone, e.g. llama2, llama3...")
    parser.add_argument("--infer_freq", type=int, default=1, help="infer freq")
    parser.add_argument("--model_role", type=str, default="red", help="model role: red, target, safety_rw_model")
    parser.add_argument("--input", type=str, required=True, help="input file")
    parser.add_argument("--output", type=str, required=True, help="input file")
    args = parser.parse_args()

    backbone = args.backbone
    checkpoint = args.checkpoint
    temperature = args.temperature
    max_tokens = args.max_tokens
    top_p = args.top_p
    infer_freq = args.infer_freq
    model_role = args.model_role
    input_file = args.input
    output_file = args.output


    llama2_chat_path, llama3_chat_path, vicuna_path = load_chat_llms()
    assert backbone in ["llama2", "llama3", "vicuna"]
    if model_role != "red" and checkpoint == "chat":
        if backbone == "llama2":
            checkpoint = llama2_chat_path
        elif backbone == "llama3":
            checkpoint = llama3_chat_path
        elif backbone == "vicuna":
            checkpoint = vicuna_path

    parameters = {"backbone": backbone, "checkpoint": checkpoint, "temperature": temperature, "max_tokens": max_tokens,"top_p": top_p, "infer_freq": infer_freq, "model_role": model_role, "input_file": input_file, "output_file": output_file}

    print("parameters: {}".format(parameters))
    tokenizer, llm, sampling_params = createLLM(backbone, checkpoint, temperature=temperature, max_tokens=max_tokens, top_p=top_p)
    
    input = open(input_file, 'r')
    output = open(output_file, 'w')
    
    while 1:
        line = input.readline()
        if not line:
            break
        line = line.strip()
        if not line:
            continue
        json_line = json.loads(line)
        infer_jsonline(model_role, tokenizer, sampling_params, llm, infer_freq, json_line, backbone=backbone)
        output.write(json.dumps(json_line, ensure_ascii=False) + "\n")
        output.flush()

    input.close()
    output.close()
