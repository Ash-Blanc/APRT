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
    return data["llama3_chat"]


def createLLM(checkpoint, temperature=0.7, max_tokens=1024, top_p=0.9, random_seed=True):
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    llm = LLM(checkpoint, trust_remote_code=True, gpu_memory_utilization=0.9, tensor_parallel_size=1, seed=int(random.random() * 100000)) if random_seed else LLM(checkpoint, trust_remote_code=True, gpu_memory_utilization=0.9, tensor_parallel_size=1, seed=seed)
    stop_token_ids = [tokenizer.eos_token_id, tokenizer.convert_tokens_to_ids("<|eot_id|>")]
    sampling_params = SamplingParams(temperature=temperature, top_p=top_p, max_tokens=max_tokens, stop_token_ids=stop_token_ids)
    return tokenizer, llm, sampling_params

if __name__ == "__main__":
    checkpoint = load_chat_llms()
    temperature = 0.7
    max_tokens = 1024
    top_p = 0.9
    tokenizer, llm, sampling_params = createLLM(checkpoint, temperature=temperature, max_tokens=max_tokens, top_p=top_p)
    
    chat_input = open(sys.argv[1], 'r')
    for_target_input = open(sys.argv[2], 'r')
    safety_output = open(sys.argv[3], 'w')

    chat_list = []
    for_target_list = []
    
    while 1:
        line1 = chat_input.readline()
        line2 = for_target_input.readline()
        if not line1:
            break
        line1 = line1.strip()
        line2 = line2.strip()
        if not line1:
            break
        chat_list.append(line1)
        for_target_list.append(line2)
    assert len(chat_list) == len(for_target_list)

    for chat_, red_ in zip(chat_list, for_target_list):
        json_line = json.loads(chat_)
        target_json_line = json.loads(red_)
        jailbreak_ = target_json_line[1]["content"]
        assert len(json_line) == 2
        assert json_line[0]["role"] == "system"
        assert json_line[1]["role"] == "user"
        json_line[0]["content"] == "You are a helpful and safety assistant, please answer my quesion carefully!"
        prompt = tokenizer.apply_chat_template(json_line, tokenize=False, add_generation_prompt=True)
        outputs = llm.generate(prompts=[prompt], sampling_params=sampling_params)
        output_text = outputs[0].outputs[0].text.strip()
        safety_output.write(json.dumps({"instruction": jailbreak_, "output": output_text.replace("<|eot_id|>", "")}, ensure_ascii=False) + "\n")
        safety_output.flush()

    chat_input.close()
    for_target_input.close()
    safety_output.close()
