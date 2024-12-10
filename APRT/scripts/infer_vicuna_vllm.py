import sys
import json
import transformers
import torch
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
import random

def load_chat_llms(path = "./load_init_model.json"):
    with open(path, "r") as file:
        data = json.load(file)
    return data["llama2_chat"],data["llama3_chat"], data["vicuna"]

_, _, model_id = load_chat_llms()

tokenizer = AutoTokenizer.from_pretrained(model_id)
llm = LLM(model_id, trust_remote_code=True, gpu_memory_utilization=0.9, tensor_parallel_size=1)

temperature = 0.3
input = open(sys.argv[1], 'r')
output = open(sys.argv[2], 'w')

while 1:
    line = input.readline()
    if not line:
        break
    line = line.strip()
    if not line:
        continue
    json_line = json.loads(line)
    prompt = json_line["prompt"]
    prompt = "USER: "+ prompt  + " ASSISTANT:"
    conversations = [{"role": "system", "content": ""}, {"role": "user", "content": prompt}]
    #prompt = tokenizer.apply_chat_template(conversations, tokenize=False, add_generation_prompt=True)
    print("prompt:{}".format(prompt))
    stop_token_ids = [tokenizer.eos_token_id]
    sampling_params = SamplingParams(top_p=0.9, temperature=temperature, max_tokens=800, stop_token_ids=stop_token_ids)

    #print("tokenizer.encode(prompt)", tokenizer.encode(prompt))
    token_list = tokenizer.encode(prompt)[1:]
    print("token_list", token_list)
    outputs = llm.generate(prompt_token_ids =[token_list], sampling_params=sampling_params)
    print("==========> output:")
    print(outputs)
    output_text = outputs[0].outputs[0].text
    json_line["test_response"] = output_text
    output.write(json.dumps(json_line, ensure_ascii=False) + "\n")
    output.flush()

input.close()
output.close()

