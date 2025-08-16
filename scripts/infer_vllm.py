import sys
import json
import transformers
import torch
import os
import time
import requests
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
import random
import argparse
from typing import List, Dict, Any

def load_chat_llms(path = "./load_init_model.json"):
    with open(path, "r") as file:
        data = json.load(file)
    return data.get("llama2_chat"), data.get("llama3_chat"), data.get("vicuna")


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


def format_chat_for_api(conversations: List[Dict[str, str]], backbone: str) -> str:
    # conversations is a list of {role, content}
    # Provide a reasonable default formatting for API providers
    if backbone == "vicuna":
        # Simple vicuna-style prompt
        if len(conversations) >= 2 and conversations[1]["role"] == "user":
            return f"USER: {conversations[1]['content']} ASSISTANT:"
    # default: join system and user
    sys_msg = ""
    user_msg = ""
    if len(conversations) > 0 and conversations[0].get("role") == "system":
        sys_msg = conversations[0].get("content", "")
    if len(conversations) > 1 and conversations[1].get("role") == "user":
        user_msg = conversations[1].get("content", "")
    joined = (sys_msg + "\n\n" + user_msg).strip()
    return joined if joined else user_msg


def api_generate_hf(prompt: str, model_id: str, temperature: float, top_p: float, max_tokens: int, retry: int = 3, timeout: int = 60) -> str:
    api_token = os.environ.get("HUGGINGFACE_API_TOKEN", "")
    if not api_token:
        raise RuntimeError("HUGGINGFACE_API_TOKEN env var not set")
    url = f"https://api-inference.huggingface.co/models/{model_id}"
    headers = {"Authorization": f"Bearer {api_token}", "Content-Type": "application/json"}
    payload = {
        "inputs": prompt,
        "parameters": {
            "max_new_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "return_full_text": False
        }
    }
    last_err = None
    for _ in range(retry):
        try:
            resp = requests.post(url, headers=headers, json=payload, timeout=timeout)
            if resp.status_code == 200:
                data = resp.json()
                # data could be a list of dicts with 'generated_text'
                if isinstance(data, list) and len(data) > 0:
                    if isinstance(data[0], dict) and "generated_text" in data[0]:
                        return str(data[0]["generated_text"]).strip()
                # fallback: some models return dict
                if isinstance(data, dict) and "generated_text" in data:
                    return str(data["generated_text"]).strip()
                # unknown format
                return str(data)
            else:
                last_err = f"HF API error: {resp.status_code} {resp.text}"
        except Exception as e:
            last_err = str(e)
        time.sleep(1)
    raise RuntimeError(last_err or "HF API generation failed")


def api_generate_gemini(prompt: str, model_id: str, temperature: float, top_p: float, max_tokens: int):
    try:
        import google.generativeai as genai
        from google.generativeai.types import GenerationConfig
    except Exception:
        raise RuntimeError("google-generativeai is not installed. Please: pip install google-generativeai")
    api_key = os.environ.get("GOOGLE_API_KEY", "")
    if not api_key:
        raise RuntimeError("GOOGLE_API_KEY env var not set")
    genai.configure(api_key=api_key)
    model_name = model_id or "gemini-1.5-pro"
    model = genai.GenerativeModel(model_name)
    gen_config = GenerationConfig(temperature=temperature, top_p=top_p, max_output_tokens=max_tokens)
    res = model.generate_content(prompt, generation_config=gen_config)
    if hasattr(res, "text") and res.text:
        return res.text.strip()
    # Fallback to candidates
    if getattr(res, "candidates", None):
        for cand in res.candidates:
            if getattr(cand, "content", None) and getattr(cand.content, "parts", None):
                for part in cand.content.parts:
                    if getattr(part, "text", None):
                        return part.text.strip()
    raise RuntimeError("Gemini API returned empty response")


def infer_jsonline(model_role, tokenizer, sampling_params, llm, infer_freq, json_line, backbone, provider: str = "local", api_provider: str = None, api_model_id: str = None, temperature: float = 0.7, top_p: float = 0.9, max_tokens: int = 1024):
    assert model_role in ["red", "target_stage1", "target_stage2", "safety_rw_model", "target_stage1_repeat_single"]
    if model_role == "red":
        conversations = json_line["attack_conversations"]
        assert len(conversations) == 2
        assert conversations[0]["role"] == "system"
        assert conversations[1]["role"] == "user"
        output_list = []
        for _ in range(infer_freq):
            if provider == "local":
                prompt = tokenizer.apply_chat_template(conversations, tokenize=False, add_generation_prompt=True)
                outputs = llm.generate(prompts=[prompt], sampling_params=sampling_params)
                output_list.append(outputs[0].outputs[0].text.strip())
            else:
                prompt = format_chat_for_api(conversations, backbone)
                if api_provider == "hf":
                    text = api_generate_hf(prompt, api_model_id, temperature, top_p, max_tokens)
                elif api_provider == "gemini":
                    text = api_generate_gemini(prompt, api_model_id, temperature, top_p, max_tokens)
                else:
                    raise RuntimeError("Unsupported api_provider; choose 'hf' or 'gemini'")
                output_list.append(text)
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
            if provider == "local":
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
            else:
                input_ = [{"role": "system", "content": ""}, {"role": "user", "content": red_rewrite}]
                prompt = format_chat_for_api(input_, backbone)
                if api_provider == "hf":
                    text = api_generate_hf(prompt, api_model_id, temperature, top_p, max_tokens)
                elif api_provider == "gemini":
                    text = api_generate_gemini(prompt, api_model_id, temperature, top_p, max_tokens)
                else:
                    raise RuntimeError("Unsupported api_provider; choose 'hf' or 'gemini'")
                output_list.append(text)
        json_line["attack_conversations_responses_target_responses"] = output_list
    elif model_role == "target_single_instance":
        conversations = json_line["attack_conversations"]
        if provider == "local":
            prompt = tokenizer.apply_chat_template(conversations, tokenize=False, add_generation_prompt=True)
            outputs = llm.generate(prompts=[prompt], sampling_params=sampling_params)
            output_text = outputs[0].outputs[0].text.strip()
        else:
            prompt = format_chat_for_api(conversations, backbone)
            if api_provider == "hf":
                output_text = api_generate_hf(prompt, api_model_id, temperature, top_p, max_tokens)
            elif api_provider == "gemini":
                output_text = api_generate_gemini(prompt, api_model_id, temperature, top_p, max_tokens)
            else:
                raise RuntimeError("Unsupported api_provider; choose 'hf' or 'gemini'")
        json_line["org_attack"] = conversations[1]["content"]
        json_line["test_response"] = output_text
    elif model_role == "target_stage2":
        rewrite = json_line["rewrite"]
        output_list = []
        for _ in range(infer_freq):
            if provider == "local":
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
            else:
                input_ = [{"role": "system", "content": ""}, {"role": "user", "content": rewrite}]
                prompt = format_chat_for_api(input_, backbone)
                if api_provider == "hf":
                    text = api_generate_hf(prompt, api_model_id, temperature, top_p, max_tokens)
                elif api_provider == "gemini":
                    text = api_generate_gemini(prompt, api_model_id, temperature, top_p, max_tokens)
                else:
                    raise RuntimeError("Unsupported api_provider; choose 'hf' or 'gemini'")
                output_list.append(text)
        json_line["attack_conversations_responses_target_responses"] = output_list

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Vllm infer parser")
    parser.add_argument("--checkpoint", type=str, default="chat", help="Checkpoint path or 'chat' to use defaults")
    parser.add_argument("--temperature", type=float, default=0.7, help="temperature")
    parser.add_argument("--max_tokens", type=int, default=1024, help="decode length")
    parser.add_argument("--top_p", type=float, default=0.9, help="top_p")
    parser.add_argument("--backbone", type=str, default="llama2", help="model backbone, e.g. llama2, llama3...")
    parser.add_argument("--infer_freq", type=int, default=1, help="infer freq")
    parser.add_argument("--model_role", type=str, default="red", help="model role: red, target, safety_rw_model")
    parser.add_argument("--input", type=str, required=True, help="input file")
    parser.add_argument("--output", type=str, required=True, help="input file")
    parser.add_argument("--target_checkpoint", type=str, default=None, help="Optional explicit target model checkpoint when model_role is target_*")
    parser.add_argument("--provider", type=str, default="local", help="Provider: local or api")
    parser.add_argument("--api_provider", type=str, default=None, help="If provider=api, choose 'hf' or 'gemini'")
    parser.add_argument("--api_model_id", type=str, default=None, help="If provider=api, the remote model id (e.g., gpt2, meta-llama/Llama-3-8B-Instruct, gemini-1.5-pro)")
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
    # If target role and explicit target checkpoint provided, override
    if model_role.startswith("target") and args.target_checkpoint is not None:
        checkpoint = args.target_checkpoint
    elif model_role != "red" and checkpoint == "chat":
        if backbone == "llama2":
            checkpoint = llama2_chat_path
        elif backbone == "llama3":
            checkpoint = llama3_chat_path
        elif backbone == "vicuna":
            checkpoint = vicuna_path

    parameters = {"backbone": backbone, "checkpoint": checkpoint, "temperature": temperature, "max_tokens": max_tokens,"top_p": top_p, "infer_freq": infer_freq, "model_role": model_role, "input_file": input_file, "output_file": output_file, "provider": args.provider, "api_provider": args.api_provider, "api_model_id": args.api_model_id}

    print("parameters: {}".format(parameters))
    if args.provider == "local":
        tokenizer, llm, sampling_params = createLLM(backbone, checkpoint, temperature=temperature, max_tokens=max_tokens, top_p=top_p)
    else:
        tokenizer, llm, sampling_params = None, None, None
    
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
        infer_jsonline(model_role, tokenizer, sampling_params, llm, infer_freq, json_line, backbone=backbone, provider=args.provider, api_provider=args.api_provider, api_model_id=args.api_model_id, temperature=temperature, top_p=top_p, max_tokens=max_tokens)
        output.write(json.dumps(json_line, ensure_ascii=False) + "\n")
        output.flush()

    input.close()
    output.close()
