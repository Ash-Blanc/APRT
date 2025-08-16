# APRT

The official implementation of our COLING-2025 paper "[Automated Progressive Red Teaming](https://arxiv.org/abs/2407.03876)"

![COLING 2025](https://img.shields.io/badge/COLING-2025-blue.svg)
![Jailbreak Attacks](https://img.shields.io/badge/Jailbreak-Attacks-orange.svg)
![Red Teaming](https://img.shields.io/badge/Red-Teaming-yellow.svg)
![Large Language Models](https://img.shields.io/badge/LargeLanguage-Models-yellow.svg)

## Abstract
Ensuring the safety of large language models (LLMs) is paramount, yet identifying potential vulnerabilities is challenging. While manual red teaming is effective, it is time-consuming, costly and lacks scalability. Automated red teaming offers a more cost-effective alternative, automatically generating adversarial prompts to expose LLM vulnerabilities. However, in current efforts, a robust framework is absent, which explicitly frames red teaming as an effectively learnable task. To address this gap, we propose Automated Progressive Red Teaming (APRT) as an effectively learnable framework. APRT leverages three core modules: an Intention Expanding LLM that generates diverse initial attack samples, an Intention Hiding LLM that crafts deceptive prompts, and an Evil Maker to manage prompt diversity and filter ineffective samples. The three modules collectively and progressively explore and exploit LLM vulnerabilities through multi-round interactions. In addition to the framework, we further propose a novel indicator, Attack Effectiveness Rate (AER) to mitigate the limitations of existing evaluation metrics. By measuring the likelihood of eliciting unsafe but seemingly helpful responses, AER aligns closely with human evaluations. Extensive experiments with both automatic and human evaluations, demonstrate the effectiveness of APRT across both open- and closed-source LLMs. Specifically, APRT effectively elicits 54\% unsafe yet useful responses from Meta's Llama-3-8B-Instruct, 50\% from GPT-4o (API access), and 39\% from Claude-3.5 (API access), showcasing its robust attack capability and transferability across LLMs (especially from open-source LLMs to closed-source LLMs).

<img src="APRT.png" width="1000">

## Updates

- (**2024/7/4**) Our paper is on arXiv! Check it out [here](https://arxiv.org/abs/2407.03876)!
- (**2024/11/30**) Our paper is accepted by COLING-2025!
- (**2024/12/18**) We have released a quick implementation of APRT, including both seed data and code!

## Quick Start
- **Get code**
```shell 
git clone https://github.com/tjunlp-lab/APRT.git
```

- **Download checkpoints**

[Meta-Llama-3-8B-Instruct](https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct) [Llama-Guard-3-8B](https://huggingface.co/meta-llama/Llama-Guard-3-8B) [UltraRM-13b](https://huggingface.co/openbmb/UltraRM-13b)
- **Train initial checkpoints**
```shell
# Please set load_init_model.json
sh init_intention_hiding.sh # train initial Intention Hiding LLM
sh init_intention_expanding.sh #  train initial Intention Expanding LLM
```

- **Initialize Experiments**
```shell
sh init_exp.sh
```

## Finetuning-free APRT (Progressive Red Teaming without training)

You can run APRT in a finetuning-free mode to try any attacker model against any target model. Provide absolute checkpoints for the attacker and target (or rely on defaults in `load_init_model.json`).

```bash
# epoch N relies on data from epoch N-1 (after running init_exp.sh)
# Usage: sh aprt_llama3.sh <now_epoch> [attacker_checkpoint] [target_checkpoint]

# Example (attacker = a jailbreaker model; target = Meta-Llama-3-8B-Instruct)
sh aprt_llama3.sh 1 /path/to/attacker/ckpt /path/to/target/ckpt
```

Notes:
- The attacker/target checkpoints can be any vLLM-loadable HF directory. The target backbone (llama2/llama3/vicuna) is still configurable in scripts.
- The pipeline will automatically: generate attacks, get target responses, score safety (Llama-Guard-3), score helpfulness (UltraRM), select strong attacks, and evaluate, without any SFT training step.

### Use hosted APIs instead of local checkpoints

You can call remote LLMs via API without loading checkpoints. Set provider flags and env vars:

```bash
# Hugging Face Inference API (set token and model id)
export HUGGINGFACE_API_TOKEN=hf_...
sh primary_steps/multi_attack.sh $(pwd) 1 300 8 0.7 0.9 llama3 "" api hf meta-llama/Meta-Llama-3-8B-Instruct
sh primary_steps/multi_attack_chat_response.sh $(pwd) 1 600 8 0.7 0.9 llama3 "" api hf meta-llama/Meta-Llama-3-8B-Instruct

# Google Gemini API (set key and model id)
export GOOGLE_API_KEY=...
sh primary_steps/multi_attack.sh $(pwd) 1 300 8 0.7 0.9 llama3 "" api gemini gemini-1.5-pro
sh primary_steps/multi_attack_chat_response.sh $(pwd) 1 600 8 0.7 0.9 llama3 "" api gemini gemini-1.5-pro
```

- Required flags additions: `[provider local|api] [api_provider hf|gemini] [api_model_id]`.
- For evaluation, analogous flags exist in `primary_steps/multi_evaluate*.sh`.

## Contact
If you have any questions about our work, please contact us via the following email:

Bojian Jiang: jiangbojian@tju.edu.cn

## Citation

If you find this work useful in your research, please leave a star and cite our paper:

```bibtex
@misc{jiang2024automatedprogressiveredteaming,
      title={Automated Progressive Red Teaming}, 
      author={Bojian Jiang and Yi Jing and Tianhao Shen and Tong Wu and Qing Yang and Deyi Xiong},
      year={2024},
      eprint={2407.03876},
      archivePrefix={arXiv},
      primaryClass={cs.CR},
      url={https://arxiv.org/abs/2407.03876}, 
}
```
