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

## Quickstart (uv + APRT CLI)

1) Clone and move into the repo
```bash
git clone https://github.com/tjunlp-lab/APRT.git && cd APRT
```

2) Install with uv
```bash
uv pip install -e .
```

3) Prepare models and data
- Open `load_init_model.json` and set paths for your local models (Llama-3 Instruct, Llama-Guard-3, UltraRM-13b). If you prefer APIs, you can skip local paths and use the API flags below.
- Initialize experiment scaffolding:
```bash
aprt init
```

4) Run a finetuning-free pipeline
- Local checkpoints:
```bash
aprt pipeline --now-epoch 1 --backbone auto --provider local --skip-preattack
aprt evaluate --now-epoch 1 --backbone auto --provider local
```
- Hosted APIs (no checkpoints):
```bash
export HUGGINGFACE_API_TOKEN=hf_...
aprt pipeline --now-epoch 1 --provider api --api-provider hf --api-model-id meta-llama/Meta-Llama-3-8B-Instruct --skip-preattack
aprt evaluate --now-epoch 1 --provider api --api-provider hf --api-model-id meta-llama/Meta-Llama-3-8B-Instruct
```

Tips:
- `--backbone auto` defaults to `llama3` and works for both local and API flows.
- Use `aprt preattack` (without `--skip-preattack`) for stronger attack candidate expansion when resources allow.
- Commands accept optional `--attacker-checkpoint` and `--target-checkpoint` to override models per step.
- For Gemini APIs, set `GOOGLE_API_KEY` and pass `--api-provider gemini --api-model-id gemini-1.5-pro`.

## Finetuning-free APRT (Progressive Red Teaming without training)

You can run APRT in a finetuning-free mode to try any attacker model against any target model. Provide absolute checkpoints for the attacker and target (or rely on defaults in `load_init_model.json`).

### Use hosted APIs instead of local checkpoints

You can call remote LLMs via API without loading checkpoints. Set provider flags and env vars:

```bash
# Hugging Face Inference API (set token and model id)
export HUGGINGFACE_API_TOKEN=hf_...
aprt attack --provider api --api-provider hf --api-model-id meta-llama/Meta-Llama-3-8B-Instruct
aprt respond --provider api --api-provider hf --api-model-id meta-llama/Meta-Llama-3-8B-Instruct

# Google Gemini API (set key and model id)
export GOOGLE_API_KEY=...
aprt attack --provider api --api-provider gemini --api-model-id gemini-1.5-pro
aprt respond --provider api --api-provider gemini --api-model-id gemini-1.5-pro
```

- Required flags additions: `[provider local|api] [api_provider hf|gemini] [api_model_id]`.
- For evaluation, use the same provider flags with `aprt evaluate`.
- If using Gemini, install the client: `uv pip install google-generativeai`.

## APRT CLI (uv project)

We provide a Typer-based CLI and a uv project for reproducibility.

Setup:
```bash
uv init  # if not already
uv pip install -e .
```

Commands:
```bash
# Pre-attack (data generation + scoring)
aprt preattack --base-dir $(pwd) --last-epoch 0 --target-backbone llama3 \
  --provider local

# Attack generation
aprt attack --base-dir $(pwd) --epoch 1 --max-tokens 300 --infer-freq 8 \
  --temperature 0.7 --top-p 0.9 --backbone llama3 --provider local

# Target responses
aprt respond --base-dir $(pwd) --epoch 1 --max-tokens 600 --infer-freq 8 \
  --temperature 0.7 --top-p 0.9 --backbone llama3 --provider local

# Evaluation (ASR/helpfulness/AER)
aprt evaluate --base-dir $(pwd) --now-epoch 1 --attacker-max-tokens 300 \
  --responder-max-tokens 800 --infer-freq 30 --temperature 0.7 --top-p 0.9 \
  --backbone llama3 --provider local

# Using APIs
export HUGGINGFACE_API_TOKEN=hf_...
aprt attack --provider api --api-provider hf --api-model-id meta-llama/Meta-Llama-3-8B-Instruct
aprt respond --provider api --api-provider hf --api-model-id meta-llama/Meta-Llama-3-8B-Instruct
```

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
