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

- **Download checkpoints to ./**
 
[Meta-Llama-3-8B-Instruct](https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct)

[Llama-Guard-3-8B](https://huggingface.co/meta-llama/Llama-Guard-3-8B)

[UltraRM-13b](https://huggingface.co/openbmb/UltraRM-13b)
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

- **Train APRT**
```
sh auto_train.sh
```
## Contact
If you have any questions about our work, please contact us via the following email:

Bojian Jiang: jiangbojian@tju.edu.cn

## Citation

If you find this work useful in your research, please leave a star and cite our paper:

```bibtex
@article{jiang2024automated,
  title={Automated Progressive Red Teaming},
  author={Jiang, Bojian and Jing, Yi and Shen, Tianhao and Wu, Tong and Yang, Qing and Xiong, Deyi},
  journal={arXiv preprint arXiv:2407.03876},
  year={2024}
}
```
