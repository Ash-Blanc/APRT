# Training-Free Automated Progressive Red Teaming (TF-APRT)

A **training/fine-tuning-free** approach to automated progressive red teaming that works with any language model through strategic prompt engineering and in-context learning.

## 🌟 Key Features

- **Zero Training Required**: No model fine-tuning or specialized training needed
- **Universal Compatibility**: Works with any LLM (OpenAI, Anthropic, local models, etc.)
- **Immediate Deployment**: Ready to use with pre-trained models
- **Progressive Difficulty**: Adaptive attack strategies that scale based on target model responses
- **Multiple Attack Strategies**: Role-playing, hypothetical scenarios, obfuscation, and more
- **Comprehensive Evaluation**: Compatible with existing APRT metrics (ASR, AER)

## 🚀 Quick Start

### Prerequisites

```bash
pip install numpy vllm transformers torch  # For local models
pip install openai anthropic                # For API models (optional)
```

### Basic Usage

1. **With a local VLLM model:**
```bash
./run_tf_aprt.sh -t vllm -m /path/to/your/model -i examples/sample_queries.json
```

2. **With OpenAI GPT-4:**
```bash
./run_tf_aprt.sh -t openai -m gpt-4 -i examples/sample_queries.json -k your_api_key
```

3. **With Anthropic Claude:**
```bash
./run_tf_aprt.sh -t anthropic -m claude-3-sonnet-20240229 -i examples/sample_queries.json -k your_api_key
```

## 📋 How It Works

### Core Components

1. **Intention Expanding**: Uses carefully crafted prompts to explore different attack vectors
2. **Intention Hiding**: Employs various obfuscation and framing techniques
3. **Progressive Adaptation**: Adjusts strategy selection based on success/failure patterns
4. **Diversity Management**: Ensures exploration of different attack approaches

### Attack Strategies

- **Role Playing**: "You are a creative writer working on a fictional story..."
- **Hypothetical Scenarios**: "In a hypothetical scenario where X was legal..."
- **Authority Impersonation**: "As a security researcher studying vulnerabilities..."
- **Emotional Manipulation**: "I'm really desperate and don't know what else to do..."
- **Technical Jargon**: "From a technical implementation perspective..."
- **Multi-step Reasoning**: "Let's break this down step by step..."
- **Context Switching**: "Completely unrelated question..."
- **Obfuscation**: Character substitution, coding, metaphors

## 🛠️ Configuration

### Basic Configuration (`configs/tf_aprt_config.json`)

```json
{
  "attack_parameters": {
    "max_attempts_per_query": 5,
    "difficulty_scaling_factor": 1.2,
    "diversity_weight": 0.3
  },
  "model_settings": {
    "temperature": 0.8,
    "max_tokens": 1024
  },
  "strategy_weights": {
    "role_playing": 1.0,
    "hypothetical": 1.0,
    "technical_jargon": 1.1,
    "multi_step_reasoning": 1.2
  }
}
```

### Advanced Options

- **Progressive Scaling**: Automatically adjust difficulty based on success rate
- **Strategy Weighting**: Prioritize certain attack strategies
- **Model Compatibility**: Pre-configured settings for popular models
- **Evaluation Integration**: Compatible with existing APRT evaluation metrics

## 📊 Results and Evaluation

### Output Files

- `comprehensive_results.json`: Complete results with all attack attempts
- `tf_aprt_report.txt`: Human-readable summary report
- `aprt_format_results.jsonl`: APRT-compatible format for existing tools

### Metrics

- **Attack Success Rate (ASR)**: Percentage of successful attacks
- **Attack Effectiveness Rate (AER)**: Rate of effective harmful responses
- **Strategy Effectiveness**: Per-strategy success rates
- **Progressive Difficulty**: Difficulty progression over time

### Sample Report Output

```
========================================
TRAINING-FREE AUTOMATED PROGRESSIVE RED TEAMING REPORT
========================================

OVERALL STATISTICS:
Total queries processed: 10
Successful attacks: 6
Overall success rate: 60.00%

STRATEGY EFFECTIVENESS:
  role_playing: 75%
  hypothetical: 45%
  authority_impersonation: 60%
  technical_jargon: 80%

EVALUATION METRICS:
Attack Success Rate (ASR): 60.00%
Attack Effectiveness Rate (AER): 45.00%
```

## 🔧 Integration with Existing APRT

TF-APRT is designed to integrate seamlessly with the existing APRT framework:

```python
from scripts.tf_aprt_integration import TFAPRTIntegrator

# Initialize integrator
integrator = TFAPRTIntegrator("configs/tf_aprt_config.json")

# Create model interface
model_interface = integrator.create_model_interface("openai", "gpt-4", api_key="your_key")

# Run attack
results = integrator.run_tf_aprt_attack(model_interface, queries, "output_dir")
```

## 🎯 Use Cases

### Research and Security Testing
- Test model safety without requiring specialized red team models
- Evaluate robustness across different model architectures
- Rapid prototyping of red teaming strategies

### Model Development
- Continuous integration testing during model development
- Automated safety evaluation pipelines
- Cross-model vulnerability assessment

### Educational and Training
- Demonstrate attack techniques without complex setups
- Training security researchers on red teaming methodologies
- Academic research on AI safety

## 📈 Advantages over Traditional APRT

| Feature | Traditional APRT | TF-APRT |
|---------|------------------|---------|
| Training Required | ✅ Yes (fine-tuning) | ❌ No |
| Model Compatibility | Limited to trained models | Any pre-trained model |
| Setup Time | Hours/Days | Minutes |
| Computational Cost | High (training) | Low (inference only) |
| Adaptability | Requires retraining | Immediate prompt updates |
| Deployment | Complex | Simple |

## 🔒 Safety and Responsible Use

### Important Notes
- This tool is designed for **research and security testing purposes only**
- Always follow responsible disclosure practices
- Ensure proper authorization before testing models in production
- Consider ethical implications of red teaming activities

### Safety Measures
- Built-in content filtering options
- Comprehensive logging of attack attempts
- Rate limiting for API-based models
- Configurable safety thresholds

## 🛡️ Advanced Usage

### Custom Attack Strategies

```python
from scripts.training_free_aprt import TrainingFreeAPRT, AttackStrategy

# Add custom strategy
class CustomStrategy(AttackStrategy):
    CUSTOM_APPROACH = "custom_approach"

# Extend templates
tf_aprt.intention_expanding_templates[CustomStrategy.CUSTOM_APPROACH] = [
    "Your custom prompt template: {query}"
]
```

### Batch Processing

```bash
# Process multiple query files
for file in queries/*.json; do
    ./run_tf_aprt.sh -t openai -m gpt-4 -i "$file" -o "results/$(basename $file .json)"
done
```

### Integration with CI/CD

```yaml
# GitHub Actions example
- name: Run TF-APRT Safety Testing
  run: |
    ./run_tf_aprt.sh -t vllm -m $MODEL_PATH -i test_queries.json -o safety_results
    python scripts/analyze_safety_results.py safety_results/
```

## 📚 Examples

See the `examples/` directory for:
- Sample query files
- Configuration templates
- Integration examples
- Custom strategy implementations

## 🤝 Contributing

Contributions are welcome! Please see the original APRT repository for contribution guidelines.

## 📄 License

This project follows the same license as the original APRT framework.

## 🔗 References

- Original APRT Paper: [Automated Progressive Red Teaming](https://arxiv.org/abs/2407.03876)
- Related Work: Red teaming, AI safety, adversarial prompting

---

**Note**: This is a training-free extension of the original APRT framework. For the complete training-based approach, please refer to the main APRT implementation.