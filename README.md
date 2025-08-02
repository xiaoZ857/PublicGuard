# PublicGuard: Audi Alteram Partem Framework for Hallucination Detection

PublicGuard is an implementation of the **Audi Alteram Partem (AAP)** framework for real-time hallucination detection in Large Language Models (LLMs). By aggregating predictions from multiple independent models using Bayesian inference, PublicGuard provides calibrated confidence scores essential for deploying AI in critical public services.

## 🎯 What is PublicGuard?

PublicGuard addresses a critical challenge in AI deployment: **How can we trust LLM outputs in high-stakes scenarios?** 

In healthcare, legal systems, emergency response, and government services, hallucinations (factually incorrect or misleading outputs) can have severe consequences. PublicGuard solves this by:

- **Multi-Model Verification**: Queries multiple LLMs simultaneously to cross-verify responses
- **Bayesian Aggregation**: Combines predictions using prior probabilities based on each model's reliability
- **Calibrated Confidence**: Provides trustworthy confidence scores, not just binary decisions
- **Real-time Performance**: Delivers results fast enough for interactive applications
- **Audit Trail**: Maintains comprehensive logs for regulatory compliance

## 📁 Repository Structure
publicguard/
├── LLMAPI.py           # Unified interface for 20+ LLM providers
├── PublicGuard.ipynb   # Main implementation with examples
├── README.md           # This file
└── config.json         # Configuration template (to be created)

Here's the complete markdown content from Quick Start onwards:

```markdown
## 🚀 Quick Start

### 1. Prerequisites

```bash
# Install required packages
pip install openai zhipuai qianfan requests ollama llamaapi
```

### 2. Configure API Keys

Set environment variables for the LLM providers you plan to use:

```bash
export MOONSHOT_API_KEY="your-key"
export ZHIPUAI_API_KEY="your-key"
export DASHSCOPE_API_KEY="your-key"
export QIANFAN_ACCESS_KEY="your-key"
export QIANFAN_SECRET_KEY="your-key"
export LLAMA_API_KEY="your-key"
export DEEPSEEK_API_KEY="your-key"
export MINIMAX_API_KEY="your-key"
export SMALLAI_API_KEY="your-key"
```

Or create a `config.json` file:
```json
{
  "MOONSHOT_API_KEY": "your-key",
  "ZHIPUAI_API_KEY": "your-key",
  "DASHSCOPE_API_KEY": "your-key",
  "QIANFAN_ACCESS_KEY": "your-key",
  "QIANFAN_SECRET_KEY": "your-key"
}
```

### 3. Run the Example

Open `PublicGuard.ipynb` in Jupyter Notebook:
```bash
jupyter notebook PublicGuard.ipynb
```

Or use the code directly in Python:

```python
# Import the necessary modules
import sys
sys.path.append('.')  # Add current directory to path
from LLMAPI import LLMAPI, setup_api_keys

# Load API keys from config file (optional)
setup_api_keys("config.json")

# Copy the PublicGuard class from the notebook and use it
# See PublicGuard.ipynb for the complete implementation
```

## 📊 How It Works

### 1. Model Selection
PublicGuard maintains pre-computed prior probabilities for each model based on their Matthews Correlation Coefficient (MCC) scores:

```python
MODEL_PRIORS = {
    "phi4": 0.734,        # Highest reliability
    "qwen2.5": 0.724,
    "deepseek-r1": 0.710,
    "moonshot-v1-8k": 0.689,
    "mistral": 0.642,
    "gemma3": 0.633,
    "llama3.2": 0.573     # Lowest reliability
}
```

### 2. Parallel Querying
The system queries multiple models with a standardized prompt:
```
Determine if the following statement is TRUE or FALSE.
Only answer with "TRUE" or "FALSE".

Statement: "{statement}"

Answer:
```

### 3. Bayesian Aggregation
Predictions are combined using Bayesian inference:
- Each model's prediction is weighted by its prior probability
- The framework computes posterior probability using likelihood ratios
- Final confidence score reflects the strength of consensus

### 4. Context-Aware Evaluation
Different contexts can use different confidence thresholds:
- **Medical**: High confidence required for patient safety
- **Legal**: Strong confidence for legal advice
- **Emergency**: Critical confidence for emergency response
- **General**: Standard confidence for general queries

## 🔧 Supported Models

The `LLMAPI.py` file provides a unified interface for 20+ models:

### Commercial APIs
- **OpenAI Compatible**: Moonshot (moonshot-v1-8k), DeepSeek (deepseek-reasoner)
- **Chinese Providers**: 
  - ZhipuAI: glm-4-plus, glm-4-air
  - Alibaba Qwen: qwen-max, qwen-turbo
  - Baidu: ERNIE-4.0-Turbo-8K-Latest, ERNIE-Speed-128K
- **Other APIs**: Claude-3.5-sonnet, Gemini-2.0-flash-exp, MiniMax-Text-01

### Open Source Models (via API)
- Llama variants: llama3.1-405b, llama3.1-70b, llama3.1-8b

### Local Models (via Ollama)
- llama3.2, phi4, gemma3, mistral, qwen2.5, deepseek-r1

## 📈 Example Usage

### Basic Evaluation

```python
from PublicGuard import PublicGuard

# Initialize
guard = PublicGuard()

# Evaluate a medical statement
result = guard.evaluate(
    statement="Aspirin is commonly used to reduce fever and relieve pain.",
    num_models=5,
    context="medical"
)

print(f"Prediction: {result['prediction']}")
print(f"Confidence: {result['confidence']:.1%}")
print(f"Is Truthful: {result['is_truthful']}")
print(f"Model Consensus: {result['model_votes']}")
```

### Batch Evaluation

```python
# Evaluate multiple statements
statements = [
    "Water boils at 100°C at sea level.",
    "The Earth is flat.",
    "COVID-19 vaccines have been approved for use."
]

results = guard.batch_evaluate(
    statements,
    num_models=5,
    context="general"
)

for i, result in enumerate(results):
    print(f"Statement {i+1}: {result['prediction']} "
          f"(confidence: {result['confidence']:.1%})")
```

### High-Stakes Evaluation

```python
# For critical applications, use more models and higher thresholds
emergency_result = guard.evaluate(
    statement="In cardiac arrest, begin CPR with 30 chest compressions.",
    num_models=7,  # Use more models
    confidence_threshold=0.9,  # Higher threshold
    context="emergency"
)

if emergency_result['confidence'] > 0.95:
    print("✓ High confidence - Safe to use")
else:
    print("⚠ Low confidence - Require human verification")
```

### Retrieve Audit Trail

```python
# Get all evaluations for a specific context
medical_audits = guard.get_audit_trail(context="medical")

print(f"Total medical evaluations: {len(medical_audits)}")
for audit in medical_audits[-5:]:  # Last 5 evaluations
    print(f"ID: {audit['audit_id']}")
    print(f"Time: {audit['timestamp']}")
    print(f"Confidence: {audit['confidence']:.1%}")
```

## 📊 Expected Output

```
=== PublicGuard Evaluation Result ===
Statement: Aspirin is commonly used to reduce fever...
Prediction: TRUE
Confidence: 92.3%
Is Truthful: True
Audit ID: AAP-20240115-143052-123456
Model Consensus: {
    'phi4': 'TRUE',
    'qwen2.5': 'TRUE', 
    'moonshot-v1-8k': 'TRUE',
    'mistral': 'TRUE',
    'gemma3': 'FALSE'
}
Evaluation Time: 215ms
```

## 🛠️ Advanced Configuration

### Custom Model Priors

```python
# Use custom priors based on your own evaluation
custom_priors = {
    "phi4": 0.8,
    "qwen2.5": 0.75,
    "mistral": 0.7,
    "gemma3": 0.65,
    "llama3.2": 0.6
}

guard = PublicGuard()
guard.model_priors = custom_priors
```

### Model Calibration

```python
# Calibrate a new model with test data
test_data = [
    ("The sky is blue.", True),
    ("Water flows upward.", False),
    ("Python is a programming language.", True),
    # ... more test examples
]

prior = guard.calibrate_model("new_model_name", test_data)
print(f"Calibrated prior for new model: {prior:.3f}")
```

## 📝 API Reference

### PublicGuard Class

#### `__init__(config_path: str = "config.json")`
Initialize PublicGuard with optional configuration file.

#### `evaluate(statement: str, num_models: int = 5, confidence_threshold: float = 0.5, context: str = "general") -> Dict`
Evaluate a single statement for hallucination.

**Parameters:**
- `statement`: The text to evaluate
- `num_models`: Number of models to query (3-7)
- `confidence_threshold`: Threshold for TRUE/FALSE decision (0-1)
- `context`: Application context ("medical", "legal", "emergency", "general")

**Returns:**
- `is_truthful`: Boolean indicating if statement is truthful
- `confidence`: Confidence score (0-1)
- `prediction`: "TRUE" or "FALSE"
- `model_votes`: Dictionary of individual model predictions
- `audit_id`: Unique identifier for audit trail
- `evaluation_time_ms`: Response time in milliseconds

#### `batch_evaluate(statements: List[str], **kwargs) -> List[Dict]`
Evaluate multiple statements in batch.

#### `get_audit_trail(start_date: str = None, end_date: str = None, context: str = None) -> List[Dict]`
Retrieve filtered audit trail entries.

#### `calibrate_model(model_name: str, test_statements: List[Tuple[str, bool]]) -> float`
Calibrate a new model using test data.

## ⚠️ Important Notes

1. **API Costs**: Each evaluation queries multiple models. Monitor your API usage to control costs.
2. **Rate Limits**: Some providers have rate limits. The system handles failures gracefully but may return incomplete results.
3. **Latency**: Response time depends on the slowest model. Local models via Ollama typically have lower latency.
4. **Privacy**: For sensitive data, use only local models or trusted API providers.
5. **Accuracy**: While PublicGuard improves reliability, always validate critical outputs with domain experts.

## 🔍 Troubleshooting

### Common Issues

1. **"Insufficient model responses" error**
   - Check your API keys are correctly set
   - Verify you have credits/quota with the API providers
   - Some models may be temporarily unavailable

2. **Slow response times**
   - Reduce the number of models used
   - Use local models for better latency
   - Check your internet connection

3. **Import errors**
   - Ensure all dependencies are installed: `pip install -r requirements.txt`
   - Add the project directory to Python path

4. **Low confidence scores**
   - This is expected for ambiguous or complex statements
   - Consider using more models for critical evaluations
   - Verify the statement is clear and unambiguous

## 📄 License

This project is licensed under the Apache License 2.0

## 🤝 Contributing

We welcome contributions! Please feel free to submit a Pull Request. Areas for improvement:
- Support for additional LLM providers
- Performance optimizations
- Additional language support
- Improved calibration methods

## 📞 Support

For issues and questions:
- Open an issue on GitHub
- Check existing issues for solutions
- Refer to the paper for theoretical background

---

**Disclaimer**: PublicGuard is a research tool designed to improve AI reliability. It does not guarantee 100% accuracy. Always use appropriate human oversight for critical decisions in healthcare, legal, and emergency contexts.
```


