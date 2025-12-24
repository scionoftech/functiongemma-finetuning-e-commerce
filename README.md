# FunctionGemma Fine-Tuning for E-Commerce Customer Support

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/scionoftech/functiongemma-finetuning-e-commerce/blob/main/FunctionGemma_fine_tuning.ipynb)

A comprehensive, production-ready tutorial for fine-tuning Google's **FunctionGemma-270M-IT** model to build an intelligent E-Commerce Customer Support AI Agent with advanced function calling capabilities.

## Overview

This notebook demonstrates how to transform Google's FunctionGemma model into a specialized customer support agent that intelligently routes customer queries to appropriate handlers. Using parameter-efficient fine-tuning (LoRA), the model learns to classify customer intents and call the correct functions with minimal computational resources.

**Key Achievement**: Build a universal AI agent that handles 7 different customer support tasks with high accuracy using only ~2.4M trainable parameters on a consumer GPU.

## Features

- **Complete End-to-End Workflow**: From data generation to production deployment
- **Synthetic Dataset Generation**: Create configurable-size training datasets (3K-30K samples)
- **Parameter-Efficient Fine-Tuning**: LoRA implementation for memory-efficient training
- **7 Customer Support Functions**:
  - Order Management (track, update, cancel)
  - Product Search (search, check availability)
  - Product Details (specs, reviews, comparisons)
  - Returns & Refunds (returns, exchanges, refunds)
  - Account Management (profile, addresses, settings)
  - Payment Support (billing, payment issues)
  - Technical Support (app/website/login issues)
- **Comprehensive Evaluation**: Accuracy, confusion matrix, per-tool analysis, error analysis
- **Production-Ready Agent**: Complete implementation with context management and statistics
- **4-bit Quantization**: Memory-efficient training on consumer GPUs (<16GB VRAM)

## What You'll Learn

1. **FunctionGemma Format**: Understanding function calling format and requirements
2. **Synthetic Data Generation**: Creating high-quality training datasets programmatically
3. **LoRA Fine-Tuning**: Parameter-efficient fine-tuning techniques
4. **Model Evaluation**: Comprehensive testing and performance analysis
5. **Production Deployment**: Building a universal routing agent for real-world use

## Prerequisites

- **Hardware**: GPU with at least 12GB VRAM (recommended: 16GB+)
  - Google Colab T4/V100/A100
  - Local NVIDIA GPU (RTX 3060 12GB or better)
- **Python**: 3.8 or higher
- **Knowledge**: Basic understanding of:
  - Machine learning concepts
  - PyTorch fundamentals
  - Transformers library

## Quick Start

### Option 1: Google Colab (Recommended)

Click the "Open in Colab" badge above to launch the notebook directly in Google Colab with free GPU access.

### Option 2: Local Setup

```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git
cd YOUR_REPO_NAME

# Install dependencies
pip install -q -U transformers==4.46.3
pip install -q -U accelerate
pip install -q -U datasets
pip install -q -U peft
pip install -q -U trl
pip install -q -U bitsandbytes
pip install -q -U scikit-learn matplotlib seaborn wordcloud

# Launch Jupyter
jupyter notebook FunctionGemma_fine_tuning.ipynb
```

## Notebook Structure

The notebook is organized into 12 main sections plus production implementation:

| Section | Description | Time Est. |
|---------|-------------|-----------|
| **1. Environment Setup** | Install libraries and verify GPU | 2-5 min |
| **2. Understanding FunctionGemma** | Function calling concepts and format | Reading |
| **3. Use Case Analysis** | E-commerce requirements and tool design | Reading |
| **4. Define Function Tools** | 7 tool definitions with parameters | 5 min |
| **5. Generate Training Dataset** | Synthetic data generation (configurable) | 5-15 min |
| **6. Train/Val/Test Split** | 70/15/15 split with no overlap | 2 min |
| **7. Format Data** | Apply FunctionGemma format requirements | 3 min |
| **8. Load Model** | Load base model with 4-bit quantization | 2-5 min |
| **9. Configure LoRA** | Set up parameter-efficient adapters | 1 min |
| **10. Training Configuration** | Hyperparameters and training setup | 2 min |
| **11. Train Model** | Fine-tune model (3 epochs) | 20-60 min |
| **12. Save & Load** | Save weights and reload for evaluation | 2 min |
| **Evaluation** | Full test suite with metrics and analysis | 5-10 min |
| **Production Agent** | Universal routing agent implementation | 10 min |

**Total Runtime**: 60-120 minutes (depending on dataset size and GPU)

## Model Details

### Base Model
- **Name**: `google/functiongemma-270m-it`
- **Parameters**: 270 million
- **Type**: Instruction-tuned causal language model
- **Specialization**: Function calling tasks

### Fine-Tuning Configuration

**LoRA Parameters**:
```python
r=16                    # Rank (moderate capacity)
lora_alpha=32          # Scaling factor (2x rank)
target_modules=[       # Attention weight matrices
    "q_proj", "k_proj", "v_proj", "o_proj"
]
lora_dropout=0.05      # Regularization
```

**Training Hyperparameters**:
```python
learning_rate=2e-4              # Optimal for LoRA
num_train_epochs=3              # Full passes
per_device_train_batch_size=4   # Per GPU
gradient_accumulation_steps=4   # Effective batch=16
lr_scheduler_type="cosine"      # Decay schedule
warmup_ratio=0.1                # 10% warmup
weight_decay=0.01               # L2 regularization
```

**Quantization**:
- 4-bit NF4 (NormalFloat4) quantization
- BFloat16 compute dtype
- Memory footprint: <16GB VRAM

## Dataset Information

### Dataset Sizes

Choose from 5 configurable presets:

| Preset | Total Samples | Training | Validation | Test | Gen Time |
|--------|--------------|----------|------------|------|----------|
| **tiny** | ~3,000 | 2,100 | 450 | 450 | ~5 min |
| **small** | ~7,000 | 4,900 | 1,050 | 1,050 | ~10 min |
| **medium** | ~15,000 | 10,500 | 2,250 | 2,250 | ~15 min |
| **large** | ~22,000 | 15,400 | 3,300 | 3,300 | ~20 min |
| **xlarge** | ~30,000 | 21,000 | 4,500 | 4,500 | ~30 min |

### Data Quality Features

- **Natural Language Variations**: Multiple phrasings for each intent
- **Politeness Variations**: Formal and casual language styles
- **Context Diversity**: Different customer scenarios and edge cases
- **Tool Coverage**: Balanced distribution across all 7 function categories
- **Format Compliance**: Strict FunctionGemma format with proper tags

### Example Format

```
<start_function_declaration>
order_management(action: str, order_id: str = None, ...)
product_search(query: str, filters: dict = None)
...
<end_function_declaration>

<query>I need to track my order #12345</query>

<function_call>order_management("track", order_id="12345")</function_call>
```

## Training Process

### Memory Optimization

The notebook uses several techniques to enable training on consumer GPUs:

1. **4-bit Quantization**: Reduces memory by ~75%
2. **LoRA Adapters**: Only ~2.4M trainable parameters (vs. 270M full fine-tune)
3. **Gradient Accumulation**: Effective batch size of 16 with small per-device batches
4. **Mixed Precision**: BFloat16 for faster computation

### Training Monitoring

- Loss logging every 20 steps
- Validation evaluation after each epoch
- Best model checkpoint saved automatically
- Training history visualization

### Expected Training Time

| Dataset Size | GPU | Time |
|--------------|-----|------|
| Small (7K) | T4 | ~30 min |
| Medium (15K) | T4 | ~60 min |
| Large (22K) | V100 | ~45 min |
| XLarge (30K) | A100 | ~30 min |

## Evaluation Metrics

The notebook provides comprehensive evaluation including:

### 1. Overall Accuracy
- Percentage of correctly predicted functions
- Baseline comparison

### 2. Confusion Matrix
- Visual heatmap showing prediction patterns
- Identifies common misclassification pairs

### 3. Per-Tool Performance
- Individual accuracy for each of 7 functions
- Precision, recall, and F1-score
- Sample count per tool

### 4. Error Analysis
- Detailed examination of misclassifications
- Pattern identification in errors
- Suggestions for improvement

### 5. Classification Report
```
                    precision    recall  f1-score   support
order_management       0.95      0.93      0.94       450
product_search         0.92      0.94      0.93       420
product_details        0.94      0.92      0.93       380
...
```

### 6. Performance Dashboard
- Latency measurements (inference time)
- Resource utilization
- Request statistics

## Production Deployment

### Universal Agent Implementation

The notebook includes a production-ready `UniversalAgent` class:

```python
agent = UniversalAgent(model, tokenizer)

# Handle customer queries
response = agent.handle_request(
    "I want to return my order #12345"
)

# Multi-turn conversations
agent.handle_request("Search for laptops")
agent.handle_request("Show me the cheapest one")

# View statistics
stats = agent.get_statistics()
```

**Features**:
- **Context Management**: Maintains conversation history
- **Task Routing**: Automatically routes to correct handler
- **7 Specialized Handlers**: Order, search, details, returns, account, payment, technical
- **Statistics Tracking**: Request counts, latency, task switches
- **Error Handling**: Graceful fallbacks for edge cases

### Model Export

The fine-tuned model is saved and packaged for deployment:

```python
# Saved to: ./functiongemma-ecommerce-final/
# Includes: model weights, tokenizer, config
# Packaged as: functiongemma-ecommerce-final.zip
```

## Expected Results

With proper training, you can expect:

- **Overall Accuracy**: 90-95% on test set
- **Per-Tool Accuracy**: 85-98% depending on tool complexity
- **Inference Latency**: 50-200ms per query (GPU-dependent)
- **Memory Usage**: <8GB during inference (with quantization)

## Usage Examples

### Testing Individual Queries

```python
# Test a single customer query
test_model(
    "Can you help me cancel order #ORD789?",
    model,
    tokenizer
)
# Expected: order_management
```

### Running Full Evaluation

```python
# Evaluate on test set
results = evaluate_model(model, tokenizer, test_dataset)

# View confusion matrix
plot_confusion_matrix(results)

# Analyze errors
analyze_errors(results)
```

### Production Agent Demo

```python
# Initialize agent
agent = UniversalAgent(model, tokenizer)

# Customer interaction
print(agent.handle_request(
    "I'm looking for wireless headphones under $100"
))
# Output: [Searches products with filters]

print(agent.handle_request(
    "What's the battery life on the first one?"
))
# Output: [Shows product details for previously found item]
```

## Troubleshooting

### Common Issues

**1. CUDA Out of Memory**
```python
# Solution: Reduce batch size or use smaller dataset
per_device_train_batch_size = 2  # Instead of 4
gradient_accumulation_steps = 8  # Instead of 4
```

**2. Slow Training**
```python
# Solution: Use smaller dataset for testing
dataset_config = create_dataset_config("small")  # Instead of large
```

**3. Model Not Loading**
```bash
# Solution: Clear cache and reload
rm -rf ~/.cache/huggingface/
# Restart notebook
```

**4. Import Errors**
```bash
# Solution: Ensure compatible versions
pip install transformers==4.46.3  # Specific version
```

### Hardware Recommendations

| GPU | VRAM | Dataset Size | Batch Size |
|-----|------|--------------|------------|
| T4 | 16GB | Small-Medium | 4 |
| V100 | 16GB | Medium-Large | 4 |
| A100 | 40GB | XLarge | 8 |
| RTX 3060 | 12GB | Tiny-Small | 2 |
| RTX 4090 | 24GB | Any | 8 |

## Advanced Customization

### Add New Functions

```python
# Define new tool in Part 4
NEW_TOOL = {
    "name": "loyalty_program",
    "description": "Manage customer loyalty points and rewards",
    "parameters": {...}
}
```

### Adjust LoRA Rank

```python
# Higher rank = more capacity but more memory
lora_config = LoraConfig(
    r=32,  # Instead of 16
    lora_alpha=64,  # Keep 2x rank
    ...
)
```

### Custom Training Schedule

```python
# Longer training
training_args = TrainingArguments(
    num_train_epochs=5,  # Instead of 3
    learning_rate=1e-4,  # Lower learning rate
    ...
)
```

## Contributing

Contributions are welcome! Areas for improvement:

- Additional customer support functions
- Multi-language support
- Real customer query datasets
- Advanced evaluation metrics
- Production deployment guides

## Citation

If you use this notebook in your research or project, please cite:

```bibtex
@misc{functiongemma-ecommerce-2024,
  title={FunctionGemma Fine-Tuning for E-Commerce Customer Support},
  author={Your Name},
  year={2024},
  url={https://github.com/YOUR_USERNAME/YOUR_REPO_NAME}
}
```

## References

- [FunctionGemma Model Card](https://huggingface.co/google/functiongemma-270m-it)
- [LoRA: Low-Rank Adaptation](https://arxiv.org/abs/2106.09685)
- [Hugging Face PEFT Documentation](https://huggingface.co/docs/peft)
- [TRL: Transformer Reinforcement Learning](https://github.com/huggingface/trl)

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Google Research for FunctionGemma
- Hugging Face for transformers and PEFT libraries
- TRL team for supervised fine-tuning tools

---

**Ready to build your AI customer support agent?** Click the "Open in Colab" badge above to get started!

For questions or issues, please open a GitHub issue or contact the maintainers.
