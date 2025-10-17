# BERT Fine-tuning Experiments

A comprehensive framework for experimenting with different BERT fine-tuning strategies on text classification tasks.

## Overview

This repository provides implementations of multiple parameter-efficient fine-tuning methods:

- **Full Fine-tuning**: Traditional approach training all parameters
- **LoRA** (Low-Rank Adaptation): Efficient fine-tuning using low-rank matrices
- **BitFit**: Fine-tuning only bias parameters
- **Prompt Tuning**: Learning soft prompts prepended to inputs

## Features

- Multiple fine-tuning strategies with automatic comparison
- Ablation studies (learning rates, LoRA ranks)
- Training convergence visualization
- Model checkpointing and resumption
- Mixed precision training (FP16)
- Early stopping

## Installation

```bash
pip install -r requirements.txt
```

## Quick Start

### Basic Usage

Run all enabled strategies with default configuration:

```bash
python main.py
```

### Run Specific Strategy

```bash
python main.py --strategy lora
```

### Custom Configuration

Modify `config.yaml` to customize:
- Model settings (BERT variant, max length)
- Dataset (AG News, IMDB, Banking77)
- Training hyperparameters
- Fine-tuning strategies and their parameters

## Usage Examples

### 1. Compare All Strategies

```bash
# Enable strategies in config.yaml
python main.py
```

Results are saved to `outputs/results/all_results.json`

### 2. Ablation Studies

Test different LoRA ranks:

```yaml
strategies:
  lora:
    rank_values: [4, 8, 16]
```

Test different learning rates:

```yaml
ablation:
  learning_rates: [1e-5, 2e-5, 5e-5]
```

### 3. Visualize Results

Generate plots after training:

```bash
python visualize.py
```

Plots are saved to `outputs/plots/`:
- Training/validation loss curves
- Performance comparison bar charts
- Few-shot learning results

## Project Structure

```
.
├── config.yaml           # Main configuration file
├── main.py              # Entry point for experiments
├── data.py              # Data loading and preprocessing
├── model.py             # Fine-tuning strategy implementations
├── trainer.py           # Training and evaluation logic
├── utils.py             # Utility functions
├── visualize.py         # Visualization script
└── requirements.txt     # Python dependencies
```

## Output Structure

```
outputs/
├── checkpoints/         # Model checkpoints by strategy
│   ├── lora/
│   ├── full_finetuning/
│   └── ...
├── results/
│   └── all_results.json # Complete experiment results
└── plots/               # Generated visualizations
    ├── convergence_train_loss.png
    ├── convergence_val_loss.png
    ├── performance_comparison.png
    └── few_shot_learning.png
```

## Configuration Options

Key settings in `config.yaml`:

```yaml
model:
  name: "bert-base-uncased"  # HuggingFace model name
  max_length: 128            # Max sequence length

data:
  name: "ag_news"            # Dataset: ag_news, imdb, banking77
  train_size: 10000          # Limit training samples (null for full)

training:
  batch_size: 64
  learning_rate: 2e-5
  num_epochs: 10
  fp16: true                 # Mixed precision training
  early_stopping:
    enabled: true
    patience: 3
```
