"""
Utility functions.
"""

import torch
import numpy as np
import json
from pathlib import Path
from typing import Dict
import pandas as pd


def set_seed(seed: int):
    """Set random seeds for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def save_results(results: Dict, save_path: str):
    """Save results to JSON."""
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    
    def convert(obj):
        if isinstance(obj, (np.ndarray, np.generic)):
            return obj.tolist()
        if isinstance(obj, (np.float32, np.float64)):
            return float(obj)
        if isinstance(obj, (np.int32, np.int64)):
            return int(obj)
        if isinstance(obj, torch.Tensor):
            return obj.cpu().numpy().tolist()
        return obj
    
    serializable = json.loads(json.dumps(results, default=convert))
    
    with open(save_path, 'w') as f:
        json.dump(serializable, f, indent=2)
    
    print(f"Results saved to {save_path}")


def print_experiment_summary(results: Dict):
    """Print summary table."""
    print("\n" + "="*80)
    print(" EXPERIMENT SUMMARY ".center(80, "="))
    print("="*80 + "\n")
    
    rows = []
    for strategy, res in results.items():
        if 'test_metrics' in res:
            rows.append({
                'Strategy': strategy,
                'Accuracy': f"{res['test_metrics']['accuracy']:.4f}",
                'F1': f"{res['test_metrics']['f1']:.4f}",
                'Precision': f"{res['test_metrics']['precision']:.4f}",
                'Recall': f"{res['test_metrics']['recall']:.4f}",
                'Epochs': res['train_stats'].get('total_epochs', 'N/A'),
                'Trainable Params': f"{res.get('trainable_params', 0):,}" if res.get('trainable_params') else 'N/A'
            })
    
    df = pd.DataFrame(rows)
    print(df.to_string(index=False))
    print("\n" + "="*80)
    
    best = max(results.items(), key=lambda x: x[1]['test_metrics']['accuracy'] if 'test_metrics' in x[1] else 0)
    print(f"\nBest Strategy: {best[0]}")
    print(f"  Accuracy: {best[1]['test_metrics']['accuracy']:.4f}")
    print(f"  F1: {best[1]['test_metrics']['f1']:.4f}")