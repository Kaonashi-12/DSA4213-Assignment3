"""
Visualization script for experiment results - Enhanced version.
Usage: python visualize.py [--results path/to/results.json]
"""

import json
import argparse
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Enhanced style configuration for better readability in reports
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 200  # Higher DPI for better quality
plt.rcParams['savefig.dpi'] = 200
plt.rcParams['font.size'] = 14  # Larger base font
plt.rcParams['axes.labelsize'] = 16
plt.rcParams['axes.titlesize'] = 18
plt.rcParams['xtick.labelsize'] = 14
plt.rcParams['ytick.labelsize'] = 14
plt.rcParams['legend.fontsize'] = 13
plt.rcParams['lines.linewidth'] = 3
plt.rcParams['lines.markersize'] = 10

COLORS = sns.color_palette("husl", 8)


def load_results(results_path: str):
    """Load results from JSON."""
    with open(results_path, 'r') as f:
        return json.load(f)


def get_strategy_list(results: dict):
    """Get list of main strategies."""
    return [k for k in results.keys() if 'ablation' not in k and 'few_shot' not in k]


def plot_convergence_curves(results: dict, output_dir: Path):
    """Plot convergence curves for all strategies."""
    print("Plotting convergence curves...")
    
    strategies = get_strategy_list(results)
    
    # 1. Training Loss
    fig, ax = plt.subplots(figsize=(10, 7))
    y_min, y_max = float('inf'), 0
    
    for i, strategy in enumerate(strategies):
        if 'train_stats' in results[strategy] and 'train_losses' in results[strategy]['train_stats']:
            losses = results[strategy]['train_stats']['train_losses']
            epochs = range(1, len(losses) + 1)
            ax.plot(epochs, losses, 'o-', label=strategy.replace('_', ' ').title(), 
                   linewidth=3.5, markersize=11, color=COLORS[i], alpha=0.85,
                   markeredgewidth=2, markeredgecolor='white')
            y_min = min(y_min, min(losses))
            y_max = max(y_max, max(losses))
    
    margin = (y_max - y_min) * 0.1
    ax.set_ylim([max(0, y_min - margin), y_max + margin])
    ax.set_xlabel('Epoch', fontsize=18, fontweight='bold')
    ax.set_ylabel('Training Loss', fontsize=18, fontweight='bold')
    ax.set_title('Training Loss Convergence', fontsize=20, fontweight='bold', pad=15)
    ax.legend(fontsize=14, framealpha=0.95, loc='best', frameon=True, fancybox=True, shadow=True)
    ax.grid(True, alpha=0.3, linewidth=1.5)
    plt.tight_layout()
    plt.savefig(output_dir / 'convergence_train_loss.png', bbox_inches='tight', dpi=200)
    plt.close()
    
    # 2. Validation Loss
    fig, ax = plt.subplots(figsize=(10, 7))
    y_min, y_max = float('inf'), 0
    
    for i, strategy in enumerate(strategies):
        if 'train_stats' in results[strategy] and 'val_losses' in results[strategy]['train_stats']:
            losses = results[strategy]['train_stats']['val_losses']
            epochs = range(1, len(losses) + 1)
            ax.plot(epochs, losses, 'o-', label=strategy.replace('_', ' ').title(),
                   linewidth=3.5, markersize=11, color=COLORS[i], alpha=0.85,
                   markeredgewidth=2, markeredgecolor='white')
            y_min = min(y_min, min(losses))
            y_max = max(y_max, max(losses))
    
    margin = (y_max - y_min) * 0.1
    ax.set_ylim([max(0, y_min - margin), y_max + margin])
    ax.set_xlabel('Epoch', fontsize=18, fontweight='bold')
    ax.set_ylabel('Validation Loss', fontsize=18, fontweight='bold')
    ax.set_title('Validation Loss Convergence', fontsize=20, fontweight='bold', pad=15)
    ax.legend(fontsize=14, framealpha=0.95, loc='best', frameon=True, fancybox=True, shadow=True)
    ax.grid(True, alpha=0.3, linewidth=1.5)
    plt.tight_layout()
    plt.savefig(output_dir / 'convergence_val_loss.png', bbox_inches='tight', dpi=200)
    plt.close()
    
    print(f"  Saved convergence curves")


def plot_performance_comparison(results: dict, output_dir: Path):
    """Plot performance comparison."""
    print("Plotting performance comparison...")
    
    strategies = get_strategy_list(results)
    
    data = {'Strategy': [], 'Accuracy': [], 'F1': [], 'Precision': [], 'Recall': []}
    
    for strategy in strategies:
        if 'test_metrics' in results[strategy]:
            metrics = results[strategy]['test_metrics']
            data['Strategy'].append(strategy.replace('_', ' ').title())
            data['Accuracy'].append(metrics['accuracy'])
            data['F1'].append(metrics['f1'])
            data['Precision'].append(metrics['precision'])
            data['Recall'].append(metrics['recall'])
    
    fig, ax = plt.subplots(figsize=(14, 8))
    x = np.arange(len(data['Strategy']))
    width = 0.2
    
    bars1 = ax.bar(x - 1.5*width, data['Accuracy'], width, label='Accuracy', 
                   color='#3498db', alpha=0.85, edgecolor='black', linewidth=1.5)
    bars2 = ax.bar(x - 0.5*width, data['F1'], width, label='F1 Score',
                   color='#e74c3c', alpha=0.85, edgecolor='black', linewidth=1.5)
    bars3 = ax.bar(x + 0.5*width, data['Precision'], width, label='Precision',
                   color='#2ecc71', alpha=0.85, edgecolor='black', linewidth=1.5)
    bars4 = ax.bar(x + 1.5*width, data['Recall'], width, label='Recall',
                   color='#f39c12', alpha=0.85, edgecolor='black', linewidth=1.5)
    
    for bars in [bars1, bars2, bars3, bars4]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.3f}', ha='center', va='bottom', 
                   fontsize=11, fontweight='bold')
    
    all_values = data['Accuracy'] + data['F1'] + data['Precision'] + data['Recall']
    y_min = min(all_values)
    y_max = max(all_values)
    margin = (y_max - y_min) * 0.15
    ax.set_ylim([max(0, y_min - margin), min(1, y_max + margin)])
    
    ax.set_ylabel('Score', fontsize=18, fontweight='bold')
    ax.set_title('Test Performance Comparison', fontsize=20, fontweight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(data['Strategy'], fontsize=15, fontweight='bold')
    ax.legend(fontsize=15, framealpha=0.95, loc='upper right', frameon=True, 
             fancybox=True, shadow=True)
    ax.grid(True, alpha=0.3, axis='y', linestyle='--', linewidth=1.5)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'performance_comparison.png', bbox_inches='tight', dpi=200)
    plt.close()
    
    print(f"  Saved performance comparison")


def plot_few_shot_results(results: dict, output_dir: Path):
    """Plot simplified few-shot learning results."""
    if 'few_shot_lora' not in results:
        return
    
    print("Plotting few-shot results...")
    
    few_shot = results['few_shot_lora']
    sample_sizes = sorted([int(k) for k in few_shot.keys()])
    accuracies = [few_shot[str(n)]['test_metrics']['accuracy'] for n in sample_sizes]
    f1_scores = [few_shot[str(n)]['test_metrics']['f1'] for n in sample_sizes]
    
    fig, ax = plt.subplots(figsize=(10, 7))
    
    # Simplified: just two lines with different colors and styles
    ax.plot(sample_sizes, accuracies, 'o-', label='Accuracy', 
           linewidth=4, markersize=13, color='#2ecc71', 
           markeredgecolor='white', markeredgewidth=2.5)
    ax.plot(sample_sizes, f1_scores, 's--', label='F1 Score',
           linewidth=4, markersize=13, color='#e74c3c',
           markeredgecolor='white', markeredgewidth=2.5)
    
    ax.set_xscale('log')
    ax.set_xlabel('Samples per Class (log scale)', fontsize=18, fontweight='bold')
    ax.set_ylabel('Performance', fontsize=18, fontweight='bold')
    ax.set_title('Few-Shot Learning Performance (LoRA)', fontsize=20, fontweight='bold', pad=15)
    ax.legend(fontsize=16, framealpha=0.95, loc='best', frameon=True, 
             fancybox=True, shadow=True)
    ax.grid(True, alpha=0.3, linewidth=1.5)
    
    # Dynamic y-axis
    y_min = min(min(accuracies), min(f1_scores))
    y_max = max(max(accuracies), max(f1_scores))
    margin = (y_max - y_min) * 0.15
    ax.set_ylim([max(0, y_min - margin), min(1, y_max + margin)])
    
    # Larger annotations
    for x, y in zip(sample_sizes, accuracies):
        ax.annotate(f'{y:.3f}', (x, y), xytext=(0, 12), textcoords='offset points',
                   ha='center', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'few_shot_learning.png', bbox_inches='tight', dpi=200)
    plt.close()
    
    print(f"  Saved few-shot results")


def main():
    parser = argparse.ArgumentParser(description='Visualize experiment results')
    parser.add_argument('--results', type=str, default='outputs/results/all_results.json')
    parser.add_argument('--output', type=str, default='outputs/plots')
    
    args = parser.parse_args()
    
    results_path = Path(args.results)
    if not results_path.exists():
        print(f"Error: Results file not found: {results_path}")
        return
    
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Loading results from: {results_path}")
    results = load_results(str(results_path))
    
    print(f"\nGenerating visualizations...\n")
    
    try:
        plot_convergence_curves(results, output_dir)
        plot_performance_comparison(results, output_dir)
        plot_few_shot_results(results, output_dir)
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
    
    print(f"\nAll visualizations saved to: {output_dir}")
    print("\nGenerated files:")
    for f in sorted(output_dir.glob('*.png')):
        print(f"  - {f.name}")


if __name__ == '__main__':
    main()