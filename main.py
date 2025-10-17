"""
Main entry point for BERT fine-tuning experiments.
"""

import os
import yaml
import argparse
import torch
from pathlib import Path
from typing import Dict

from data import load_data, get_dataloader, get_few_shot_dataset
from model import get_strategy
from trainer import Trainer, run_few_shot_experiments, run_ablation_study
from utils import set_seed, save_results, print_experiment_summary


def run_single_strategy(strategy_name: str, config: Dict, train_loader, val_loader, 
                       test_loader, num_labels: int):
    """Run training for a single strategy."""
    print(f"\n{'='*60}")
    print(f" Training with {strategy_name.upper()} ".center(60, '='))
    print('='*60)
    
    checkpoint_dir = Path(config['checkpoint']['checkpoint_dir']) / strategy_name
    checkpoint_path = checkpoint_dir / 'best_model.pt'
    
    if checkpoint_path.exists():
        print(f"Found existing checkpoint for {strategy_name}")
        print(f"Loading from: {checkpoint_path}")
        
        strategy = get_strategy(strategy_name, config['model']['name'], num_labels, config)
        model = strategy.get_model()
        trainer = Trainer(model, config)
        trainer.load_checkpoint(str(checkpoint_path))
        
        trainable_params, total_params = strategy.get_trainable_params()
        print(f"Trainable: {trainable_params:,} / Total: {total_params:,} ({100 * trainable_params / total_params:.2f}%)")
        
        test_metrics = trainer.test(test_loader)
        
        train_stats = {
            'train_time': trainer.train_time,
            'peak_memory_gb': trainer.peak_memory_gb,
            'best_val_accuracy': trainer.best_val_accuracy,
            'final_train_loss': trainer.train_losses[-1] if trainer.train_losses else 0,
            'final_val_loss': trainer.val_losses[-1] if trainer.val_losses else 0,
            'final_val_accuracy': trainer.val_accuracies[-1] if trainer.val_accuracies else 0,
            'total_epochs': len(trainer.train_losses),
            'train_losses': trainer.train_losses,
            'val_losses': trainer.val_losses,
            'val_accuracies': trainer.val_accuracies
        }
    else:
        print(f"No checkpoint found, training from scratch...")
        
        strategy = get_strategy(strategy_name, config['model']['name'], num_labels, config)
        model = strategy.get_model()
        
        trainable_params, total_params = strategy.get_trainable_params()
        print(f"Trainable: {trainable_params:,} / Total: {total_params:,} ({100 * trainable_params / total_params:.2f}%)")
        
        trainer = Trainer(model, config)
        train_stats = trainer.train(train_loader, val_loader, strategy_name=strategy_name)
        
        # Add training history
        train_stats['train_losses'] = trainer.train_losses
        train_stats['val_losses'] = trainer.val_losses
        train_stats['val_accuracies'] = trainer.val_accuracies
        
        test_metrics = trainer.test(test_loader)
    
    results = {
        'train_stats': train_stats,
        'test_metrics': test_metrics,
        'trainable_params': trainable_params,
        'total_params': total_params
    }
    
    return results, trainer, model


def run_all_experiments(config: Dict):
    """Run all experiments."""
    
    set_seed(config['training']['seed'])
    
    output_dir = Path(config['experiment']['output_dir'])
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / 'results').mkdir(exist_ok=True)
    
    print("Loading data...")
    train_dataset, val_dataset, test_dataset, num_labels = load_data(config)
    config['model']['num_labels'] = num_labels
    
    num_workers = config['training'].get('num_workers', 2)
    train_loader = get_dataloader(train_dataset, config['training']['batch_size'], 
                                  shuffle=True, num_workers=num_workers)
    val_loader = get_dataloader(val_dataset, config['training']['batch_size'], 
                               shuffle=False, num_workers=num_workers)
    test_loader = get_dataloader(test_dataset, config['training']['batch_size'], 
                                shuffle=False, num_workers=num_workers)
    
    print(f"Dataset: {config['data']['name']}")
    print(f"Train: {len(train_dataset)} | Val: {len(val_dataset)} | Test: {len(test_dataset)}")
    print(f"Labels: {num_labels}")
    
    all_results = {}
    
    strategies_to_run = [s for s in config['strategies'] if config['strategies'][s].get('enabled', False)]
    
    if not strategies_to_run:
        print("No strategies enabled!")
        return {}
    
    print(f"\nStrategies to run: {', '.join(strategies_to_run)}")
    
    for strategy_name in strategies_to_run:
        try:
            results, trainer, model = run_single_strategy(
                strategy_name, config, train_loader, val_loader, test_loader, num_labels
            )
            all_results[strategy_name] = results
        except Exception as e:
            print(f"Error running {strategy_name}: {str(e)}")
            continue
        finally:
            if 'model' in locals():
                del model
            if 'trainer' in locals():
                del trainer
            torch.cuda.empty_cache()
    
    if not all_results:
        print("No strategies completed successfully!")
        return {}
    
    # Few-shot experiments
    if 'lora' in strategies_to_run and config['data'].get('few_shot_sizes'):
        print("\n" + "="*60)
        print(" FEW-SHOT LEARNING ".center(60, '='))
        print("="*60)
        
        try:
            def get_lora_model():
                strategy = get_strategy('lora', config['model']['name'], num_labels, config)
                return strategy.get_model()
            
            few_shot_results = run_few_shot_experiments(
                get_lora_model, train_dataset, val_loader, test_loader,
                config['data']['few_shot_sizes'], config
            )
            all_results['few_shot_lora'] = few_shot_results
        except Exception as e:
            print(f"Few-shot experiments failed: {str(e)}")
    
    # Ablation studies
    if 'lora' in strategies_to_run and config['strategies']['lora'].get('rank_values'):
        print("\n" + "="*60)
        print(" ABLATION: LORA RANK ".center(60, '='))
        print("="*60)
        
        try:
            def get_lora_with_rank(rank=None):
                strategy = get_strategy('lora', config['model']['name'], num_labels, config, rank=rank)
                return strategy.get_model()
            
            lora_ablation = run_ablation_study(
                get_lora_with_rank, train_loader, val_loader, test_loader,
                'lora_rank', config['strategies']['lora']['rank_values'], config
            )
            all_results['ablation_lora_rank'] = lora_ablation
        except Exception as e:
            print(f"LoRA ablation failed: {str(e)}")
    
    # Learning rate ablation
    if config['ablation'].get('learning_rates') and strategies_to_run:
        best_strategy = max(strategies_to_run, 
                          key=lambda s: all_results.get(s, {}).get('test_metrics', {}).get('accuracy', 0))
        
        print("\n" + "="*60)
        print(f" ABLATION: LEARNING RATE ({best_strategy}) ".center(60, '='))
        print("="*60)
        
        try:
            def get_best_model():
                strategy = get_strategy(best_strategy, config['model']['name'], num_labels, config)
                return strategy.get_model()
            
            lr_ablation = run_ablation_study(
                get_best_model, train_loader, val_loader, test_loader,
                'learning_rate', config['ablation']['learning_rates'], config
            )
            all_results['ablation_learning_rate'] = lr_ablation
        except Exception as e:
            print(f"Learning rate ablation failed: {str(e)}")
    
    # Save results
    results_path = output_dir / 'results/all_results.json'
    save_results(all_results, str(results_path))
    
    # Print summary
    strategy_results = {k: v for k, v in all_results.items() if k in strategies_to_run}
    if strategy_results:
        print_experiment_summary(strategy_results)
    
    print(f"\n\nAll experiments completed!")
    print(f"Results saved to: {results_path}")
    print(f"\nTo visualize results, run: python visualize.py")
    
    return all_results


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='BERT Fine-tuning Experiments')
    parser.add_argument('--config', type=str, default='config.yaml')
    parser.add_argument('--strategy', type=str, default=None)
    parser.add_argument('--quick', action='store_true')
    
    args = parser.parse_args()
    
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    if args.quick:
        config['training']['num_epochs'] = 2
        config['data']['train_size'] = 1000
        config['data']['few_shot_sizes'] = [10, 50]
        config['strategies']['lora']['rank_values'] = [4, 8]
        config['ablation']['learning_rates'] = [1e-5, 2e-5]
        print("Quick mode enabled")
    
    if args.strategy:
        for s in config['strategies']:
            config['strategies'][s]['enabled'] = (s == args.strategy)
        print(f"Running only {args.strategy}")
    
    if torch.cuda.is_available():
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
        print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    else:
        print("No GPU available, using CPU")
    
    try:
        results = run_all_experiments(config)
    except Exception as e:
        print(f"\nFatal error: {str(e)}")
        import traceback
        traceback.print_exc()
        return None
    
    return results


if __name__ == '__main__':
    main()