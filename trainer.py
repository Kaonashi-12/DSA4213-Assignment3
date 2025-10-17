"""
Training and evaluation utilities.
"""

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.amp import autocast, GradScaler
from transformers import get_linear_schedule_with_warmup
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from typing import Dict, Optional
from pathlib import Path
import time
import gc


class EarlyStopping:
    """Early stopping handler."""
    
    def __init__(self, patience: int = 3, min_delta: float = 0.0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_score = None
        
    def __call__(self, score: float) -> bool:
        if self.best_score is None:
            self.best_score = score
            return False
        
        if score > self.best_score + self.min_delta:
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            return self.counter >= self.patience
        return False


class Trainer:
    """Main trainer class."""
    
    def __init__(self, model, config: Dict, device: str = None):
        self.model = model
        self.config = config
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        
        self.train_losses = []
        self.val_losses = []
        self.val_accuracies = []
        self.best_val_accuracy = 0
        self.best_model_state = None
        self.train_time = 0
        self.peak_memory_gb = 0
        
        self.use_amp = config['training'].get('fp16', False) and torch.cuda.is_available()
        self.scaler = GradScaler('cuda') if self.use_amp else None
        
        if self.use_amp:
            print("Mixed precision training (FP16) enabled")
        
        early_config = config['training'].get('early_stopping', {})
        self.early_stopping = EarlyStopping(
            patience=early_config.get('patience', 3),
            min_delta=early_config.get('min_delta', 0.001)
        ) if early_config.get('enabled', False) else None
    
    def train_epoch(self, train_loader, optimizer, scheduler):
        """Train for one epoch."""
        self.model.train()
        total_loss = 0
        grad_accum = self.config['training'].get('gradient_accumulation_steps', 1)
        
        for i, batch in enumerate(tqdm(train_loader, desc="Training", mininterval=1.0)):
            input_ids = batch['input_ids'].to(self.device, non_blocking=True)
            attention_mask = batch['attention_mask'].to(self.device, non_blocking=True)
            labels = batch['labels'].to(self.device, non_blocking=True)
            
            if self.use_amp:
                with autocast('cuda'):
                    loss = self.model(input_ids, attention_mask, labels=labels).loss / grad_accum
                self.scaler.scale(loss).backward()
                
                if (i + 1) % grad_accum == 0:
                    self.scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    self.scaler.step(optimizer)
                    self.scaler.update()
                    optimizer.zero_grad()
                    scheduler.step()
            else:
                loss = self.model(input_ids, attention_mask, labels=labels).loss / grad_accum
                loss.backward()
                
                if (i + 1) % grad_accum == 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    optimizer.step()
                    optimizer.zero_grad()
                    scheduler.step()
            
            total_loss += loss.item() * grad_accum
        
        return total_loss / len(train_loader)
    
    def evaluate(self, eval_loader):
        """Evaluate the model."""
        self.model.eval()
        total_loss = 0
        all_preds, all_labels = [], []
        
        with torch.no_grad():
            for batch in tqdm(eval_loader, desc="Evaluating", mininterval=1.0):
                input_ids = batch['input_ids'].to(self.device, non_blocking=True)
                attention_mask = batch['attention_mask'].to(self.device, non_blocking=True)
                labels = batch['labels'].to(self.device, non_blocking=True)
                
                if self.use_amp:
                    with autocast('cuda'):
                        outputs = self.model(input_ids, attention_mask, labels=labels)
                else:
                    outputs = self.model(input_ids, attention_mask, labels=labels)
                
                total_loss += outputs.loss.item()
                preds = torch.argmax(outputs.logits, dim=-1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        return {
            'loss': total_loss / len(eval_loader),
            'accuracy': accuracy_score(all_labels, all_preds),
            'f1': f1_score(all_labels, all_preds, average='weighted'),
            'precision': precision_score(all_labels, all_preds, average='weighted', zero_division=0),
            'recall': recall_score(all_labels, all_preds, average='weighted', zero_division=0)
        }
    
    def save_checkpoint(self, save_path: str):
        """Save model checkpoint."""
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'best_val_accuracy': self.best_val_accuracy,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'val_accuracies': self.val_accuracies,
            'train_time': self.train_time,
            'peak_memory_gb': self.peak_memory_gb,
        }, save_path)
        print(f"Checkpoint saved to {save_path}")
    
    def load_checkpoint(self, load_path: str):
        """Load model checkpoint."""
        checkpoint = torch.load(load_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.best_val_accuracy = checkpoint.get('best_val_accuracy', 0)
        self.train_losses = checkpoint.get('train_losses', [])
        self.val_losses = checkpoint.get('val_losses', [])
        self.val_accuracies = checkpoint.get('val_accuracies', [])
        self.train_time = checkpoint.get('train_time', 0)
        self.peak_memory_gb = checkpoint.get('peak_memory_gb', 0)
        print(f"Checkpoint loaded from {load_path}")
    
    def train(self, train_loader, val_loader, learning_rate: Optional[float] = None,
              scheduler_type: str = 'linear', num_epochs: Optional[int] = None,
              strategy_name: str = None):
        """Training loop with warmup handling."""
        
        epochs = num_epochs or self.config['training']['num_epochs']
        lr = float(learning_rate or self.config['training']['learning_rate'])
        optimizer = AdamW(self.model.parameters(), lr=lr, 
                         weight_decay=self.config['training']['weight_decay'])
        
        grad_accum = self.config['training'].get('gradient_accumulation_steps', 1)
        num_steps = (epochs * len(train_loader)) // grad_accum
        scheduler = get_linear_schedule_with_warmup(
            optimizer, 
            num_warmup_steps=self.config['training']['warmup_steps'],
            num_training_steps=num_steps
        )
        
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.empty_cache()
        
        checkpoint_dir = Path(self.config['checkpoint']['checkpoint_dir'])
        if strategy_name:
            checkpoint_dir = checkpoint_dir / strategy_name
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Warmup epoch (not counted)
        print("Running warmup epoch...")
        self.train_epoch(train_loader, optimizer, scheduler)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        
        # Reset for actual training
        optimizer = AdamW(self.model.parameters(), lr=lr, 
                         weight_decay=self.config['training']['weight_decay'])
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=self.config['training']['warmup_steps'],
            num_training_steps=num_steps
        )
        
        total_train_time = 0
        
        for epoch in range(epochs):
            print(f"\nEpoch {epoch + 1}/{epochs}")
            
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            
            epoch_start = time.time()
            train_loss = self.train_epoch(train_loader, optimizer, scheduler)
            
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            
            epoch_time = time.time() - epoch_start
            total_train_time += epoch_time
            self.train_losses.append(train_loss)
            
            val_metrics = self.evaluate(val_loader)
            self.val_losses.append(val_metrics['loss'])
            self.val_accuracies.append(val_metrics['accuracy'])
            
            print(f"Train Loss: {train_loss:.4f} | Time: {epoch_time:.1f}s")
            print(f"Val Loss: {val_metrics['loss']:.4f} | Acc: {val_metrics['accuracy']:.4f} | F1: {val_metrics['f1']:.4f}")
            
            if val_metrics['accuracy'] > self.best_val_accuracy:
                self.best_val_accuracy = val_metrics['accuracy']
                self.best_model_state = self.model.state_dict().copy()
                print(f"New best: {self.best_val_accuracy:.4f}")
                
                if torch.cuda.is_available():
                    self.peak_memory_gb = torch.cuda.max_memory_allocated() / 1024**3
                self.train_time = total_train_time
                
                if self.config['checkpoint'].get('save_best_model', True):
                    self.save_checkpoint(str(checkpoint_dir / 'best_model.pt'))
            
            if torch.cuda.is_available():
                print(f"Peak Memory: {torch.cuda.max_memory_allocated() / 1024**3:.3f} GB")
            
            if self.early_stopping and self.early_stopping(val_metrics['accuracy']):
                print(f"Early stopping at epoch {epoch + 1}")
                break
        
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            self.peak_memory_gb = torch.cuda.max_memory_allocated() / 1024**3
        
        if self.best_model_state:
            self.model.load_state_dict(self.best_model_state)
        
        self.train_time = total_train_time
        
        return {
            'train_time': total_train_time,
            'peak_memory_gb': self.peak_memory_gb,
            'best_val_accuracy': self.best_val_accuracy,
            'final_train_loss': self.train_losses[-1],
            'final_val_loss': self.val_losses[-1],
            'final_val_accuracy': self.val_accuracies[-1],
            'total_epochs': len(self.train_losses)
        }
    
    def test(self, test_loader):
        """Evaluate on test set."""
        print("\nEvaluating on test set...")
        metrics = self.evaluate(test_loader)
        print(f"Test Acc: {metrics['accuracy']:.4f} | F1: {metrics['f1']:.4f}")
        return metrics


def run_few_shot_experiments(strategy_fn, train_dataset, val_loader, test_loader, 
                            few_shot_sizes, config):
    """Run few-shot experiments."""
    from data import get_few_shot_dataset, get_dataloader
    
    results = {}
    num_workers = config['training'].get('num_workers', 4)
    
    for n_samples in few_shot_sizes:
        print(f"\n{'='*50}")
        print(f"Few-shot: {n_samples} samples per class")
        print('='*50)
        
        strategy_name = f'few_shot_{n_samples}'
        checkpoint_path = Path(config['checkpoint']['checkpoint_dir']) / strategy_name / 'best_model.pt'
        
        if checkpoint_path.exists():
            print(f"Loading checkpoint: {checkpoint_path}")
            model = strategy_fn()
            trainer = Trainer(model, config)
            trainer.load_checkpoint(str(checkpoint_path))
            test_metrics = trainer.test(test_loader)
            
            stats = {
                'train_time': trainer.train_time,
                'peak_memory_gb': trainer.peak_memory_gb,
                'best_val_accuracy': trainer.best_val_accuracy,
                'total_epochs': len(trainer.train_losses)
            }
        else:
            few_shot_dataset = get_few_shot_dataset(train_dataset, n_samples)
            few_shot_loader = get_dataloader(
                few_shot_dataset, config['training']['batch_size'],
                shuffle=True, num_workers=num_workers
            )
            
            model = strategy_fn()
            trainer = Trainer(model, config)
            stats = trainer.train(few_shot_loader, val_loader, num_epochs=3,
                                strategy_name=strategy_name)
            test_metrics = trainer.test(test_loader)
        
        results[n_samples] = {
            'train_stats': stats,
            'test_metrics': test_metrics,
            'num_samples': n_samples * 4
        }
        
        del model, trainer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    return results


def run_ablation_study(base_strategy_fn, train_loader, val_loader, test_loader,
                      ablation_type: str, ablation_values, config):
    """Run ablation study."""
    results = {}
    
    for value in ablation_values:
        print(f"\n{'='*50}")
        print(f"Ablation: {ablation_type} = {value}")
        print('='*50)
        
        strategy_name = f'ablation_{ablation_type}_{value}'
        checkpoint_path = Path(config['checkpoint']['checkpoint_dir']) / strategy_name / 'best_model.pt'
        
        if checkpoint_path.exists():
            print(f"Loading checkpoint: {checkpoint_path}")
            
            if ablation_type == 'lora_rank':
                model = base_strategy_fn(rank=value)
            else:
                model = base_strategy_fn()
            
            trainer = Trainer(model, config)
            trainer.load_checkpoint(str(checkpoint_path))
            test_metrics = trainer.test(test_loader)
            
            stats = {
                'train_time': trainer.train_time,
                'peak_memory_gb': trainer.peak_memory_gb,
                'best_val_accuracy': trainer.best_val_accuracy,
                'total_epochs': len(trainer.train_losses)
            }
        else:
            if ablation_type == 'learning_rate':
                model = base_strategy_fn()
                trainer = Trainer(model, config)
                stats = trainer.train(train_loader, val_loader, learning_rate=value,
                                    strategy_name=strategy_name)
            elif ablation_type == 'lora_rank':
                model = base_strategy_fn(rank=value)
                trainer = Trainer(model, config)
                stats = trainer.train(train_loader, val_loader, strategy_name=strategy_name)
            else:
                raise ValueError(f"Unknown ablation type: {ablation_type}")
            
            test_metrics = trainer.test(test_loader)
        
        results[value] = {
            'train_stats': stats,
            'test_metrics': test_metrics
        }
        
        del model, trainer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    return results