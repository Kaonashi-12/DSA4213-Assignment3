"""
Data loading and preprocessing utilities.
"""

import torch
from torch.utils.data import DataLoader, Dataset, Subset
from transformers import AutoTokenizer
from datasets import load_dataset
import numpy as np
from typing import Dict, List, Optional, Tuple


class TextDataset(Dataset):
    """Text dataset wrapper."""
    
    def __init__(self, texts: List[str], labels: List[int], tokenizer, max_length: int = 128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        encoding = self.tokenizer(
            self.texts[idx],
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(self.labels[idx], dtype=torch.long)
        }


def sample_dataset(texts: List[str], labels: List[int], 
                  max_size: Optional[int] = None, seed: int = 42) -> Tuple[List[str], List[int]]:
    """Sample dataset with class balance."""
    if max_size is None or len(texts) <= max_size:
        return texts, labels
    
    np.random.seed(seed)
    label_indices = {}
    for idx, label in enumerate(labels):
        label_indices.setdefault(label, []).append(idx)
    
    samples_per_class = max_size // len(label_indices)
    selected = []
    
    for indices in label_indices.values():
        sampled = np.random.choice(indices, min(samples_per_class, len(indices)), replace=False)
        selected.extend(sampled)
    
    np.random.shuffle(selected)
    return [texts[i] for i in selected], [labels[i] for i in selected]


def load_data(config: Dict) -> Tuple[Dataset, Dataset, Dataset, int]:
    """Load and preprocess dataset."""
    dataset_name = config['data']['name']
    tokenizer = AutoTokenizer.from_pretrained(config['model']['name'])
    
    print(f"Loading {dataset_name} dataset...")
    
    dataset_config = {
        'ag_news': ('ag_news', 'text', 'label', 4),
        'imdb': ('imdb', 'text', 'label', 2),
        'banking77': ('banking77', 'text', 'label', 77)
    }
    
    if dataset_name not in dataset_config:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    
    ds_name, text_key, label_key, num_labels = dataset_config[dataset_name]
    dataset = load_dataset(ds_name)
    
    train_texts = dataset['train'][text_key]
    train_labels = dataset['train'][label_key]
    test_texts = dataset['test'][text_key]
    test_labels = dataset['test'][label_key]
    
    if config['data'].get('train_size'):
        train_texts, train_labels = sample_dataset(
            train_texts, train_labels, config['data']['train_size'], config['training']['seed']
        )
    
    val_size = int(len(train_texts) * config['data']['val_size'])
    indices = np.random.RandomState(config['training']['seed']).permutation(len(train_texts))
    
    train_idx, val_idx = indices[val_size:], indices[:val_size]
    
    return (
        TextDataset([train_texts[i] for i in train_idx], [train_labels[i] for i in train_idx], 
                   tokenizer, config['model']['max_length']),
        TextDataset([train_texts[i] for i in val_idx], [train_labels[i] for i in val_idx],
                   tokenizer, config['model']['max_length']),
        TextDataset(test_texts, test_labels, tokenizer, config['model']['max_length']),
        num_labels
    )


def get_few_shot_dataset(dataset: Dataset, n_samples: int, seed: int = 42) -> Dataset:
    """Create few-shot dataset."""
    np.random.seed(seed)
    label_indices = {}
    
    for idx in range(len(dataset)):
        label = dataset[idx]['labels'].item()
        label_indices.setdefault(label, []).append(idx)
    
    selected = []
    for indices in label_indices.values():
        n = min(n_samples, len(indices))
        selected.extend(np.random.choice(indices, n, replace=False))
    
    np.random.shuffle(selected)
    return Subset(dataset, selected)


def get_dataloader(dataset: Dataset, batch_size: int, shuffle: bool = True, 
                   num_workers: int = 4) -> DataLoader:
    """Create DataLoader."""
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=num_workers > 0
    )