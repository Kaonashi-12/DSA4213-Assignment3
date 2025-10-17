"""
Fine-tuning strategy implementations.
"""

import torch
from transformers import AutoModelForSequenceClassification
from peft import LoraConfig, TaskType, get_peft_model, PromptTuningConfig, PromptTuningInit
from typing import Dict, Optional, List


class BaseFineTuner:
    """Base class for fine-tuning strategies."""
    
    def __init__(self, model_name: str, num_labels: int, config: Dict):
        self.model_name = model_name
        self.num_labels = num_labels
        self.config = config
        self.model = None
        
    def get_model(self):
        raise NotImplementedError
    
    def get_trainable_params(self) -> tuple:
        if self.model is None:
            return 0, 0
        trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.model.parameters())
        return trainable, total


class FullFineTuning(BaseFineTuner):
    """Full fine-tuning."""
    
    def get_model(self):
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name, num_labels=self.num_labels
        )
        return self.model


class LoRAFineTuning(BaseFineTuner):
    """LoRA fine-tuning."""
    
    def __init__(self, model_name: str, num_labels: int, config: Dict, rank: Optional[int] = None):
        super().__init__(model_name, num_labels, config)
        self.rank = rank or config['strategies']['lora']['r']
    
    def get_model(self):
        base_model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name, num_labels=self.num_labels
        )
        
        lora_config = LoraConfig(
            task_type=TaskType.SEQ_CLS,
            r=self.rank,
            lora_alpha=self.config['strategies']['lora']['lora_alpha'],
            lora_dropout=self.config['strategies']['lora']['lora_dropout'],
            target_modules=self.config['strategies']['lora']['target_modules']
        )
        
        self.model = get_peft_model(base_model, lora_config)
        self.model.print_trainable_parameters()
        return self.model


class BitFitFineTuning(BaseFineTuner):
    """BitFit fine-tuning."""
    
    def get_model(self):
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name, num_labels=self.num_labels
        )
        
        for name, param in self.model.named_parameters():
            param.requires_grad = 'bias' in name
        
        return self.model


class PromptTuning(BaseFineTuner):
    """Prompt tuning."""
    
    def get_model(self):
        base_model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name, num_labels=self.num_labels
        )
        
        prompt_config = PromptTuningConfig(
            task_type=TaskType.SEQ_CLS,
            num_virtual_tokens=self.config['strategies']['prompt_tuning']['num_virtual_tokens'],
            prompt_tuning_init=PromptTuningInit.RANDOM,
        )
        
        self.model = get_peft_model(base_model, prompt_config)
        self.model.print_trainable_parameters()
        return self.model


def get_strategy(strategy_name: str, model_name: str, num_labels: int, 
                 config: Dict, **kwargs) -> BaseFineTuner:
    """Factory function for strategies."""
    strategies = {
        'full_finetuning': FullFineTuning,
        'lora': LoRAFineTuning,
        'bitfit': BitFitFineTuning,
        'prompt_tuning': PromptTuning
    }
    
    if strategy_name not in strategies:
        raise ValueError(f"Unknown strategy: {strategy_name}")
    
    strategy_class = strategies[strategy_name]
    
    if strategy_name == 'lora' and 'rank' in kwargs:
        return strategy_class(model_name, num_labels, config, rank=kwargs['rank'])
    return strategy_class(model_name, num_labels, config)


def extract_attention_weights(model, input_ids, attention_mask):
    """Extract attention weights for visualization."""
    if hasattr(model, 'base_model'):
        bert = model.base_model.model.bert if hasattr(model.base_model, 'model') else model.base_model.bert
    else:
        bert = model.bert if hasattr(model, 'bert') else model
    
    outputs = bert(input_ids, attention_mask, output_attentions=True)
    return torch.stack(outputs.attentions)


def extract_layer_features(model, input_ids, attention_mask, layer_indices: List[int]):
    """Extract layer features."""
    if hasattr(model, 'base_model'):
        bert = model.base_model.model.bert if hasattr(model.base_model, 'model') else model.base_model.bert
    else:
        bert = model.bert if hasattr(model, 'bert') else model
    
    outputs = bert(input_ids, attention_mask, output_hidden_states=True)
    
    return {f'layer_{idx}': outputs.hidden_states[idx] 
            for idx in layer_indices if idx < len(outputs.hidden_states)}