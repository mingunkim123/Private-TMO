"""
LoRA Training Pipeline for On-Device Personalization

This module implements:
1. QLoRA training for memory-efficient fine-tuning
2. Personal LoRA adapter creation from user conversation history
3. Group LoRA adapter creation from domain-specific data
4. Adapter management (save, load, merge)
"""

import os
import json
import torch
from pathlib import Path
from typing import Optional, List, Dict, Union
from dataclasses import dataclass

from .config import LoRAConfig, PrivacyTMOConfig


@dataclass
class ConversationSample:
    """Single conversation sample for training"""
    instruction: str
    input: str = ""
    output: str = ""
    
    def to_prompt(self, include_response: bool = True) -> str:
        """Convert to training prompt format"""
        if self.input:
            prompt = f"### Instruction:\n{self.instruction}\n\n### Input:\n{self.input}\n\n### Response:\n"
        else:
            prompt = f"### Instruction:\n{self.instruction}\n\n### Response:\n"
        
        if include_response:
            prompt += self.output
        return prompt


class LoRATrainer:
    """
    LoRA Training Pipeline for Privacy-Preserving Personalization
    
    Key features:
    - QLoRA (4-bit quantization + LoRA) for memory efficiency
    - On-device training without sending data to server
    - Support for Personal and Group adapters
    """
    
    def __init__(self, config: Optional[PrivacyTMOConfig] = None):
        self.config = config or PrivacyTMOConfig()
        self.lora_config = self.config.lora
        
        self.model = None
        self.tokenizer = None
        self.peft_model = None
        
        self._check_dependencies()
    
    def _check_dependencies(self):
        """Check if required packages are installed"""
        required_packages = {
            'transformers': 'transformers',
            'peft': 'peft',
            'bitsandbytes': 'bitsandbytes',
            'datasets': 'datasets',
            'trl': 'trl',
        }
        
        missing = []
        for package, pip_name in required_packages.items():
            try:
                __import__(package)
            except ImportError:
                missing.append(pip_name)
        
        if missing:
            print(f"⚠️ Missing packages: {missing}")
            print(f"Install with: pip install {' '.join(missing)}")
    
    def setup_model(self, model_name: Optional[str] = None):
        """
        Load base model with QLoRA quantization
        
        Args:
            model_name: HuggingFace model ID (default from config)
        """
        from transformers import (
            AutoModelForCausalLM,
            AutoTokenizer,
            BitsAndBytesConfig,
        )
        
        model_name = model_name or self.lora_config.base_model
        
        print(f"🔧 Loading base model: {model_name}")
        
        # QLoRA quantization config
        if self.lora_config.use_4bit:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=self.lora_config.use_double_quant,
                bnb_4bit_quant_type=self.lora_config.bnb_4bit_quant_type,
                bnb_4bit_compute_dtype=getattr(torch, self.lora_config.bnb_4bit_compute_dtype),
            )
        else:
            bnb_config = None
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            cache_dir=self.config.model_cache_dir,
            trust_remote_code=True,
        )
        
        # Set padding token if not exists
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        
        # Load model with quantization
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=bnb_config,
            device_map="auto" if self.config.device == "auto" else {"": self.config.device},
            cache_dir=self.config.model_cache_dir,
            trust_remote_code=True,
        )
        
        # Enable gradient checkpointing for memory efficiency
        self.model.gradient_checkpointing_enable()
        
        print(f"✅ Model loaded successfully")
        self._print_model_info()
    
    def _print_model_info(self):
        """Print model memory usage information"""
        if self.model is None:
            return
        
        # Count parameters
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        print(f"📊 Model Info:")
        print(f"   Total parameters: {total_params:,}")
        print(f"   Trainable parameters: {trainable_params:,}")
        print(f"   Trainable %: {100 * trainable_params / total_params:.2f}%")
        
        # GPU memory
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / 1024**3
            reserved = torch.cuda.memory_reserved() / 1024**3
            print(f"   GPU Memory: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved")
    
    def setup_lora(self, adapter_name: str = "default"):
        """
        Configure LoRA adapter for training
        
        Args:
            adapter_name: Name for this adapter (e.g., "personal", "coding", "medical")
        """
        from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
        
        if self.model is None:
            raise ValueError("Model not loaded. Call setup_model() first.")
        
        # Prepare model for k-bit training
        self.model = prepare_model_for_kbit_training(self.model)
        
        # LoRA configuration
        lora_config = LoraConfig(
            r=self.lora_config.r,
            lora_alpha=self.lora_config.lora_alpha,
            lora_dropout=self.lora_config.lora_dropout,
            target_modules=self.lora_config.target_modules,
            bias="none",
            task_type="CAUSAL_LM",
        )
        
        # Apply LoRA to model
        self.peft_model = get_peft_model(self.model, lora_config, adapter_name=adapter_name)
        
        print(f"✅ LoRA adapter '{adapter_name}' configured")
        self._print_model_info()
    
    def prepare_dataset(
        self, 
        data: Union[List[ConversationSample], List[Dict], str],
        validation_split: float = 0.1
    ):
        """
        Prepare dataset for training
        
        Args:
            data: Training data - can be:
                - List of ConversationSample objects
                - List of dicts with 'instruction', 'input', 'output' keys
                - Path to JSON file
            validation_split: Fraction of data for validation
        """
        from datasets import Dataset
        
        # Load data if path provided
        if isinstance(data, str):
            with open(data, 'r') as f:
                data = json.load(f)
        
        # Convert to ConversationSample if needed
        if data and isinstance(data[0], dict):
            data = [ConversationSample(**d) for d in data]
        
        # Convert to prompts
        prompts = [{"text": sample.to_prompt(include_response=True)} for sample in data]
        
        # Create dataset
        dataset = Dataset.from_list(prompts)
        
        # Split into train/validation
        if validation_split > 0:
            split = dataset.train_test_split(test_size=validation_split)
            self.train_dataset = split['train']
            self.eval_dataset = split['test']
        else:
            self.train_dataset = dataset
            self.eval_dataset = None
        
        print(f"📚 Dataset prepared:")
        print(f"   Training samples: {len(self.train_dataset)}")
        if self.eval_dataset:
            print(f"   Validation samples: {len(self.eval_dataset)}")
    
    def train(
        self, 
        output_dir: Optional[str] = None,
        adapter_name: str = "personal"
    ) -> str:
        """
        Train LoRA adapter
        
        Args:
            output_dir: Directory to save adapter
            adapter_name: Name for the adapter
            
        Returns:
            Path to saved adapter
        """
        from transformers import TrainingArguments
        from trl import SFTTrainer
        
        if self.peft_model is None:
            raise ValueError("LoRA not configured. Call setup_lora() first.")
        
        if not hasattr(self, 'train_dataset'):
            raise ValueError("Dataset not prepared. Call prepare_dataset() first.")
        
        output_dir = output_dir or os.path.join(self.config.adapter_dir, adapter_name)
        
        # Training arguments
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=self.lora_config.num_epochs,
            per_device_train_batch_size=self.lora_config.batch_size,
            gradient_accumulation_steps=self.lora_config.gradient_accumulation_steps,
            learning_rate=self.lora_config.learning_rate,
            warmup_ratio=self.lora_config.warmup_ratio,
            logging_steps=self.lora_config.logging_steps,
            save_steps=self.lora_config.save_steps,
            save_total_limit=2,
            fp16=True,
            optim="paged_adamw_8bit",
            report_to="none",  # Disable wandb/tensorboard for privacy
        )
        
        # Trainer
        trainer = SFTTrainer(
            model=self.peft_model,
            args=training_args,
            train_dataset=self.train_dataset,
            eval_dataset=self.eval_dataset,
            tokenizer=self.tokenizer,
            dataset_text_field="text",
            max_seq_length=self.lora_config.max_seq_length,
        )
        
        print(f"🚀 Starting LoRA training for '{adapter_name}'...")
        trainer.train()
        
        # Save adapter
        adapter_path = os.path.join(output_dir, "final")
        self.peft_model.save_pretrained(adapter_path)
        self.tokenizer.save_pretrained(adapter_path)
        
        print(f"✅ Adapter saved to: {adapter_path}")
        return adapter_path
    
    def load_adapter(self, adapter_path: str, adapter_name: str = "loaded"):
        """
        Load a pre-trained LoRA adapter
        
        Args:
            adapter_path: Path to saved adapter
            adapter_name: Name for the loaded adapter
        """
        from peft import PeftModel
        
        if self.model is None:
            raise ValueError("Model not loaded. Call setup_model() first.")
        
        self.peft_model = PeftModel.from_pretrained(
            self.model,
            adapter_path,
            adapter_name=adapter_name,
        )
        
        print(f"✅ Adapter loaded from: {adapter_path}")
    
    def merge_and_export(self, output_path: str):
        """
        Merge LoRA adapter with base model and export
        
        Useful for creating standalone models without adapter overhead
        
        Args:
            output_path: Path to save merged model
        """
        if self.peft_model is None:
            raise ValueError("No adapter loaded")
        
        print("🔄 Merging adapter with base model...")
        merged_model = self.peft_model.merge_and_unload()
        
        merged_model.save_pretrained(output_path)
        self.tokenizer.save_pretrained(output_path)
        
        print(f"✅ Merged model saved to: {output_path}")
    
    def export_to_ollama(self, adapter_path: str, model_name: str) -> str:
        """
        Export LoRA adapter for use with Ollama
        
        Creates a Modelfile and registers with Ollama
        
        Args:
            adapter_path: Path to LoRA adapter
            model_name: Name for the Ollama model
            
        Returns:
            Ollama model name
        """
        # First, merge adapter with base model
        merged_path = os.path.join(adapter_path, "merged")
        self.merge_and_export(merged_path)
        
        # Create Modelfile for Ollama
        modelfile_content = f"""FROM {merged_path}

PARAMETER temperature 0.7
PARAMETER top_p 0.9
PARAMETER stop "### Instruction:"
PARAMETER stop "### Response:"
"""
        
        modelfile_path = os.path.join(adapter_path, "Modelfile")
        with open(modelfile_path, 'w') as f:
            f.write(modelfile_content)
        
        print(f"📝 Modelfile created at: {modelfile_path}")
        print(f"To register with Ollama, run:")
        print(f"  ollama create {model_name} -f {modelfile_path}")
        
        return model_name
    
    def create_sample_training_data(self, output_path: str, num_samples: int = 100):
        """
        Create sample training data for testing the pipeline
        
        Args:
            output_path: Path to save sample data
            num_samples: Number of samples to generate
        """
        samples = []
        
        # Personal preference samples
        personal_samples = [
            {
                "instruction": "What is my favorite programming language?",
                "input": "",
                "output": "Based on your history, your favorite programming language is Python. You frequently use it for machine learning projects and prefer its clean syntax."
            },
            {
                "instruction": "Summarize my schedule for today",
                "input": "",
                "output": "Today you have: 9 AM - Team meeting, 2 PM - Code review, 5 PM - Gym session. Would you like me to add any reminders?"
            },
            {
                "instruction": "What restaurants do I usually order from?",
                "input": "",
                "output": "Your most frequent orders are from: 1) Thai Basil (Thai food), 2) Sushi Palace (Japanese), 3) Pizza Corner (Italian). You usually order on Friday evenings."
            },
        ]
        
        # Domain-specific samples (coding)
        coding_samples = [
            {
                "instruction": "Write a Python function to sort a list",
                "input": "Sort in descending order",
                "output": "```python\ndef sort_descending(lst):\n    return sorted(lst, reverse=True)\n\n# Example usage\nnumbers = [3, 1, 4, 1, 5, 9, 2, 6]\nprint(sort_descending(numbers))  # [9, 6, 5, 4, 3, 2, 1, 1]\n```"
            },
            {
                "instruction": "Explain list comprehension in Python",
                "input": "",
                "output": "List comprehension is a concise way to create lists in Python. Syntax: [expression for item in iterable if condition]\n\nExample:\n```python\n# Traditional way\nsquares = []\nfor x in range(10):\n    squares.append(x**2)\n\n# List comprehension\nsquares = [x**2 for x in range(10)]\n```"
            },
        ]
        
        # Extend samples
        samples.extend(personal_samples * (num_samples // 5))
        samples.extend(coding_samples * (num_samples // 5))
        
        # Save to file
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(samples[:num_samples], f, indent=2)
        
        print(f"✅ Sample training data saved to: {output_path}")
        print(f"   Total samples: {min(num_samples, len(samples))}")
        
        return output_path


def train_personal_lora(
    user_data_path: str,
    output_dir: str = "./lora_adapters/personal",
    config: Optional[PrivacyTMOConfig] = None
) -> str:
    """
    Convenience function to train a personal LoRA adapter
    
    Args:
        user_data_path: Path to user's conversation history (JSON)
        output_dir: Directory to save the adapter
        config: Configuration object
        
    Returns:
        Path to the trained adapter
    """
    trainer = LoRATrainer(config)
    trainer.setup_model()
    trainer.setup_lora(adapter_name="personal")
    trainer.prepare_dataset(user_data_path)
    return trainer.train(output_dir=output_dir, adapter_name="personal")


def train_group_lora(
    domain: str,
    data_path: str,
    output_dir: str = "./lora_adapters",
    config: Optional[PrivacyTMOConfig] = None
) -> str:
    """
    Convenience function to train a domain-specific group LoRA adapter
    
    Args:
        domain: Domain name (e.g., "coding", "medical", "cooking")
        data_path: Path to domain-specific training data
        output_dir: Base directory for adapters
        config: Configuration object
        
    Returns:
        Path to the trained adapter
    """
    trainer = LoRATrainer(config)
    trainer.setup_model()
    trainer.setup_lora(adapter_name=domain)
    trainer.prepare_dataset(data_path)
    return trainer.train(
        output_dir=os.path.join(output_dir, domain),
        adapter_name=domain
    )


if __name__ == "__main__":
    # Test the training pipeline
    print("🧪 Testing LoRA Training Pipeline")
    
    config = PrivacyTMOConfig()
    trainer = LoRATrainer(config)
    
    # Create sample data
    sample_data_path = "./data/sample_training_data.json"
    trainer.create_sample_training_data(sample_data_path, num_samples=50)
    
    print("\n✅ LoRA Training Pipeline ready!")
    print("To train a personal adapter, run:")
    print("  python -m privacy_tmo.lora_trainer --train --data ./data/user_history.json")
