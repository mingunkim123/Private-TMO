"""
Configuration management for Privacy-TMO

Handles all configuration including:
- Model settings
- LoRA hyperparameters
- Privacy budget settings
- API keys (from environment variables)
"""

import os
from dataclasses import dataclass, field
from typing import Optional, List, Dict
from pathlib import Path


@dataclass
class LoRAConfig:
    """LoRA training configuration"""
    # Model settings
    base_model: str = "meta-llama/Llama-3.2-3B"  # HuggingFace model ID
    
    # LoRA hyperparameters
    r: int = 8  # LoRA rank
    lora_alpha: int = 16  # LoRA alpha (scaling factor)
    lora_dropout: float = 0.05
    target_modules: List[str] = field(default_factory=lambda: [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj"
    ])
    
    # QLoRA settings (4-bit quantization)
    use_4bit: bool = True
    bnb_4bit_compute_dtype: str = "float16"
    bnb_4bit_quant_type: str = "nf4"  # Normal Float 4
    use_double_quant: bool = True
    
    # Training settings
    learning_rate: float = 2e-4
    batch_size: int = 4
    gradient_accumulation_steps: int = 4
    num_epochs: int = 3
    max_seq_length: int = 512
    warmup_ratio: float = 0.03
    
    # Save settings
    output_dir: str = "./lora_adapters"
    save_steps: int = 100
    logging_steps: int = 10


@dataclass
class PrivacyConfig:
    """Privacy-related configuration"""
    # Privacy budget (epsilon)
    privacy_budget: float = 1.0  # Total allowed privacy leakage
    
    # Sensitivity levels
    sensitivity_levels: int = 3  # 0: Public, 1: Semi-sensitive, 2: Private
    
    # Privacy risk weight in reward function
    privacy_risk_weight: float = 1.0  # β₄ in reward function
    
    # Thresholds for sensitivity classification
    sensitivity_thresholds: Dict[str, float] = field(default_factory=lambda: {
        "public": 0.3,      # Below this -> Public (Level 0)
        "semi_sensitive": 0.7,  # Between public and this -> Semi-sensitive (Level 1)
        # Above semi_sensitive -> Private (Level 2)
    })
    
    # PII entity types to detect
    sensitive_entity_types: List[str] = field(default_factory=lambda: [
        "PER",  # Person names
        "LOC",  # Locations
        "ORG",  # Organizations
        "EMAIL",
        "PHONE",
        "SSN",
        "CREDIT_CARD",
        "PASSWORD",
    ])


@dataclass
class InferenceConfig:
    """Inference configuration"""
    # Local inference (Ollama)
    local_model: str = "llama3.2:3b"
    ollama_host: str = "http://localhost:11434"
    
    # Cloud inference (Groq)
    cloud_model: str = "llama-3.3-70b-versatile"
    groq_api_key: Optional[str] = None  # Loaded from environment
    
    # Timeouts
    local_timeout: int = 60
    cloud_timeout: int = 30
    
    # Retry settings
    max_retries: int = 3
    retry_delay: float = 1.0


@dataclass
class PrivacyTMOConfig:
    """Main configuration class for Privacy-TMO"""
    lora: LoRAConfig = field(default_factory=LoRAConfig)
    privacy: PrivacyConfig = field(default_factory=PrivacyConfig)
    inference: InferenceConfig = field(default_factory=InferenceConfig)
    
    # Paths
    data_dir: str = "./data"
    model_cache_dir: str = "/data/models"
    adapter_dir: str = "./lora_adapters"
    
    # Device settings
    device: str = "cuda"  # "cuda", "cpu", or "auto"
    
    def __post_init__(self):
        """Load sensitive values from environment variables"""
        # Load API key from environment
        self.inference.groq_api_key = os.environ.get("GROQ_API_KEY")
        
        # Create directories if they don't exist
        Path(self.data_dir).mkdir(parents=True, exist_ok=True)
        Path(self.adapter_dir).mkdir(parents=True, exist_ok=True)
    
    @classmethod
    def from_yaml(cls, path: str) -> "PrivacyTMOConfig":
        """Load configuration from YAML file"""
        import yaml
        with open(path, 'r') as f:
            config_dict = yaml.safe_load(f)
        
        lora_config = LoRAConfig(**config_dict.get('lora', {}))
        privacy_config = PrivacyConfig(**config_dict.get('privacy', {}))
        inference_config = InferenceConfig(**config_dict.get('inference', {}))
        
        return cls(
            lora=lora_config,
            privacy=privacy_config,
            inference=inference_config,
            **{k: v for k, v in config_dict.items() 
               if k not in ['lora', 'privacy', 'inference']}
        )
    
    def to_dict(self) -> dict:
        """Convert configuration to dictionary"""
        from dataclasses import asdict
        return asdict(self)


# Default configuration instance
default_config = PrivacyTMOConfig()
