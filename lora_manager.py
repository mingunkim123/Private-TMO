"""
Hierarchical LoRA Manager for Privacy-TMO

Manages three layers of LoRA adapters:
1. Personal Layer - User-specific personalization (highest priority)
2. Group Layer - Domain-specific expertise (coding, medical, cooking, etc.)
3. General Layer - Base model capabilities

Key Features:
- Dynamic adapter selection based on task type and sensitivity
- On-device adapter loading without server communication
- Fallback to system prompts when adapters not available
"""

import os
from pathlib import Path
from typing import Optional, Tuple, Dict, List
from dataclasses import dataclass, field
from enum import Enum


class AdapterLayer(Enum):
    """Adapter hierarchy layers"""
    PERSONAL = "personal"
    GROUP = "group"
    GENERAL = "general"


@dataclass
class AdapterInfo:
    """Information about a loaded adapter"""
    name: str
    path: str
    layer: AdapterLayer
    is_loaded: bool = False
    fallback_prompt: str = ""


class LoRAManager:
    """
    Hierarchical LoRA Adapter Manager
    
    Selects appropriate adapter based on:
    - Task type (coding, medical, cooking, general)
    - Sensitivity level (private data should use Personal adapter)
    - Adapter availability
    """
    
    def __init__(self, adapter_dir: str = "./lora_adapters"):
        self.adapter_dir = Path(adapter_dir)
        self.adapter_dir.mkdir(parents=True, exist_ok=True)
        
        # Base model for Ollama
        self.base_model = "llama3.2:3b"
        
        # Registered adapters
        self.adapters: Dict[str, AdapterInfo] = {}
        
        # Fallback system prompts (used when adapter not available)
        self.fallback_prompts = {
            "general": "You are a helpful AI assistant.",
            
            # Group Layer - Domain specialists
            "coding": "You are an expert programmer. Provide clean, well-documented code with explanations.",
            "medical": "You are a knowledgeable medical consultant. Provide safe, general health advice. Always recommend consulting a doctor for serious concerns.",
            "cooking": "You are a professional chef. Provide detailed recipes with tips for best results.",
            "legal": "You are a legal information assistant. Provide general legal information. Always recommend consulting a lawyer for specific cases.",
            "finance": "You are a financial advisor assistant. Provide general financial guidance. Always recommend consulting a professional for major decisions.",
            
            # Personal Layer
            "personal": "You are a personal assistant with knowledge of the user's preferences, schedule, and history. Provide personalized responses.",
        }
        
        # Domain keywords for task classification
        self.domain_keywords = {
            "coding": ["code", "python", "javascript", "function", "class", "debug", "programming", "algorithm", "api", "database", "sql"],
            "medical": ["pain", "medicine", "symptom", "doctor", "health", "disease", "treatment", "diagnosis", "prescription"],
            "cooking": ["cook", "recipe", "ingredient", "bake", "food", "dish", "meal", "kitchen", "cuisine"],
            "legal": ["law", "legal", "contract", "lawsuit", "attorney", "court", "rights", "agreement"],
            "finance": ["money", "invest", "stock", "budget", "tax", "loan", "savings", "retirement", "portfolio"],
        }
        
        # Sensitivity keywords
        self.sensitivity_keywords = {
            "high": ["password", "secret", "private", "confidential", "ssn", "credit card", "bank account"],
            "medium": ["address", "phone", "email", "schedule", "appointment", "my "],
        }
        
        # Initialize default adapters
        self._initialize_adapters()
    
    def _initialize_adapters(self):
        """Initialize adapter registry with available adapters"""
        # Check for existing adapters in directory
        if self.adapter_dir.exists():
            for adapter_path in self.adapter_dir.iterdir():
                if adapter_path.is_dir():
                    adapter_name = adapter_path.name
                    layer = self._determine_layer(adapter_name)
                    
                    self.adapters[adapter_name] = AdapterInfo(
                        name=adapter_name,
                        path=str(adapter_path),
                        layer=layer,
                        is_loaded=False,
                        fallback_prompt=self.fallback_prompts.get(adapter_name, self.fallback_prompts["general"])
                    )
        
        # Ensure default adapters are registered (even if not trained yet)
        for name, prompt in self.fallback_prompts.items():
            if name not in self.adapters:
                layer = AdapterLayer.PERSONAL if name == "personal" else (
                    AdapterLayer.GENERAL if name == "general" else AdapterLayer.GROUP
                )
                self.adapters[name] = AdapterInfo(
                    name=name,
                    path=str(self.adapter_dir / name),
                    layer=layer,
                    is_loaded=False,
                    fallback_prompt=prompt
                )
    
    def _determine_layer(self, adapter_name: str) -> AdapterLayer:
        """Determine which layer an adapter belongs to"""
        if adapter_name == "personal":
            return AdapterLayer.PERSONAL
        elif adapter_name == "general":
            return AdapterLayer.GENERAL
        else:
            return AdapterLayer.GROUP
    
    def classify_task(self, prompt: str) -> str:
        """
        Classify the task type based on prompt content
        
        Args:
            prompt: User's input prompt
            
        Returns:
            Task type (e.g., "coding", "medical", "general")
        """
        prompt_lower = prompt.lower()
        
        # Check each domain
        domain_scores = {}
        for domain, keywords in self.domain_keywords.items():
            score = sum(1 for kw in keywords if kw in prompt_lower)
            if score > 0:
                domain_scores[domain] = score
        
        # Return domain with highest score, or "general" if none match
        if domain_scores:
            return max(domain_scores.items(), key=lambda x: x[1])[0]
        return "general"
    
    def assess_sensitivity(self, prompt: str) -> Tuple[bool, float]:
        """
        Assess the sensitivity level of a prompt
        
        Args:
            prompt: User's input prompt
            
        Returns:
            Tuple of (is_sensitive, sensitivity_score)
            - is_sensitive: True if contains sensitive information
            - sensitivity_score: 0.0 (public) to 1.0 (highly private)
        """
        prompt_lower = prompt.lower()
        
        # Check high sensitivity keywords
        for kw in self.sensitivity_keywords["high"]:
            if kw in prompt_lower:
                return True, 1.0
        
        # Check medium sensitivity keywords
        for kw in self.sensitivity_keywords["medium"]:
            if kw in prompt_lower:
                return True, 0.6
        
        return False, 0.0
    
    def select_adapter(
        self, 
        task_type: str, 
        is_sensitive: bool = False,
        sensitivity_score: float = 0.0
    ) -> Tuple[str, str, str]:
        """
        Select appropriate adapter based on task and sensitivity
        
        Args:
            task_type: Type of task (e.g., "coding", "medical")
            is_sensitive: Whether the prompt contains sensitive information
            sensitivity_score: Sensitivity score (0.0 to 1.0)
            
        Returns:
            Tuple of (model_name, system_prompt, layer_name)
        """
        selected_layer = AdapterLayer.GENERAL
        selected_adapter = "general"
        
        # Priority 1: Personal Layer for sensitive data
        if is_sensitive or sensitivity_score > 0.5:
            if self._is_adapter_available("personal"):
                selected_layer = AdapterLayer.PERSONAL
                selected_adapter = "personal"
                print(f"🔒 [Manager] Sensitive data detected (score: {sensitivity_score:.2f}) -> Personal Layer")
            else:
                print(f"🔒 [Manager] Sensitive data detected, but Personal adapter not available -> Using fallback")
                selected_adapter = "personal"  # Use fallback prompt
        
        # Priority 2: Group Layer for domain-specific tasks
        elif task_type in self.adapters and task_type != "general":
            if self._is_adapter_available(task_type):
                selected_layer = AdapterLayer.GROUP
                selected_adapter = task_type
                print(f"🎯 [Manager] Domain task ({task_type}) -> Group Layer")
            else:
                print(f"🎯 [Manager] Domain task ({task_type}), adapter not available -> Using fallback")
                selected_adapter = task_type  # Use fallback prompt
        
        # Priority 3: General Layer
        else:
            print(f"📋 [Manager] General task -> General Layer")
        
        # Get model name and system prompt
        adapter_info = self.adapters.get(selected_adapter, self.adapters["general"])
        model_name = self._get_ollama_model_name(selected_adapter)
        system_prompt = adapter_info.fallback_prompt
        
        return model_name, system_prompt, selected_layer.value.capitalize()
    
    def _is_adapter_available(self, adapter_name: str) -> bool:
        """Check if a trained adapter is available"""
        if adapter_name not in self.adapters:
            return False
        
        adapter_path = Path(self.adapters[adapter_name].path)
        
        # Check if adapter files exist
        return (adapter_path / "final").exists() or (adapter_path / "adapter_model.bin").exists()
    
    def _get_ollama_model_name(self, adapter_name: str) -> str:
        """
        Get Ollama model name for an adapter
        
        If adapter is trained and registered with Ollama, return custom model name.
        Otherwise, return base model name.
        """
        # Check if custom model exists in Ollama
        custom_model_name = f"privacy-tmo-{adapter_name}"
        
        # For now, always return base model
        # TODO: Check Ollama for registered custom models
        return self.base_model
    
    def register_adapter(
        self, 
        name: str, 
        path: str, 
        layer: AdapterLayer,
        fallback_prompt: str = ""
    ):
        """
        Register a new adapter
        
        Args:
            name: Adapter name
            path: Path to adapter files
            layer: Which layer this adapter belongs to
            fallback_prompt: System prompt to use if adapter fails
        """
        self.adapters[name] = AdapterInfo(
            name=name,
            path=path,
            layer=layer,
            is_loaded=False,
            fallback_prompt=fallback_prompt or self.fallback_prompts.get(name, self.fallback_prompts["general"])
        )
        print(f"✅ Registered adapter: {name} ({layer.value})")
    
    def list_adapters(self) -> List[Dict]:
        """List all registered adapters with their status"""
        result = []
        for name, info in self.adapters.items():
            result.append({
                "name": name,
                "layer": info.layer.value,
                "path": info.path,
                "available": self._is_adapter_available(name),
            })
        return result
    
    def get_adapter_status(self) -> str:
        """Get a formatted status report of all adapters"""
        lines = ["📊 Adapter Status:"]
        lines.append("-" * 50)
        
        for layer in AdapterLayer:
            layer_adapters = [a for a in self.adapters.values() if a.layer == layer]
            lines.append(f"\n{layer.value.upper()} LAYER:")
            
            for adapter in layer_adapters:
                available = "✅" if self._is_adapter_available(adapter.name) else "❌"
                lines.append(f"  {available} {adapter.name}")
        
        return "\n".join(lines)


# Convenience function for quick adapter selection
def get_adapter_for_prompt(prompt: str, manager: Optional[LoRAManager] = None) -> Tuple[str, str, str]:
    """
    Quick function to get appropriate adapter for a prompt
    
    Args:
        prompt: User's input prompt
        manager: LoRAManager instance (creates new if not provided)
        
    Returns:
        Tuple of (model_name, system_prompt, layer_name)
    """
    if manager is None:
        manager = LoRAManager()
    
    task_type = manager.classify_task(prompt)
    is_sensitive, sensitivity_score = manager.assess_sensitivity(prompt)
    
    return manager.select_adapter(task_type, is_sensitive, sensitivity_score)


if __name__ == "__main__":
    # Test the LoRA Manager
    print("🧪 Testing LoRA Manager\n")
    
    manager = LoRAManager()
    
    # Print adapter status
    print(manager.get_adapter_status())
    print()
    
    # Test prompts
    test_prompts = [
        "Write a Python function to sort a list",
        "What's the best treatment for a headache?",
        "How do I make pasta carbonara?",
        "What is my password for the bank account?",
        "Tell me about machine learning",
    ]
    
    print("\n🔍 Testing adapter selection:")
    print("-" * 50)
    
    for prompt in test_prompts:
        print(f"\nPrompt: {prompt[:50]}...")
        model, sys_prompt, layer = get_adapter_for_prompt(prompt, manager)
        print(f"  → Layer: {layer}, Model: {model}")
        print(f"  → System prompt: {sys_prompt[:50]}...")
