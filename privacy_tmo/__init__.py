"""
Privacy-TMO: Privacy-Preserving Personalized LLM Offloading

This module implements:
1. On-Device LoRA Personalization (Phase 1)
2. Sensitivity-Aware Selective Offloading (Phase 2)
3. Privacy-Aware RL Policy (Phase 2)
4. Privacy Attack Evaluation (Phase 3)
5. Benchmarking & Profiling (Phase 3)
"""

# Phase 1: Core Components
from .config import PrivacyTMOConfig, LoRAConfig, PrivacyConfig
from .lora_trainer import LoRATrainer, train_personal_lora, train_group_lora
from .sensitivity_classifier import (
    SensitivityClassifier, 
    SensitivityLevel, 
    SensitivityResult,
    MultimodalSensitivityClassifier,
    MultimodalSensitivityResult,
)
from .image_sensitivity import ImageSensitivityClassifier
from .privacy_manager import (
    PrivacyManager, 
    PrivacyBudget, 
    MultimodalSensitivity,
    ACTION_TO_MODALITY_INDICES,
)

# Phase 2: Algorithms
from .query_decomposer import (
    QueryDecomposer, 
    SelectiveOffloader, 
    DecomposedQuery,
    MultimodalQueryDecomposer,
    MultimodalDecomposedQuery,
)
from .privacy_rl import (
    PrivacyRLConfig, 
    PrivacyAwareEnv, 
    PrivacyConstrainedPPO,
    create_privacy_reward_function
)
from .response_aggregator import (
    ResponseAggregator, 
    AggregationStrategy,
    HybridInferenceEngine
)

# Phase 3: Evaluation
from .privacy_attacks import (
    CanaryInsertionAttack,
    MembershipInferenceAttack,
    PrivacyAttackSimulator
)
from .benchmark import BenchmarkSuite, BenchmarkConfig
from .profiler import PerformanceProfiler

__version__ = "0.1.0"
__all__ = [
    # Config
    "PrivacyTMOConfig",
    "LoRAConfig", 
    "PrivacyConfig",
    
    # Phase 1
    "LoRATrainer",
    "train_personal_lora",
    "train_group_lora",
    "SensitivityClassifier",
    "SensitivityLevel",
    "SensitivityResult",
    "MultimodalSensitivityClassifier",
    "MultimodalSensitivityResult",
    "ImageSensitivityClassifier",
    "PrivacyManager",
    "PrivacyBudget",
    "MultimodalSensitivity",
    "ACTION_TO_MODALITY_INDICES",
    
    # Phase 2
    "QueryDecomposer",
    "SelectiveOffloader",
    "DecomposedQuery",
    "MultimodalQueryDecomposer",
    "MultimodalDecomposedQuery",
    "PrivacyRLConfig",
    "PrivacyAwareEnv",
    "PrivacyConstrainedPPO",
    "create_privacy_reward_function",
    "ResponseAggregator",
    "AggregationStrategy",
    "HybridInferenceEngine",
    
    # Phase 3
    "CanaryInsertionAttack",
    "MembershipInferenceAttack",
    "PrivacyAttackSimulator",
    "BenchmarkSuite",
    "BenchmarkConfig",
    "PerformanceProfiler",
]
