# Privacy-TMO

**Privacy-Preserving Personalized LLM Offloading for Edge-Cloud Collaboration**

[![Python 3.10](https://img.shields.io/badge/Python-3.10-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> 기존 [TMO (MobiHoc 2025)](./TMO/README.md) 프레임워크를 확장하여, **프라이버시 보호**와 **On-Device 개인화**를 추가한 Edge-Cloud LLM 오프로딩 시스템

---

## Overview

Privacy-TMO는 민감한 사용자 데이터를 보호하면서 고품질 LLM 응답을 제공합니다.

```
┌─────────────────────────────────────────────────────────────────┐
│                        User Query                               │
│           "My password is secret123. What is Python?"           │
└─────────────────────────────┬───────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                  Sensitivity Classifier                         │
│                 (Rule + NER + ML Hybrid)                        │
│                                                                 │
│    🟢 PUBLIC    🟡 SEMI-SENSITIVE    🔴 PRIVATE                │
└─────────────────────────────┬───────────────────────────────────┘
                              │
              ┌───────────────┼───────────────┐
              ▼               ▼               ▼
        ┌──────────┐   ┌──────────────┐   ┌──────────┐
        │  Cloud   │   │   Hybrid     │   │  Local   │
        │   LLM    │   │ (Selective)  │   │   LLM    │
        └──────────┘   └──────────────┘   └──────────┘
              │               │               │
              └───────────────┼───────────────┘
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                  Response Aggregation                           │
│              Privacy-Preserving Final Output                    │
└─────────────────────────────────────────────────────────────────┘
```

---

## Key Features

### 1. On-Device LoRA Personalization
- **QLoRA (4-bit)** 양자화로 Jetson 8GB에서 학습 가능
- 개인 데이터는 **절대로 디바이스를 떠나지 않음**
- Personal / Group / General 계층적 어댑터 관리

### 2. Sensitivity-Aware Selective Offloading
- **3단계 민감도 분류**: Public / Semi-sensitive / Private
- **쿼리 분해**: 민감한 부분만 로컬에서 처리
- **Partial Offloading**: Binary 결정이 아닌 세밀한 제어

### 3. Privacy-Aware Reinforcement Learning
- **확장된 보상 함수**: 기존 TMO + Privacy Risk 패널티
- **Privacy Budget**: ε-differential privacy 스타일 제약
- **Lagrangian Relaxation**으로 제약 조건 처리

### 4. Comprehensive Evaluation
- **Privacy Attack Simulation**: Canary Insertion, Membership Inference
- **Baseline Comparison**: No Protection, Local Only, Threshold-based
- **Jetson Profiling**: 지연시간, 메모리, 전력 측정

---

## Installation

```bash
# Clone repository
git clone https://github.com/your-repo/Privacy-TMO.git
cd Privacy-TMO

# Install dependencies
pip install -r requirements.txt

# (Optional) For Jetson deployment
pip install pynvml  # GPU monitoring
```

### Requirements
- Python >= 3.10
- PyTorch >= 2.2.0
- Transformers >= 4.36.0
- PEFT >= 0.7.0 (for LoRA)
- stable-baselines3 >= 2.2.1

---

## Project Structure

```
Privacy-TMO/
├── privacy_tmo/                    # Core module
│   ├── config.py                   # Configuration management
│   ├── lora_trainer.py             # On-device LoRA training (QLoRA)
│   ├── sensitivity_classifier.py   # 3-level sensitivity classification
│   ├── privacy_manager.py          # Privacy budget management
│   ├── query_decomposer.py         # Query decomposition & selective offloading
│   ├── privacy_rl.py               # Privacy-aware RL (extended reward)
│   ├── response_aggregator.py      # Hybrid response aggregation
│   ├── privacy_attacks.py          # Attack simulations (Canary, MIA)
│   ├── benchmark.py                # Benchmarking suite
│   └── profiler.py                 # Performance profiler
│
├── lora_manager.py                 # Hierarchical LoRA adapter manager
├── tmo_interface.py                # Inference interface (Ollama + Groq)
├── requirements.txt                # Dependencies
│
└── TMO/                            # Original TMO framework
    └── main/
        ├── main.py                 # Training entry point
        ├── models.py               # RC_PPO, RC_A2C, RC_DQN
        └── utils.py                # M4A1 Environment
```

---

## Quick Start

### 1. Basic Usage

```python
from privacy_tmo import (
    PrivacyManager,
    SensitivityClassifier,
    QueryDecomposer,
    HybridInferenceEngine
)

# Initialize components
privacy_manager = PrivacyManager()
classifier = SensitivityClassifier()

# Classify query sensitivity
query = "My password is secret123. What is Python?"
result = classifier.classify(query)

print(f"Level: {result.level.name}")  # SEMI_SENSITIVE
print(f"Score: {result.score:.2f}")   # 0.75

# Make offloading decision
decision = privacy_manager.make_offloading_decision(query)
print(f"Decision: {decision.decision.value}")  # hybrid
```

### 2. Train Personal LoRA

```python
from privacy_tmo import LoRATrainer, train_personal_lora

# Quick training
adapter_path = train_personal_lora(
    user_data_path="./data/user_history.json",
    output_dir="./lora_adapters/personal"
)

# Or with full control
trainer = LoRATrainer()
trainer.setup_model("meta-llama/Llama-3.2-3B")
trainer.setup_lora(adapter_name="personal")
trainer.prepare_dataset("./data/user_history.json")
trainer.train()
```

### 3. Run Benchmark

```python
from privacy_tmo import BenchmarkSuite, BenchmarkConfig

config = BenchmarkConfig(
    num_episodes=100,
    privacy_budgets=[0.3, 0.5, 0.7, 1.0]
)

suite = BenchmarkSuite(config)
results = suite.run_benchmark()
print(suite.generate_report())
```

### 4. Privacy Attack Evaluation

```python
from privacy_tmo import PrivacyAttackSimulator

simulator = PrivacyAttackSimulator()

# Prepare canary attack
canaries = simulator.prepare_canary_attack(num_canaries=10)

# Run attack
result = simulator.run_canary_attack(inference_fn)
print(f"Extraction rate: {result.success_rate:.2%}")
```

---

## Technical Contributions

### Extended Reward Function

**Original TMO:**
```
R = α·Quality + β₁·Association - β₂·Latency - β₃·Cost
```

**Privacy-TMO:**
```
R = α·Quality + β₁·Association - β₂·Latency - β₃·Cost 
    - β₄·PrivacyRisk + γ·BudgetBonus

subject to: Σₜ PrivacyRisk(qₜ, aₜ) ≤ ε
```

### Sensitivity Classification

| Level | Description | Action |
|-------|-------------|--------|
| 🟢 PUBLIC | General knowledge queries | Cloud OK |
| 🟡 SEMI-SENSITIVE | Context-dependent, some PII | Hybrid |
| 🔴 PRIVATE | Contains passwords, SSN, etc. | Local Only |

### Query Decomposition Strategies

| Strategy | Use Case | Example |
|----------|----------|---------|
| Sentence | Multi-sentence queries | Split by sentence, route separately |
| Entity | Clear PII entities | Mask entities, send masked version |
| Clause | Complex single sentence | Split by clauses |

---

## Benchmark Results

```

```

---

## Hardware Requirements



---

## References

- **TMO**: Local-Cloud Inference Offloading for LLMs (MobiHoc 2025)
- **FrugalGPT**: How to Use LLMs While Reducing Cost
- **QLoRA**: Efficient Finetuning of Quantized LLMs
- **PEFT**: Parameter-Efficient Fine-Tuning

---

## Citation

```bibtex
@article{privacy-tmo,
  title={Privacy-Preserving Personalized LLM Offloading for Edge-Cloud Collaboration},
  author={},
  year={2025}
}
```

---

## License

MIT License - see [LICENSE](./TMO/LICENSE) for details.
