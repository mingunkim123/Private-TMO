<div align="center">

# Privacy-TMO

**Privacy-Preserving Personalized LLM Offloading for Edge-Cloud Collaboration**

[![Python 3.10](https://img.shields.io/badge/Python-3.10-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.2+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

*Extending [TMO (MobiHoc 2025 Best Paper Runner-Up)](./TMO/README.md) with privacy protection, on-device personalization, and multimodal sensitivity-aware offloading*

</div>

---

## Overview

Privacy-TMO optimizes **quality, latency, cost, and privacy** simultaneously for multimodal LLM inference.  
The RL agent learns to select the optimal modality combination (action 0–8) while minimizing privacy leakage.

### Key Contributions

| Contribution | Description |
|--------------|-------------|
| **On-Device LoRA** | QLoRA-based personalization; user data never leaves the device |
| **Sensitivity Classification** | 3-level text classification (Public / Semi-sensitive / Private) |
| **Multimodal Privacy** | Per-image sensitivity analysis with modality-aware routing |
| **Privacy Budget** | ε-differential privacy style constraint on cumulative risk |

---

## System Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              User Query + Images                            │
└─────────────────────────────────────┬───────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                         Sensitivity Analysis                                │
│  ┌─────────────────────┐    ┌───────────────────────────────────────────┐   │
│  │   Text Classifier   │    │         Image Classifier (x3)              │   │
│  │  Rule + NER + ML    │    │   Face Detection / OCR / Simulation       │   │
│  │                     │    │                                           │   │
│  │  PUBLIC | SEMI |    │    │   img0: 0.4   img1: 0.2   img2: 0.7       │   │
│  │       PRIVATE       │    │                                           │   │
│  └──────────┬──────────┘    └──────────────────┬────────────────────────┘   │
│             │                                  │                            │
│             └──────────────┬───────────────────┘                             │
│                            ▼                                                │
│              ┌───────────────────────────────┐                              │
│              │   Multimodal Sensitivity      │                              │
│              │   + Modality Risk Calc        │                              │
│              └───────────────┬───────────────┘                              │
└─────────────────────────────┼──────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                          RL Agent (RC_PPO/A2C/DQN)                          │
│                                                                             │
│   State: [history] + [text_sens, text_score, budget] + [img0, img1, img2]   │
│                                                                             │
│   Action: 0 (Local) | 1-8 (Cloud + Modality Combinations)                  │
└─────────────────────────────────────┬───────────────────────────────────────┘
                                      │
                    ┌─────────────────┴─────────────────┐
                    ▼                                   ▼
          ┌─────────────────┐                 ┌─────────────────┐
          │   Local LLM     │                 │   Cloud LLM     │
          │  (Ollama+LoRA)  │                 │   (Groq API)    │
          └────────┬────────┘                 └────────┬────────┘
                   │                                   │
                   └─────────────┬─────────────────────┘
                                 ▼
                    ┌─────────────────────────┐
                    │  Response Aggregation   │
                    └─────────────┬───────────┘
                                  │
                                  ▼
                    ┌─────────────────────────┐
                    │      Reward Calc        │
                    │  Quality + Association  │
                    │  - Latency - Cost       │
                    │  - TextRisk             │
                    │  - ModalityRisk         │
                    │  + BudgetBonus          │
                    └─────────────────────────┘
```

---

## Data Flow (Detailed)

### 1. Initialization

```
main.py
├── args_parser()
│   └── options.py (load configuration)
│
├── M4A1_Env(dataset, weights, ...)
│   ├── PrivacyManager(enable_ner, enable_ml, enable_image_sensitivity)
│   │   ├── SensitivityClassifier
│   │   └── ImageSensitivityClassifier (optional)
│   │
│   ├── QueryDecomposer(classifier)
│   ├── ResponseAggregator()
│   │
│   └── observation_space
│       └── base(5×time_span) + privacy(3) + image_sens(3)
│           = 25 + 3 + 3 = 31 dimensions (default)
│
└── RL Model (RC_PPO / RC_A2C / RC_DQN)
    └── MlpPolicy with resource_constraint()
```

### 2. Training Loop (Each Step)

```
RL Agent
│
├── model.predict(observation)
│   └── action ∈ {0, 1, 2, ..., 8}
│
└── env.step(action)
    │
    ├── [1] Extract Prompt
    │   └── _get_current_prompt()
    │
    ├── [2] Text Sensitivity Analysis
    │   └── PrivacyManager.analyze_query(prompt)
    │       └── SensitivityClassifier.classify()
    │           ├── RuleBasedDetector (regex patterns)
    │           ├── NERBasedDetector (BERT-based)
    │           └── Weighted voting → SensitivityResult
    │
    ├── [3] Image Sensitivity Analysis (if enabled)
    │   └── PrivacyManager.analyze_multimodal(text, images, simulate)
    │       └── ImageSensitivityClassifier.classify_simulated()
    │           └── Per-image sensitivity scores
    │
    ├── [4] Privacy Risk Calculation
    │   ├── text_risk = calculate_privacy_risk(sensitivity, cloud)
    │   └── modality_risk = calculate_modality_privacy_risk(mm_sens, action)
    │
    ├── [5] Inference Execution
    │   │
    │   ├── action == 0 (Local Only)
    │   │   └── tmo_interface.get_local_inference(prompt)
    │   │       └── LoRAManager.select_adapter()
    │   │           └── Ollama (llama3.2:3b)
    │   │
    │   └── action > 0 (Cloud / Hybrid)
    │       ├── QueryDecomposer.decompose(prompt)
    │       │   └── Strategy: sentence / entity / clause / auto
    │       │
    │       ├── [Hybrid] if has_sensitive:
    │       │   ├── get_local_inference(local_query)
    │       │   ├── get_cloud_inference(cloud_query)
    │       │   └── ResponseAggregator.aggregate()
    │       │
    │       └── [Cloud Only] else:
    │           └── get_cloud_inference(prompt)
    │               └── Groq API (llama-3.3-70b)
    │
    ├── [6] Privacy Budget Update
    │   └── PrivacyBudget.consume(text_risk + modality_risk)
    │       └── Track cumulative risk vs ε
    │
    ├── [7] Reward Calculation
    │   │
    │   │   R = α·Quality
    │   │     + β₁·Association
    │   │     - β₂·Latency
    │   │     - β₃·Cost
    │   │     + β₄·SecurityScore
    │   │     - β₄·TextPrivacyRisk
    │   │     - β₅·ModalityPrivacyRisk    ← NEW
    │   │     + γ·BudgetBonus
    │   │
    │   └── subject to: Σₜ Risk(qₜ, aₜ) ≤ ε
    │
    └── [8] Next State Generation
        └── _augment_state(base, text_sens, mm_sens)
            └── [base_state, sens_level, sens_score, budget_ratio,
                 img0_sens, img1_sens, img2_sens]
```

### 3. RL Model Training

```
RC_PPO.train()
│
├── Sample from rollout_buffer
│
├── resource_constraint(observations)
│   ├── excess_latency = max(0, total_latency - budget)
│   ├── excess_usage = max(0, total_usage - budget)
│   └── excess_privacy = f(sensitivity, budget_ratio)    ← NEW
│
└── loss = policy_loss + value_loss + λ·constraint_penalty
```

---

## Multimodal Action Space

| Action | Description | Modalities to Cloud | Privacy Risk |
|:------:|-------------|---------------------|--------------|
| 0 | Local only | None | Zero |
| 1 | Cloud (text) | Text | text_risk |
| 2 | Cloud + img0 | Text + Image0 | text_risk + img0_risk |
| 3 | Cloud + img1 | Text + Image1 | text_risk + img1_risk |
| 4 | Cloud + img2 | Text + Image2 | text_risk + img2_risk |
| 5 | Cloud + img0,1 | Text + Image0 + Image1 | text_risk + img0_risk + img1_risk |
| 6 | Cloud + img0,2 | Text + Image0 + Image2 | text_risk + img0_risk + img2_risk |
| 7 | Cloud + img1,2 | Text + Image1 + Image2 | text_risk + img1_risk + img2_risk |
| 8 | Cloud + all | Text + All Images | text_risk + Σ img_risk |

---

## Installation

```bash
git clone https://github.com/your-repo/Privacy-TMO.git
cd Privacy-TMO
pip install -r requirements.txt
```

### Requirements

- Python >= 3.10
- PyTorch >= 2.2.0
- stable-baselines3 >= 2.2.1
- transformers >= 4.36.0
- peft >= 0.7.0
- opencv-python >= 4.8.0 (for image sensitivity)

---

## Quick Start

### Training with Privacy-Aware RL

```bash
python TMO/main/main.py \
  --privacy_budget 1.0 \
  --beta_security 0.3 \
  --beta_modality_privacy 0.2 \
  --simulate_image_sensitivity \
  --use_privacy_rl
```

### Sensitivity Analysis Example

```python
from privacy_tmo import PrivacyManager

pm = PrivacyManager(enable_ner=False, enable_ml=False)

# Text sensitivity
result = pm.analyze_query("My password is secret123")
print(f"Level: {result.level.name}, Score: {result.score:.2f}")
# Output: Level: PRIVATE, Score: 0.95

# Multimodal sensitivity
mm = pm.analyze_multimodal(
    text="Show me the kitchen",
    simulate_image_sensitivity=True,
    context={"task_cat": "Assistive System"}
)
print(f"Image sensitivities: {[mm.images[i].score for i in [0,1,2]]}")
```

---

## CLI Options

| Option | Default | Description |
|--------|---------|-------------|
| `--privacy_budget` | 1.0 | Privacy budget ε |
| `--beta_security` | 0.0 | Text privacy risk weight |
| `--beta_modality_privacy` | 0.2 | Image modality risk weight |
| `--use_image_sensitivity` | False | Enable real image analysis (OpenCV) |
| `--simulate_image_sensitivity` | False | Simulate image sensitivity |
| `--use_privacy_rl` | False | Use PrivacyConstrainedPPO |

---

## Project Structure

```
Privacy-TMO/
│
├── privacy_tmo/                        # Core privacy module
│   ├── config.py                       # Configuration management
│   ├── sensitivity_classifier.py      # Text sensitivity (Rule+NER+ML)
│   ├── image_sensitivity.py            # Image sensitivity (Face/OCR/Sim)
│   ├── privacy_manager.py              # Central privacy orchestrator
│   ├── query_decomposer.py             # Query decomposition strategies
│   ├── response_aggregator.py          # Hybrid response merging
│   ├── privacy_rl.py                   # PrivacyAwareEnv, ConstrainedPPO
│   ├── lora_trainer.py                 # On-device LoRA training
│   ├── privacy_attacks.py              # Canary, MIA simulations
│   ├── benchmark.py                    # Evaluation suite
│   └── profiler.py                     # Jetson performance profiling
│
├── tmo_interface.py                    # Ollama (local) + Groq (cloud)
├── lora_manager.py                     # Hierarchical LoRA adapters
├── requirements.txt
│
└── TMO/                                # Original TMO framework
    ├── dataset/M4A1.json               # M4AI multimodal dataset
    └── main/
        ├── main.py                     # Entry point
        ├── models.py                   # RC_PPO, RC_A2C, RC_DQN
        ├── utils.py                    # M4A1_Env with privacy integration
        └── options.py                  # CLI argument parser
```

---

## Technical Details

### Reward Function

**Original TMO:**
```
R = α·Quality + β₁·Association - β₂·Latency - β₃·Cost
```

**Privacy-TMO (Extended):**
```
R = α·Quality + β₁·Association - β₂·Latency - β₃·Cost
    + β₄·SecurityScore
    - β₄·TextPrivacyRisk
    - β₅·ModalityPrivacyRisk
    + γ·BudgetBonus

subject to: Σₜ PrivacyRisk(qₜ, aₜ) ≤ ε
```

### Sensitivity Levels

| Level | Score Range | Cloud Policy | Example |
|-------|-------------|--------------|---------|
| PUBLIC | 0.0 - 0.3 | Allowed | "What is Python?" |
| SEMI_SENSITIVE | 0.3 - 0.7 | Hybrid | "Schedule meeting with John" |
| PRIVATE | 0.7 - 1.0 | Local Only | "My password is secret123" |

---

## Evaluation

- **Privacy Attack Simulation**: Canary insertion, Membership inference attack
- **Baseline Comparison**: No Protection, Local Only, Threshold-based
- **Jetson Profiling**: Latency, memory, power consumption on edge devices

---

## Citation

```bibtex
@article{privacy-tmo-2025,
  title={Privacy-Preserving Personalized LLM Offloading for Edge-Cloud Collaboration},
  author={},
  year={2025}
}
```

---

## License

MIT License - see [LICENSE](./TMO/LICENSE) for details.
