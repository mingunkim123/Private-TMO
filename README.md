# 🛡️ Jetson-Secure-TMO
## Privacy-Preserving Offloading for Personalized LLMs

본 프로젝트는 **TMO (Task / Model Offloading)** 프레임워크를 기반으로 하며,  
**Jetson Orin Nano (8GB)** 환경에서 **TensorRT-LLM**을 지원하도록  
추론 엔진을 최적화하고,  
**계층적 개인화(Hierarchical Personalization)** 및  
**개인정보 보호(Privacy Guard)** 기능을 통합한  
프라이버시 중심 LLM 오프로딩 시스템입니다.

---

## 🚀 Key Features

### 🔹 Edge–Cloud Collaboration
데이터 소유권(Data Ownership)과 포터빌리티(Data Portability) 원칙을  
엣지 엔드포인트(Jetson)에 직접 구현합니다.  
지연 시간(Latency)과 비용(Cost)을 고려하여  
로컬(Jetson)과 클라우드 간 추론 작업을 동적으로 스케줄링합니다.

---

### 🔹 Hierarchical LoRA Selection
사용자의 작업 성격과 민감도에 따라  
Personal / Group / General 계층의  
LoRA 어댑터를 실시간으로 교체합니다.

이는 민감도 기반 데이터 노출을 제한하는  
계층적 사용자 프로필 설계 원칙을 따릅니다.

---

### 🔹 Privacy-First Guard
BERT 기반 NER 모델을 활용하여  
질문 내 개인 식별 정보(PII)를 탐지합니다.

보안 점수(Security Score)에 따라  
클라우드 전송 여부를 결정하며,  
하드웨어 수준 보호와 결합된 아키텍처 제어를 통해  
불필요한 데이터 노출을 차단합니다.

---

### 🔹 Hardware Optimization
Jetson Orin Nano의 제한된 메모리(8GB) 환경에서도  
원활한 구동을 위해 다음을 적용했습니다:

- INT4 양자화
- TensorRT-LLM 가속
- 하드웨어 기반 보안 메커니즘 통합

이를 통해  
**신뢰 가능한 실행 환경(TEE 지향 구조)** 의 기반을 마련합니다.

---

## 🏗️ System Architecture

본 시스템은 단순한 모델 실행을 넘어,  
다음과 같은 **학술적 메커니즘을 통합**하여 설계되었습니다.

---

### 1️⃣ Context-Aware Policy
질문의 복잡도(Complexity)와  
지연 시간·비용 예산(Budget Constraints)을 분석하여  
**최적의 추론 경로 및 모델을 선택**합니다.

> 참고: *Efficient Contextual LLM Cascades*

---

### 2️⃣ Personalization–Generalization Split
- **민감한 개인 정보** → 로컬 LoRA 어댑터
- **일반적인 지식 질의** → 클라우드 LLM

데이터 격리(Data Isolation) 및  
보안 집계(Secure Aggregation)를 통해  
**프라이버시를 유지하면서 정보 비대칭을 해소**합니다.

---

### 3️⃣ Cost–Performance Optimization
**FrugalGPT 전략**을 차용하여  
응답 품질(Quality)을 유지하면서도  
클라우드 API 호출 비용을 최소화합니다.

---

## 📚 References & Acknowledgments

본 프로젝트는 다음 연구들을 기반으로 설계 및 구현되었습니다.

- **TMO Framework**  
  Task / Model Offloading Framework for Edge–Cloud LLM Inference

- **MoA-OFF**  
  Adaptive Heterogeneous Modality-Aware Offloading with Edge–Cloud Collaboration

- **PerLLM**  
  Personalized Inference Scheduling with Edge–Cloud Collaboration

- **FrugalGPT**  
  How to Use Large Language Models While Reducing Cost and Improving Performance

- **Federated Split Learning**  
  Joint Personalization–Generalization for Inference-Stage Optimization

- **Privacy-Preserving Personalization**  
  Hierarchical User Profiling Methods for Privacy Protection

---

