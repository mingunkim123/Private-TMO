🛡️ Jetson-Secure-TMO: Privacy-Preserving Offloading for Personalized LLMs

이 프로젝트는 TMO (Task/Model Offloading) 프레임워크를 기반으로 하며, Jetson Orin Nano (8GB) 환경에서 TensorRT-LLM을 지원하도록 추론 엔진을 최적화하고 계층적 개인화(Hierarchical Personalization) 및 개인정보 보호(Privacy Guard) 기능을 추가한 시스템입니다.
🚀 Key Features

    Edge-Cloud Collaboration: 지연 시간(Latency)과 비용(Cost)을 고려하여 로컬(Jetson)과 클라우드 간의 추론 작업을 동적으로 스케줄링합니다.

Hierarchical LoRA Selection: 사용자의 작업 성격과 민감도에 따라 Personal / Group / General 계층의 LoRA 어댑터를 실시간으로 교체합니다.

Privacy-First Guard: BERT 기반 NER 모델을 활용하여 질문 내 개인 식별 정보(PII)를 탐지하고, 보안 점수에 따라 클라우드 전송 여부를 결정합니다.

Hardware Optimization: 젯슨 오린 나노의 제한된 메모리(8GB) 내에서 원활한 구동을 위해 INT4 양자화 및 TensorRT-LLM 가속을 적용했습니다.

🏗️ System Architecture

본 시스템은 단순한 모델 실행을 넘어, 다음의 학술적 메커니즘을 통합하여 구현되었습니다:

    Context-Aware Policy: 질문의 복잡성과 예산 제약을 분석하여 최적의 모델을 선택합니다 (참고: Efficient Contextual LLM Cascades ).

Personalization-Generalization Split: 민감한 개인 정보는 로컬 어댑터에서 처리하고, 일반적인 지식은 클라우드 모델을 활용하여 정보 비대칭을 해결합니다.

Cost-Performance Optimization: FrugalGPT의 전략을 차용하여 응답 품질을 유지하면서도 클라우드 API 호출 비용을 최소화합니다 (참고: FrugalGPT ).

📚 References & Acknowledgments

본 프로젝트는 다음의 연구들을 참고하여 개발되었습니다:

    TMO Framework: [Original Repository Link]

    MoA-OFF: Adaptive Heterogeneous Modality-Aware Offloading with Edge-Cloud Collaboration.

PerLLM: Personalized Inference Scheduling with Edge-Cloud Collaboration.

FrugalGPT: How to Use Large Language Models While Reducing Cost and Improving Performance.

Federated Split Learning: Joint Personalization-Generalization for Inference-Stage Optimization.

Privacy-Preserving Personalization: Hierarchical User Profiling Methods for Privacy Protection.
