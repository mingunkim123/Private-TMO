class LoRAManager:
    def __init__(self):
        # === 1. 모델 레지스트리 (나중에 실제 학습된 어댑터 이름으로 교체) ===
        self.base_model = "llama3.2:3b"
        
        # === 2. 가짜 LoRA (시스템 프롬프트로 흉내내기) ===
        # 실제 LoRA가 준비되면 이 부분을 모델 이름 매핑으로 바꾸면 됩니다.
        self.personas = {
            "general": "You are a helpful AI assistant.",
            
            # [Group Layer] 직군별 전문가
            "coding": "You are an expert Python programmer. Answer with code blocks.",
            "medical": "You are a knowledgeable medical consultant. Provide safe advice.",
            "cooking": "You are a professional chef. Provide delicious recipes.",
            
            # [Personal Layer] 나만의 비서
            "personal": "You are Mingun's personal secretary. You know his schedule and preferences."
        }

    def select_adapter(self, task_type, is_sensitive=False):
        """
        입력: 작업 종류, 민감 여부
        출력: (모델이름, 시스템프롬프트, 선택된_계층)
        """
        
        # 1. Personal Layer (최우선 - 민감 정보)
        if is_sensitive:
            print(f"🔍 [Manager] 민감 정보 감지 -> 'Personal LoRA' 선택")
            return self.base_model, self.personas["personal"], "Personal"

        # 2. Group Layer (도메인 전문 지식)
        if task_type in self.personas:
            print(f"🔍 [Manager] 전문 작업({task_type}) 감지 -> 'Group LoRA' 선택")
            return self.base_model, self.personas[task_type], "Group"

        # 3. General Layer (기본)
        print(f"🔍 [Manager] 일반 질문 -> 'Base Model' 선택")
        return self.base_model, self.personas["general"], "General"
