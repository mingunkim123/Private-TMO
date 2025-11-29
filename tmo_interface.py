import ollama
import time
import sys
import os
from transformers import pipeline, AutoTokenizer, AutoModelForTokenClassification
# [추가] 방금 만든 매니저 불러오기
from lora_manager import LoRAManager

# 1. 매니저 & 보안관 초기화
manager = LoRAManager()
print("🛡️ 보안관 모델 로딩 중... (처음엔 다운로드하느라 시간 좀 걸림)")

# [핵심] cache_dir을 /data로 지정해야 재부팅해도 모델이 안 날아감!
try:
    print("DEBUG: Starting model load...", flush=True)
    model_name = "dslim/bert-base-NER"
    cache_dir = '/data/models'
    
    print(f"DEBUG: Loading tokenizer for {model_name}...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
    print("DEBUG: Tokenizer loaded.", flush=True)

    print(f"DEBUG: Loading model for {model_name}...", flush=True)
    model = AutoModelForTokenClassification.from_pretrained(model_name, cache_dir=cache_dir)
    print("DEBUG: Model loaded.", flush=True)
    
    print("DEBUG: Creating pipeline...", flush=True)
    security_classifier = pipeline(
        "token-classification", 
        model=model, 
        tokenizer=tokenizer,
        device='cpu' # 젯슨 GPU 메모리 아끼려면 CPU 추천 (작은 모델이라 CPU도 빠름)
    )
    print("✅ 보안관 모델 로딩 완료!", flush=True)
except Exception as e:
    print(f"❌ 모델 로딩 실패: {e}", flush=True)
    raise

# 라이브러리 경로 강제 추가 (컨테이너 내부 경로 기준)
# 현재 파일 위치: /data/my_tmo_project/tmo_interface.py
# 라이브러리 위치: /data/my_tmo_project/libs
current_dir = os.path.dirname(os.path.abspath(__file__))
libs_path = os.path.join(current_dir, 'libs')
if libs_path not in sys.path:
    sys.path.append(libs_path)

try:
    from groq import Groq
except ImportError:
    print(f"❌ Groq import failed. Current sys.path: {sys.path}")
    print(f"Checking libs path: {libs_path}")
    if os.path.exists(libs_path):
        print(f"Libs contents: {os.listdir(libs_path)}")
    raise

# Groq 클라이언트 초기화 (API 키 필요)
# 주의: 실제 사용 시 API 키를 안전하게 관리하세요.
client = Groq(api_key="Key")


def get_local_inference(prompt):
    """
    로컬 젯슨에서 Ollama를 통해 추론을 수행하고 지연 시간을 측정합니다.
    매니저(LoRAManager)가 상황에 맞는 모델과 시스템 프롬프트를 선택합니다.
    """
    start_time = time.time()

    # ---------------------------------------------------------
    # [Step 1] 상황 분석 (Task 분류 - 임시로 키워드 기반)
    # ---------------------------------------------------------
    task_type = "general"
    if "code" in prompt or "python" in prompt: task_type = "coding"
    elif "pain" in prompt or "medicine" in prompt: task_type = "medical"
    elif "cook" in prompt or "recipe" in prompt: task_type = "cooking"

    # [Step 2] 민감도 분석 (보안관 호출)
    # (간단하게 키워드로만 체크하거나 BERT 결과 활용)
    is_sensitive = False
    if "password" in prompt or "address" in prompt or "secret" in prompt:
        is_sensitive = True

    # ---------------------------------------------------------
    # [Step 3] 매니저에게 "누가 나갈까?" 물어보기 (핵심!)
    # ---------------------------------------------------------
    model_name, sys_prompt, layer_name = manager.select_adapter(task_type, is_sensitive)
    
    print(f"🤖 [System] {layer_name} 계층 활성화 | 역할: {sys_prompt[:30]}...")

    # ---------------------------------------------------------
    # [Step 4] 실행 (선택된 페르소나 적용)
    # ---------------------------------------------------------
    try:
        response = ollama.chat(model=model_name, messages=[
            {'role': 'system', 'content': sys_prompt}, # <--- 여기가 바뀜!
            {'role': 'user', 'content': prompt}
        ])
        content = response['message']['content']
    except Exception as e:
        print(f"❌ Error: {e}")
        return "", 0.0

    end_time = time.time()
    latency = end_time - start_time
    
    return content, latency

def get_cloud_inference(prompt):
    """
    Groq API를 통해 클라우드 추론을 수행하고 지연 시간을 측정합니다.
    """
    # 재시도 로직 (최대 3번 시도)
    for attempt in range(3):
        try:
            start_time = time.time()
            completion = client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[{"role": "user", "content": prompt}]
            )
            end_time = time.time()
            latency = end_time - start_time
            
            # 성공하면 바로 리턴
            return completion.choices[0].message.content, latency

        except Exception as e:
            if "429" in str(e): # 429 에러 = "너 너무 많이 썼어!"
                print(f"⚠️ 무료 한도 초과! 10초 쉽니다... (시도 {attempt+1}/3)")
                time.sleep(10) # 10초 휴식
            else:
                print(f"❌ 알 수 없는 에러: {e}")
                return "", 0.0
    
    return "Cloud Error", 0.0

def get_security_score(prompt, action):
    # 1. 로컬(젯슨) 선택 시: 무조건 안전 (1.0점)
    if action == 0:
        return 1.0
    
    # 2. 클라우드 선택 시: 민감 정보 검사
    # BERT 모델이 문장 분석
    results = security_classifier(prompt)
    
    # 결과 필터링 (사람 이름, 조직, 위치 등 개인정보가 감지되면)
    sensitive_entities = [res['word'] for res in results if res['score'] > 0.9] # 확신 90% 이상만
    
    if len(sensitive_entities) > 0:
        print(f"🚨 [Security Alert] 민감 정보 감지됨: {sensitive_entities} -> 클라우드 전송 차단!")
        return 0.0 # 보안 위반! (점수 깎음)
    
    return 1.0 # 안전함
        
# --- 테스트 코드 (직접 실행 시에만 작동) ---
if __name__ == "__main__":
    test_prompt = "AI가 미래에 인류에게 미칠 영향은?"
    print(f"질문: {test_prompt}")
    print("⏳ 생각 중...")
    
    answer, lat = get_local_inference(test_prompt)
    
    print("-" * 30)
    print(f"💡 답변: {answer[:100]}...") # 너무 기니까 앞부분만 출력
    print("-" * 30)
    print(f"⏱️ 지연 시간(Latency): {lat:.4f}초")
    print("✅ TMO 시스템에 사용할 준비 완료!")
