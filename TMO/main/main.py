import json
import os
from stable_baselines3 import PPO, A2C, DQN
import pickle
from models import RC_PPO, RC_A2C, RC_DQN

from options import args_parser
from utils import *

import warnings
# 불필요한 경고 메시지 무시 설정
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=RuntimeWarning)


if __name__ == '__main__':
    # 커맨드 라인 인자 파싱 (설정값 로드)
    args = args_parser()

    # 데이터셋 로드 (M4A1.json 파일)
    with open('../dataset/M4A1.json', 'r') as f:
        dataset = json.load(f)

    # 환경 설정 변수 초기화
    # weights: [알파(가중치), 연관성 베타, 지연시간 베타, 사용량 베타, 보안 베타]
    weights = [args.alpha, args.beta_association, args.beta_latency, args.beta_usage, args.beta_security]
    latency_budget = args.latency_budget  # 지연시간 예산 (제약조건)
    usage_budget = args.usage_budget      # 사용량 예산 (제약조건)
    privacy_budget = args.privacy_budget  # 프라이버시 예산 (제약조건)
    local_device = args.local_device      # 로컬 디바이스 스펙
    cloud_server = args.cloud_server      # 클라우드 서버 스펙

    time_span = args.time_span  # 시뮬레이션 시간 범위
    repeat = args.repeat        # 실험 반복 횟수

    # 실험할 자원 제약 조건 설정 (False: 제약 없음, True: 제약 있음)
    Resource_Constraints = [False, True]
    
    # 비교할 모델들 정의
    # Random/Local/Cloud: 베이스라인 모델 (학습 없음)
    # PPO/A2C/DQN: 일반 강화학습 모델
    # RC_*: 자원 제약(Resource Constrained)을 고려한 강화학습 모델
    models = {'Random': None, 'Local': None, 'Cloud': None, 
            'PPO': PPO, 'A2C': A2C, 'DQN': DQN,
            'RC_PPO': RC_PPO, 'RC_A2C': RC_A2C, 'RC_DQN': RC_DQN}
    if args.use_privacy_rl:
        from privacy_tmo import PrivacyAwareEnv, PrivacyConstrainedPPO
        models['PRC_PPO'] = PrivacyConstrainedPPO
    
    results = {} # 결과 저장용 딕셔너리
    Train = {}   # 학습 관련 데이터 저장용

    # 실험 반복 루프
    for _ in range(repeat):
        # 데이터셋을 학습용과 테스트용으로 분할 (8:2 비율)
        train_dataset, test_dataset = split_dataset(dataset, test_ratio=0.2)
        
        # 자원 제약 조건별로 실험 수행
        for resource_constraint in Resource_Constraints:
            # 학습 환경 생성
            train_env = M4A1_Env(train_dataset, weights, local_device, cloud_server, latency_budget, usage_budget, privacy_budget, resource_constraint, time_span, Train=True)
            if args.use_privacy_rl:
                train_env = PrivacyAwareEnv(train_env)
            
            # 학습 환경의 보상 및 신경망 정보 저장 (테스트 환경에서 참조용)
            Train['rewards'] = train_env.rewards; Train['nn'] = train_env.nn
            
            # 테스트 환경 생성
            test_env = M4A1_Env(test_dataset, weights, local_device, cloud_server, latency_budget, usage_budget, privacy_budget, resource_constraint, time_span, Train)
            if args.use_privacy_rl:
                test_env = PrivacyAwareEnv(test_env)
            
            # 각 모델별로 학습 및 평가 수행
            for model_name, model_cls in models.items():
                model_key = (resource_constraint, model_name, model_cls)
                
                # 해당 모델을 현재 제약 조건에서 실행해야 하는지 확인
                if should_process_model(model_name, resource_constraint):
                    # 모델 처리 (학습 및 평가 후 results에 결과 저장)
                    process_model(model_key, model_cls, train_env, test_env, latency_budget, usage_budget, results)


    # 결과 저장 디렉토리 생성
    if not os.path.exists('results'):
        os.makedirs('results')

    # 최종 결과를 pickle 파일로 저장
    with open('results/Main_Results.pkl', 'wb') as f:
        pickle.dump(results, f)

