import pickle
import os

result_path = 'results/Main_Results.pkl'

if os.path.exists(result_path):
    with open(result_path, 'rb') as f:
        data = pickle.load(f)
    
    print("✅ 결과 파일 로드 성공!")
    print(f"데이터 타입: {type(data)}")
    print(f"데이터 내용 (일부): {data}")
    # 만약 data가 리스트라면 길이 출력
    if isinstance(data, list):
        print(f"총 에피소드 수: {len(data)}")
else:
    print("❌ 결과 파일이 없습니다! 경로를 확인하세요.")