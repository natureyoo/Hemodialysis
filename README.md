# Hemodialysis
MLP 데이터 생성 및 MLP 모델 학습
현재 regression 과 classification 까지 학습 가능
Classification 은 학습까지만 완성 (메트릭, 로깅 등 아직 미완성)

## Directory layout 

    .
    ├── run.py                       # 학습 실행
    ├── csv_parser.py                # tensor 데이터 생성 
    ├── loader.py                    # Dataloader
    ├── models.py                    # 모델 (MLP)
    ├── sampler.py                   # imbalanced sampler 
    ├── utils.py
    ├── Visualization.ipynb          # 시각화용 쥬피터 노트북 
    ├── tensor_data                  # tensor data (run.py 실행 시 구조 참고; 데이터 파일은 구글드라이브)  
    |   ├── MLP
    |       ├── Train.pt
    |       ├── Test.pt
    |       ├── Validation.pt
    └── README.md  
      
## Usage 
run.py 에서 아래 명령어로 학습 시작 
```python
#type = 'sbp' or 'dbp'
mlp_clas(type)  
mlp_regression(type)
```
