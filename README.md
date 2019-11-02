# Hemodialysis
~~MLP 데이터 생성 및 MLP 모델 학습   
현재 regression 과 classification 까지 학습 가능   
Classification 은 학습까지만 완성 (메트릭, 로깅 등 아직 미완성)~~  
__10.19 Update__  
* SBP DBP 동시 학습 
* Logging, snapshot  
* Classification 은 그래프 그리지 않고 sensitivity 와 specificity class 별로 로깅   


__11.02 Updata__
* 35-class 학습하는 code 작성. (option으로 주지 않고 주석 처리해서 써야됨. 아직 training 원인을 알지 못해 code수정 계획X)
* confusion matrix
* MLP model : deep하게 & batchnorm추가
