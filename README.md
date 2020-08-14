# Hemodialysis
~~MLP 데이터 생성 및 MLP 모델 학습   
현재 regression 과 classification 까지 학습 가능   
Classification 은 학습까지만 완성 (메트릭, 로깅 등 아직 미완성)~~  
__10.19 Update__  
* SBP DBP 동시 학습 
* Logging, snapshot  
* Classification 은 그래프 그리지 않고 sensitivity 와 specificity class 별로 로깅   


__~11.15 Update__

* 3가지 기준에 대해 각각 binary classification
  * 기준1 : sbp가 1시간 이내 'sbp 초기값 대비' 20 이상 떨어졌는가? --> 떨어졌으면 1 / 아니면 0
  * 기준2 : map가 1시간 이내 'map 초기값 대비' 10 이상 떨어졌는가? --> 떨어졌으면 1 / 아니면 0
  * 기준3 : sbp가 1시간 이내 90 이하로 떨어졌는가? --> 떨어졌으면 1 / 아니면 0
 
* utils
  * ROC curve plot & TRP, FPR 이용한 auroc 계산
  * interpolation

* data loader - interpolation 반영

* GRU 모델 몇개 추가 

* 새 기준에 대한 eval 방식 추가
  * interpolation되지 않은 부분만 eval하도록
  * 정상인 frame에서만 prediction잘하는지 eval (--> 비정상인 경우는 rule base로 이미 alarm이 갔을 것이기 때문에?)
