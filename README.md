# Hemodialysis
* 명령어 : ~your_PATH/Project_Hemo/Hemodialysis_200813$ sh train_rnn_v4.sh 
* sh 파일 열어보면, 동일한 명령어가 4번 반복되어 있음 --> 하나 끝나면 다음꺼 자동 실행

* 결과 : result/result.csv 에 저장됨

* gpu : defalut 1 --> 두개 중, 두번째 gpu

* 매 eval마다 confusion matrix 저장하면 너무 오래 걸림 --> args.draw_confusion_matrix에서 조정. (default=False)

* snapshot
  * 매 epoch마다 저장
  * model만 저장하고, optim은 저장 안함 --> 따라서 저장한 것을 불러오면 성능이 더 낮아질 수도 있음
  * 30 epoch 다 저장하면 약 4GB 이니, 필요하다면 저장을 끄거나, best만 저장하도록.


* 현재 feature : fix는 MLP, seq는 RNN 방식이 아님. 그냥 전체를 RNN 에.
  * fix_size==85 --> one-hot들
  * seq_size==45 --> numeric
  * 따로 구분한 이유 : input batchnorm을 numeric만 하는게 좋을 수도 있다고 판단 되어서. (실험은 안 해봄)
  
  
* weight_loss_ratio
  * 0 이 아니면 실행
  * positive를 더 잘 찾기 위해 적용해봤으나 효과있다는 것을 발견 못 했음.
  * calibration graph가 망가지는 것은 봤었음.
  * 효과가 없는 것인지... 적절한 지점을 못 찾은 것인지....

* topk_loss_ratio
  * 1.0 이 아니면 실행
  * minibatch 안의 전체 timestamp 에서, loss 순으로 sorting 후, 큰 loss만 backward 하겠다는 컨셉
  * e.g) 0.5 면, loss 크기 상위 50% 만 학습
  
*  안해본 것
  * 5 outcome이 아닌, 1 outcome RNN 을 5개 만들면 성능이 어떻게 되는지?
  * 한번의 투석에서 모든 frame의 loss 를 계산하는 것이 아닌, 일부만 loss 계산 (run_rnn_v4_0813.py --line215,217,218)
  * 이전처럼 fixed --> MLP  /  seq --> RNN
  * gradient cliping을 해왔는데, 끄면 어떻게 되는지?
