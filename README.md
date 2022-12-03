# 2022-1-3
Face age prediction

## Description
-  baby:0, teenager:1, young:2, midlife:3, senior:4 로 label 되어 있는 얼굴 이미지들을 분류 하는 문제입니다.
-  Training data에는 gender (male, female), race (Asian, Black, Caucasian), direction (forward, rightward, leftward) 에 대한 label이 제공됩니다.
-  Testset은 각 age class가 균등하게 분배되어 있습니다.
-  Trainingset은 그렇지 않습니다.

- Testset에 대해서 높은 classification accuracy를 올릴 수 있는 모델을 만드는 task 입니다.

## Restriction
- Inference에 필요한 모델은 하나의 PyTorch model class로 만들어 주시기 바랍니다.
- Recognition (Classification) 모델은 scratch 부터 학습이 되어야 합니다.
- Detection, landmark detection 모델을 pre-processing 으로 사용하려는 경우에 해당 모델들은 pre-trained를 사용할 수 있습니다. 
- 입력 이미지 size는 장축 기준 224 이하로 설정해 주세요
- inference 모델의 파라미터 개수가 12M이 넘어가는 경우 저장이 되지 않습니다 (변경 금지).

## How to run:

Example training model
```bash
nsml run -v -d airush2022-1-3 -g 1 --memory 12G --shm-size 32G --cpus 10 -e main.py --esm yourESM
```

How to list checkpoints saved:

```bash
nsml model ls YOURSESSION
```

How to submit:

```bash
nsml submit -v YOURSESSION SAVED_IDX
```
