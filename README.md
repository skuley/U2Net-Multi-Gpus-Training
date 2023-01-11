# [U-2-Net](https://github.com/xuebinqin/U-2-Net) multi-gpu training


This is for researchers who are supported with multi-gpus!

** For this particular process you will need [pytorch lightning](https://www.pytorchlightning.ai)

## Required Libraries
Python 3.8<br>
Pytorch 1.12.0+cu102<br>
PyTorch Lightning 1.8.6<br>
Numpy 1.23.5<br>
Opencv-Python 4.6.0<br>
Albumentations 1.3.0<br>
wandb 0.13.8 (excluded in requirements)

## Install
1. Conda
```sh
conda create -n <env-name> python=3.8
```
2. pip
```sh
pip install -r requirements.txt
```

## Training

1. arguments

|short|long|eng_explanation|kor_explanation|default|type|
|-----|----|---------------|---------------|-------|----|
|-root|--data-root|data root directory|학습데이터경로|None|string|
|-data_json_path|--data_json_path|splitted data json path|train/val split된 txt파일|None|string|
|-bs|--batch_size|batch size|배치 사이즈|64|integer|
|-sp|--save_model_path|Path to save trained models|가중치 저장 경로|saved_models|string|
|-mn|--model_name|Model name|모델명|resnet50|string|
|-ne|--min_epochs|minimum epochs|최소 에포크|100|integer|
|-xe|--max_epochs|maximum epochs|최대 에포크|200|integer|
|-gn|--gpu_num|gpu index number|gpu 인덱스번호|1|integer|
|-so|--shuffle|shuffle option|데이터로더 순서 섞기|True|boolean|
|-nw|--num_workers|number of workers|데이터로더 연산코어 갯수|8|integer|
|-pf|--profiler|profiler type|프로파일러 타입|simple|string|
|-al|--accelerator|Supports passing different accelerator types (cpu,gpu,tpu,ipu,auto)|연산 하드웨어 종류|gpu|string|
