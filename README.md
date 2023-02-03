# [U-2-Net](https://github.com/xuebinqin/U-2-Net) multi-gpu training


This is for researchers who are supported with multi-gpus!

** For this particular process you will need [pytorch lightning](https://www.pytorchlightning.ai)

## UPDATES!
- Pretrained weights (DUTS-TR):<br>
checkpoints : [u2net.ckpt](https://drive.google.com/file/d/1m4_POtSvbcH0zjpq0yPM6dd6yf71VZzQ/view?usp=sharing)<br> state_dict: [u2net.pth](https://drive.google.com/file/d/1soMzlTkKH2pl6-ZkmZRwA18OnxAzFIEK/view?usp=sharing)
- I have updated ```inference.py``` code to test u2net<br>

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

|keyword|type|
|-------|----|
|--min_epoch|int|
|--max_epoch|int|
|--batch_size|int|
|--lr|float|
|--epsilon|float|
|--tr_im_path|string|
|--tr_gt_path|string|
|--vd_im_path|string|
|--vd_gt_path|string|
|--pretrained_path|string|
|--save_weight_path|string|

2. Script<br>
** I made it easier to change arguments by changing in python script 

```sh
python train_u2net.py
```


