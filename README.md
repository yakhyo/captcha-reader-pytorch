## OCR: Captcha recognition using PyTorch

## Description
**Trained weight is exist in `weights` folder**

Implementation of RNN for Captcha recognition using PyTorch.

<div align='center'>
  <img src='assets/2b827.png' height="50px">
  <img src='assets/2bg48.png' height="50px">
  <img src='assets/2cegf.png' height="50px">
  <img src='assets/2cg58.png' height="50px">
</div>

## Prerequisites

- torch 1.9.0+cu111
- scikit-learn 0.24.2
- tqdm 4.62.3
- pillow 8.3.0

## Installation

```
git clone https://github.com/yakhyo/OCR-pt.git
curl https://github.com/yakhyo/Captcha-Reader-Keras/blob/master/captcha_images_v2.zip
```

**Data**

```
Data: https://github.com/yakhyo/Captcha-Reader-Keras/blob/master/captcha_images_v2.zip

├── OCR-pt
│   ├── assets
│   ├── nets
│   ├── utils
│   └── weights
└── captcha_images_v2
    ├── n8pfe.png
    ├── 2bg48.png
    └── .....
```

## Train
To train the model modify `train.py` and run the command:
```
python train.py
```

## Inference
To inference single image modify `recognize.py` and run the command:
```
python recognize.py
```
## Reference

1. https://github.com/abhishekkrthakur/captcha-recognition-pytorch

