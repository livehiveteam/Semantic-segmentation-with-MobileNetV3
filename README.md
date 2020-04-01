# Semantic segmentation with MobileNetV3

This repository contains the code for training of MobileNetV3 for segmentation as well as default model for classification. Every module here is subject for subsequent customizing.
## Content
*  [Requirements](#requirements)
*  [Quick setup and start](#quickstart)
*  [CNN architectures](#cnn)
*  [Loss functions](#loss_functions)
*  [Augmentations](#augmentations)
*  [Training](#training)
*  [Trained model](#trained_model)

## Requirements  <a name="requirements"/>
    Machine with an NVIDIA GPU
    NVIDIA driver >= 418
    CUDA >= 10.1
    Docker >= 19.03
    NVIDIA Container Toolkit (https://github.com/NVIDIA/nvidia-docker)
    
## Quick setup and start  <a name="quickstart"/>

### Docker 이미지 준비 
* Clone the repo, build a docker image using provided Makefile and Dockerfile. 

```
git clone 
make build
```

### 데이터셋 준비
clone한 repo에서 data.zip 파일을 아래 페이지에서 다운받고 압축을 푼다.
https://drive.google.com/open?id=1hvv_9Aj1s4jbIaWqxFeC00ss00EX_JQe
```
unzip data.zip
```

### 학습하기
Docker container 내에서 train.py를 실행한다.
```
make run
python train.py
```

### 테스트
Docker container 내에서 test.py를 실행하면 data 폴더 내에 inference 결과 이미지들이 저장된다. 
```
make run
python test.py
```

## CNN architectures <a name="cnn"/> 

MobileNetV3 backnone with Lite-RASSP modules were implemented.
Architecture may be found in [modules/keras_models.py](modules/keras_models.py)

## Loss functions  <a name="loss_functions"/>

F-beta and FbCombinedLoss (F-beta with Cross Entropy) losses were implemented.
Loss functions may be found in [modules/loss.py](modules/loss.py)

## Augmentations <a name="augmentations"/>

There were implemented the following augmentations:
Random rotation, random crop, scaling,
 horizontal flip, brightness, gamma and contrast augmentations,
  Gaussian blur and noise.
  
Details of every augmentation may be found in [modules/segm_transforms.py](modules/segm_transforms.py)
    
## Training  <a name="training"/>
 
 Training process is implemented in [notebooks/train_mobilenet.ipynb](notebooks/train_mobilenet.ipynb) notebook.
 
 Provided one has at least Pixart and Supervisely Person Dataset it is only needed to run every cell in the notebook subsequently.
 
## Trained model  <a name="trained_model"/>
 
 To successfully convert this version of MobileNetV3 model to TFLite optional argument "training" must be removed from every batchnorm layer in the model and after that pretrained weights may be loaded and notebook cells for automatic conversion may be executed.
