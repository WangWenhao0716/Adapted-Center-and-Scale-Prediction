ATTENTION: I will release my codes about the paper [**Adapted Center and Scale Prediction: More stable and More Accurate**](<https://arxiv.org/abs/2002.09053>). This is the official pytorch implementation. However, because my paper is based on [**High-level Semantic Feature Detection: A New Perspective for Pedestrian Detection**](<https://arxiv.org/abs/1904.02948>) and my ability is limited, I choose to use and change the codes from [**CSP PyTorch Implementation**](<https://github.com/lw396285v/CSP-pedestrian-detection-in-pytorch>). Many thanks for his contribution！

#I will release my codes(adaptations) as soon as possible! However, due to heavy schooling, it may be delayed.
#Welcome to my [**personal website**](<https://wenhaowang.org>)

# ACSP PyTorch Implementation
![image](https://github.com/WangWenhao0716/pictures/blob/master/4.png)

## NOTE
Please do not run the codes until I finish uploading. The followings are constructing!

## Requirement
* Python 3.6
* Pytorch 0.4.1.post2
* Numpy 1.16.0
* OpenCV 3.4.2
* Torchvision 0.2.0

## Reproduction 
* Test our models: One GPU with about/over 4G memory.
* Train new models: Two GPUs with 32G memory per GPU.

## Installation
You can directly get the codes by:
```
  git clone https://github.com/WangWenhao0716/Adapted-Center-and-Scale-Prediction.git
```

## Preparation
1. CityPersons Dataset

2. Pretrained Models
The backbone of our ACSP is modified ResNet-101, i.e. replacing all BN layers with SN layers. You can download from [here](https://pan.baidu.com/s/1rK-ukAjEIPql2ECi38hRbQ). The weight is stored in `./models`.
3. Our Trained Models

## Training

## Test


## Citation
If you think our work is useful in your research, please consider citing:
```
@article{wang2020adapted,
  title={Adapted Center and Scale Prediction: More Stable and More Accurate},
  author={Wang, Wenhao},
  journal={arXiv preprint arXiv:2002.09053},
  year={2020}
}
```
