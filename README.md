# Self-Supervised Learning of Face Representations for Video Face Clustering (FG 2019)

[Vivek Sharma](http://vivoutlaw.github.io), 
[Makarand Tapaswi](http://www.cs.toronto.edu/~makarand/), 
[M. Saquib Sarfraz](https://sites.google.com/site/saquibsarfraz/), 
and [Rainer Stiefelhagen](https://cvhci.anthropomatik.kit.edu/people_596.php)

IEEE International Conference on Automatic Face and Gesture Recognition, FG 2019 

### Table of Contents
1. [Introduction](#introduction)
1. [Citation](#citation)
1. [Requirements and Dependencies](#requirements-and-dependencies)
1. [Installation](#installation)
1. [Training and Testing SSiam](#training-ssiam)

### Introduction
Analyzing the story behind TV series and movies often requires understanding who the characters are and what they are doing. With improving deep face models, this may seem like a solved problem. However, as face detectors get better, clustering/identification needs to be revisited to address increasing diversity in facial appearance. In this paper, we
address video face clustering using unsupervised methods. Our emphasis is on distilling the essential information, identity, from the representations obtained using deep pre-trained face networks. We propose a self-supervised Siamese network that can be trained without the need for video/track based supervision, and thus can also be applied to image collections. We
evaluate our proposed method on three video face clustering datasets. The experiments show that our methods outperform current state-of-the-art methods on all datasets. Video face clustering is lacking a common benchmark as current works are often evaluated with different metrics and/or different sets of face tracks
For more details and evaluation results, please check out our [paper](https://arxiv.org/pdf/1903.01000.pdf).

![SSiam Architecture](SSIAM.png)


### Citation

If you find the code and datasets useful in your research, please cite:
    
    @inproceedings{LapSRN,
        author    = {Sharma, Vivek and Tapaswi, Makarand and Sarfraz, M. Saquib and Stiefelhagen, Rainer}, 
        title     = {Self-Supervised Learning of Face Representations for Video Face Clustering}, 
        booktitle = {IEEE International Conference on Automatic Face and Gesture Recognition},
        year      = {2019}
    }

### Requirements and Dependencies
- MATLAB (we test with MATLAB R2018b on Ubuntu 16.04)
- Cuda & Cudnn (we test with Cuda 8.0 and Cudnn 5.1)

### Installation
Please install MatConvNet in your own path, you need to change the corresponding path `path2MatconNet` in `demo.m`.


### Training and Testing SSiam

To train and test SSiam from scratch, first download the training datasets:

    $ cd experiments/input_data
    $ wget http://cvhci.anthropomatik.kit.edu/~vsharma/bbt.tar.gz
    $ tar -xzf bbt.tar.gz
    $ cd ..


This script will train and test the SSiam method on `BBT-0101` dataset:

    >> demo()
    
Note that we only train/test our code on single-GPU mode. Running `demo.m` will reproduce the results in our paper for `BBT-0101`.
