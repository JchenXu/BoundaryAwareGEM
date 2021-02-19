# Boundary-Aware Geometric Encoding for Semantic Segmentation of Point Clouds
by Jingyu Gong*, JiachenXu*, Xin Tan, Jie Zhou, Yanyun Qu, Yuan Xie and Lizhuang Ma. (*=equal contribution)

<p align="center"> <img src="imgs/framework.jpg" width="80%"> </p>

## Introduction
This project is based on our AAAI2021 paper. 

Boundary information plays a significant role in 2D image segmentation, while usually being ignored in 3D point cloud segmentation where ambiguous features might be generated in feature extraction, leading to misclassification in the transition area between two objects. In this paper, firstly, we propose a Boundary Prediction Module (BPM) to predict boundary points. Based on the predicted boundary, a boundary-aware Geometric Encoding Module (GEM) is designed to encode geometric information and aggregate features with discrimination in a neighborhood, so that the local features belonging to different categories will not be polluted by each other. To provide extra geometric information for boundary-aware GEM, we also propose a light-weight Geometric Convolution Operation (GCO), making the extracted features more distinguishing. Built upon the boundary-aware GEM, we build our network and test it on benchmarks like ScanNet v2, S3DIS. Results show our methods can significantly improve the baseline and achieve state-of-the-art performance.

## Installation
The code is based on [PointNet](https://github.com/charlesq34/pointnet)，[PointNet++](https://github.com/charlesq34/pointnet2), and [PointConv](https://github.com/DylanWusee/pointconv). Please install [TensorFlow](https://www.tensorflow.org/install/), and follow the instruction in [PointNet++](https://github.com/charlesq34/pointnet2) to compile the customized TF operators.  
The code has been tested with Python 3.6, TensorFlow 1.13.1, CUDA 10.0 and cuDNN 7.3 on Ubuntu 18.04.

## Usage
### ScanNet DataSet Segmentation

Download the ScanNetv2 dataset from [here](http://www.scan-net.org/), and see `scannet/README` for details of preprocessing.

To train a model to segment Scannet Scenes:

```
CUDA_VISIBLE_DEVICES=0 python train_scannet_IoU.py --model bagem_scannet --log_dir bagem_scannet_ --batch_size 8
```

After training, to evaluate the segmentation IoU accuracies:

```
CUDA_VISIBLE_DEVICES=0 python evaluate_scannet.py --model bagem_scannet --batch_size 8 --model_path bagem_scannet_%s --with_rgb 
```

Modify the model_path to your .ckpt file path.

###S3DIS DataSet Segmentation

Incoming :)

## License
This repository is released under MIT License (see LICENSE file for details).
