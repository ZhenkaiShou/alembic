---
title: Deep Learning Experiments on CIFAR-10 Dataset
categories:
- Deep Learning
---

In this blog I will share my experience of playing with [CIFAR-10](https://en.wikipedia.org/wiki/CIFAR-10) dataset using deep learning. I will show the impact of some deep learning techniques on the performance of a neural network.

<!-- more -->

## Contents
- [Motivation](#motivation)
- [Neural Network Architecture](#neural-network-architecture)

## Motivation
I have been studying deep learning / reinforcement learning for quite some time now. I have always been eager to know how each component can influence the performance of a neural network. However, I never get the chance to have a systematic study of this topic. That is why this time I decide to spend some time (and money) to run these experiments and write this blog.

## Neural Network Architecture
In the experiments I use the following network architecture:
- 1 [convolutional block](#conv-block),
- 4 [residual blocks](#res-block),
- global average pooling,
- a dense layer with 10 output neurons,
- a softmax operation to convert the logits to probability distribution.

<a name = "conv-block"></a>
The convolutional block contains:
- a 2D convolution,
- a dropout wrapper,
- batch normalization,
- ReLU operation.

<a name = "res-block"></a>
Each residual block contains:
- a 2D convolution,
- a dropout wrapper,
- batch normalization,
- ReLU operation,
- a 2D convolution,
- a dropout wrapper,
- batch normalization,
- skip connection,
- ReLU operation.

Below lists the output dimension of each layer:

|       Components       |  Output Dimensions |
|:----------------------:|:------------------:|
|       Input Image      |  (None, 32, 32, 3) |
|   Convolutional Block  | (None, 32, 32, 32) |
|    Residual Block 1    | (None, 32, 32, 32) |
|    Residual Block 2    | (None, 16, 16, 64) |
|    Residual Block 3    |  (None, 8, 8, 128) |
|    Residual Block 4    |  (None, 4, 4, 256) |
| Global Average Pooling |  (None, 1, 1, 256) |
|       Dense Layer      |     (None, 10)     |
|         Softmax        |     (None, 10)     |
