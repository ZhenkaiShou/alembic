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
- [Experiments](#experiments)
  - [Network Type](#network-type)

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

|        **Layer**       |**Output Dimension**|
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

<br>

## Experiments
Unless otherwise mentioned, all the experiments are conducted under the following settings:
```python
epoch = 120                        # Number of epochs
batch_size = 100                   # Minibatch size

optimizer = "Adam"                 # Available optimizer, choose between ("Momentum" | "Adam")
learning_rate = [1e-3, 1e-4, 1e-5] # Learning rate for each phase
lr_schedule = [60, 90]             # Total number of epochs required until reaching next learning rate phase

normalize_data = False             # Whether input images are normalized
flip_data = False                  # Whether input images might be flipped
crop_data = False                  # Whether input images are zero-padded and then randomly cropped

network_type = "Res4"              # Network type, choose between ("Res4" | "Conv8" | "None")
dropout_rate = 0.2                 # Dropout rate, value of 0 means no dropout
c_l2 = 0.0                         # L2 regularization, also known as weight decay
batch_norm = True                  # Whether batch normalization is applied
global_average_pool = True         # Whether global average pooling is applied before dense layer
```
