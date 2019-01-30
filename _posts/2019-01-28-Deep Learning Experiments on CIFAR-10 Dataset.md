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
  - [Regularizations](#regularizations)
  - [Batch Normalization](#batch-normalization)
  - [Global Average Pooling](#global-average-pooling)
  - [Data Normalization](#data-normalization)
  - [Data Augmentation](#data-augmentation)
  - [Optimizer](#optimizer)
- [Conclusion](#conclusion)

## Motivation
I have been studying deep learning and reinforcement learning for quite some time now. I have always been eager to know how each component can influence the performance of a neural network. However, I never get the chance to have a systematic study of this topic. That is why this time I decide to spend some time (and money) to run these experiments and write this blog.

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
Unless otherwise mentioned, all experiments use the default settings below:
```python
epoch = 120                        # Number of epochs
batch_size = 100                   # Minibatch size

optimizer = "Adam"                 # Available optimizer, choose between ("Momentum" | "Adam")
learning_rate = [1e-3, 1e-4, 1e-5] # Learning rate for each phase
lr_schedule = [60, 90]             # Epochs required to reach the next learning rate phase

normalize_data = False             # Whether input images are normalized
flip_data = False                  # Whether input images are flipped with 50% chance
crop_data = False                  # Whether input images are zero-padded and randomly cropped

network_type = "Res4"              # Network type, choose between ("Res4" | "Conv8" | "None")
dropout_rate = 0.2                 # Dropout rate, value of 0 means no dropout
c_l2 = 0.0                         # L2 regularization, also known as weight decay
batch_norm = True                  # Whether batch normalization is applied
global_average_pool = True         # Whether global average pooling is applied
```

[**Fig. 1**](#fig-1) shows the performance of the network with default settings. **Bold lines** represent the test loss (and error), while thin lines represent the training loss (and error). For convenience, the default network is denoted as `res4` and later on it will be comapred to other variants.

<a name="fig-1"></a>
{% include figure.html image="https://zhenkaishou.github.io/my-site/assets/Deep%20Learning%20Experiments%20on%20CIFAR-10%20Dataset/Res4.png" caption="<b>Fig. 1:</b> Performance of the network with default settings." width="100%" %}

###### Network Type
To compare different network structures, I trained the following variants:
- `conv8`: the 4 residual blocks are replaced by 8 convolutional blocks
  - `network_type = "Conv8"`
- `simple network`: the 4 residual blocks are removed from the graph
  - `network_type = "None"`

From [**Fig. 2**](#fig-2) we can see that
- Both `res4` and `conv8` have similar performance; 
- Both `res4` and `conv8` outperform `simple network` by a large margin. 

This result implies that
- Convolutional network is on par with its residual counterpart when the network is relatively shallow;
- Residual network only shines when the network goes much deeper.

<a name="fig-2"></a>
{% include figure.html image="https://zhenkaishou.github.io/my-site/assets/Deep%20Learning%20Experiments%20on%20CIFAR-10%20Dataset/Network%20Type%20Comparison.png" caption="<b>Fig. 2:</b> Comparison of different network types." width="100%" %}

###### Regularizations
To compare different regularization methods, I trained the following variants:
- `res4, no dropout`: remove dropout
  - `dropout_rate = 0.0`
- `res4, L2`: add L2 regularization to the total loss (dropout still applies)
  - `c_l2 = 1e-4`
- `res4, L2, no dropout`: add L2 regularization to the total loss, and remove dropout
  - `dropout_rate = 0.0, c_l2 = 1e-4`

From [**Fig. 3**](#fig-3) we can see that
- Dropout increases training training loss (and error) but reduces test loss (and error);
- L2 regularization can reduce the test loss by a large margin, but has limited impact on the test error.

This result implies that
- Dropout is in general a good regularization method since it adds robustness to the network to reduce overfitting;
- L2 regularization can only reduce overfitting to some extent since it directly manipulates the loss function. 

<a name="fig-3"></a>
{% include figure.html image="https://zhenkaishou.github.io/my-site/assets/Deep%20Learning%20Experiments%20on%20CIFAR-10%20Dataset/Regularization%20Comparison.png" caption="<b>Fig. 3:</b> Comparison of different regularization methods." width="100%" %}

###### Batch Normalization
To see the impact of batch normalization, I trained the following variant:
- `res4, no batch norm`: remove batch normalization
  - `batch_norm = False`

From [**Fig. 4**](#fig-4) we can see that
- Batch normalization reduces both training and test loss (and error) by a large margin.

This result implies that
- Batch normalization can improve the overall performance of a network since it makes learning of each layer more independently.

<a name="fig-4"></a>
{% include figure.html image="https://zhenkaishou.github.io/my-site/assets/Deep%20Learning%20Experiments%20on%20CIFAR-10%20Dataset/Batch%20Norm%20Comparison.png" caption="<b>Fig. 4:</b> Impact of batch normalization." width="100%" %}

###### Global Average Pooling
To see the impact of gloal average pooling, I trained the following variant:
- `res4, no global pool`: remove global average pooling
  - `global_average_pool = False`

From [**Fig. 5**](#fig-5) we can see that
- Global average pooling reduces test loss (and error) by a large margin.

This result implies that
- Global average pooling can reduce overfitting by greatly reducing the number of parameters in the next dense layer.

<a name="fig-5"></a>
{% include figure.html image="https://zhenkaishou.github.io/my-site/assets/Deep%20Learning%20Experiments%20on%20CIFAR-10%20Dataset/Global%20Pool%20Comparison.png" caption="<b>Fig. 5:</b> Impact of global average pooling." width="100%" %}

###### Data Normalization
To see the impact of data normalization, I trained the following variant:
- `res4, normalize data`: normalize the image data by substracting the per-pixel mean
  - `normalize_data = True`

From [**Fig. 6**](#fig-6) we can see that
- Data normalization has almost no impact on the performance.

This result implies that
- Data normalization does not seem to be helpful, which is probably because the output of each layer has already been normalized by batch normalization.

<a name="fig-6"></a>
{% include figure.html image="https://zhenkaishou.github.io/my-site/assets/Deep%20Learning%20Experiments%20on%20CIFAR-10%20Dataset/Normalize%20Data%20Comparison.png" caption="<b>Fig. 6:</b> Impact of data normalization." width="100%" %}

###### Data Augmentation
To see the impact of data augmentation, I trained the following variants:
- `res4, flip data`: image data are horizontally flipped with 50% chance
  - `flip_data = True`
- `res4, crop data`: image data are first padded with zeros on each side and then randomly cropped
  - `crop_data = True`
- `res4, augment data`: a combination of the above two variants
  - `flip_data = True, crop_data = True`

From [**Fig. 7**](#fig-7) we can see that
- Both flipping data and cropping data increase training loss (and error), but reduce test loss (and error) by a large margin;
- A combination of both provides even better performance.

This result implies that
- Data augmentation can significantly reduce overfitting since more data can be generated from a fixed dataset.

<a name="fig-7"></a>
{% include figure.html image="https://zhenkaishou.github.io/my-site/assets/Deep%20Learning%20Experiments%20on%20CIFAR-10%20Dataset/Augment%20Data%20Comparison.png" caption="<b>Fig. 7:</b> Impact of data augmentation." width="100%" %}

###### Optimizer
To compare different optimizers, I trained the following variant:
- `res4, momentum optimizer`: use stochastic gradient descent with momentum to minimize the loss
  - `optimizer = "Momentum", learning_rate = [1e-1, 1e-2, 1e-3]`

From [**Fig. 8**](#fig-8) we can see that
- Adam optimizer achieves slightly lower test loss (and error).

This result implies that
- The choice of optimizer may influence the final performance to some degree;
- There is no universal answer to which optimizer is better.

<a name="fig-8"></a>
{% include figure.html image="https://zhenkaishou.github.io/my-site/assets/Deep%20Learning%20Experiments%20on%20CIFAR-10%20Dataset/Momentum%20vs%20Adam%20Optimizer.png" caption="<b>Fig. 8:</b> Comparison of different optimizers." width="100%" %}

## Conclusion
In this blog I showed you how different techniques can influence the performance of a neural network. Below I will sort those techniques according to their impact on the performance:
- [Deeper Network](#network-type)
  - ★★★★★
  - Enable learning of high-level features
  - Overall improvement
- [Dropout](#regularization) 
  - ★★★★✰
  - Add robustness
  - Reduce overfitting
- [Data Augmentation](#data-augmentation)
  - ★★★★✰
  - Generate more training data
  - Reduce overfitting
- [Batch Normalization](#batch-normalization)
  - ★★★★✰
  - Make learning of each layer independently 
  - Overall improvement
- [Global Average Pooling](#global-average-pooling)
  - ★★★✰✰
  - Reduce network parameters
  - Reduce overfitting
- [L2 Regularization](#regularization)
  - ★★✰✰✰
  - Add penalty loss
  - Reduce overfitting
- [Data Normalization](#data-normalization)
  - ★✰✰✰✰
  - Redistribute input data
  - Almost no impact (when batch normalization has been applied)
- [Optimizer](#optimizer)
  - ★✰✰✰✰
  - Different optimization methods
  - Almost no impact
