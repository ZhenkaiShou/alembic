---
title: Code Collection
categories: 
- Misc
---

This post serves as a code collection of my personal implementations for different algorithms, which cover general deep learning algorithms and some paper algorithms.

<!-- more -->

## Content
- [General Algorithms](#general-algorithms)
  - [Deep Deterministic Policy Gradient](#deep-deterministic-policy-gradient)
  - [Mixture Density Networks](#mixture-density-networks)
  - [Neural Machine Translation](#neural-machine-translation)
  - [Residual Networks](#residual-networks)
  - [Variational Autoencoder](#variational-autoencoder)
- [Paper Algorithms](#paper-algorithms)
  - [Deep Q-Network](#deep-q-network)
  - [Large-Scale Study of Curiosity-Driven Learning](#large-scale-study-of-curiosity-driven-learning)
  - [Prioritized Experience Replay](#prioritized-experience-replay)
  - [World Models](#world-models)

## General Algorithms
This category contains stand alone implementation of general machine learning algorithms.

###### Deep Deterministic Policy Gradient
- Reinforcement learning algorithm.
- Policy optimization algorithm that works well in continous action space.
- Repository: [DDPG](https://github.com/ZhenkaiShou/project/tree/master/stand%20alone%20implementation/DDPG)

###### Mixture Density Networks
- Neural network architecture.
- Network model that learns a mixture of probability distributions of the input variables.
- Repository: [MDN](https://github.com/ZhenkaiShou/project/tree/master/stand%20alone%20implementation/MDN)

###### Neural Machine Translation
- Neural network architecture.
- Sequence to sequence model that transforms an input sequence into an output sequence.
- Repository: [NMT](https://github.com/ZhenkaiShou/project/tree/master/stand%20alone%20implementation/NMT)

###### Residual Networks
- Neural network architecture.
- Network model that uses residual connection to resolve vanishing gradient problem of deep neural networks.
- Repository: [ResNet](https://github.com/ZhenkaiShou/project/tree/master/stand%20alone%20implementation/ResNet)

###### Variational Autoencoder
- Neural network architecture.
- Generative model that learns useful features for image reconstruction while restricting the features close to normal distribution.
- Repository: [VAE](https://github.com/ZhenkaiShou/project/tree/master/stand%20alone%20implementation/VAE)

## Paper Algorithms
This category contains implementation of specific algorithms proposed in some papers.

###### Deep Q-Network
- Reinforcement learning algorithm.
- The first algorithm that achieves human-level performance on Atari games with raw-pixel input.
- Repository: [Deep Q-Network](https://github.com/ZhenkaiShou/project/tree/master/paper%20reproduction/DQN)

###### Large-Scale Study of Curiosity-Driven Learning
- Reinforcement learning algorithm.
- Explore what will happen if an agent is trained purely by its own curiosity.
- Repository: [Large-Scale Study of Curiosity-Driven Learning](https://github.com/ZhenkaiShou/project/tree/master/paper%20reproduction/Large-Scale%20Study%20of%20Curiosity-Driven%20Learning)

###### Prioritized Experience Replay
- Reinforcement learning algorithm.
- An extension to Deep Q-Network (DQN). Samples with higher training error is more likely to be sampled for training.
- Repository: [PER](https://github.com/ZhenkaiShou/project/tree/master/paper%20reproduction/PER)

###### World Models
- Reinforcement learning algorithm.
- Build an agent inspired by our own cognitive system: a vision model, a memory model, and a controller model.
- Repository: [World Models](https://github.com/ZhenkaiShou/project/tree/master/paper%20reproduction/World%20Models)
