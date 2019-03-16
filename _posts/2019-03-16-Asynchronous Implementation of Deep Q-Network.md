---
title: Asynchronous Implementation of Deep Q-Network
categories:
- Reinforcement Learning
---

In this blog I will share my personal experience of implementing some basic deep learning algorithm asynchronously, including some difficulties I have encountered.

<!-- more -->

## Contents
- [Deep Q-Network](#deep-q-network)
- [Asynchronous DQN](#asynchronous-dqn)

## Deep Q-Network
[Deep Q-Network](https://deepmind.com/research/dqn/) (DQN) is a basic reinforcement learning algorithm that is able to play Atari games with visual input. Its training pipeline is shown below:
- Initialize network variables;
- Initialize the environment;
- Initialize the replay buffer;
- Loop until reaching maximum steps: 
  - Sample an action using exploration policy;
  - Simulate the environemnt with the sampled action;
  - Store data into the replay buffer;
  - Sample training data from the replay buffer;
  - Train the network for one mini-batch.

## Asynchronous DQN
Sometimes we want to accelerate the training progress via asynchronous training. The basic idea is to create multiple processes so that the network is shared and trained simultaneously by different processes. The training pipeline of asynchronous DQN is shown below:
- Initialize the global network variables;
- Create N processes;
- For each process do the following **simultaneously**: 
  - Initialize a local environment;
  - Initialize a local replay buffer;
  - Loop until reaching maximum global steps: 
    - Get a copy of the latest global network;
    - Sample an action using exploration policy;
    - Simulate the local environment with the sampled action;
    - Store data into the local replay buffer;
    - Sample training data from the local replay buffer;
    - Compute the gradients of the copied network based on the training data;
    - Apply the gradients to the global network.

In TensorFlow, asynchronous training can be achieved by [Distributed TensorFlow](https://github.com/tensorflow/examples/blob/master/community/en/docs/deploy/distributed.md).
