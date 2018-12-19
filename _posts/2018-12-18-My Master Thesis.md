---
title: My Master Thesis
categories: 
- Misc
---

In this blog I will make a summary about my master thesis: **Learning to Plan in Large Domains with Deep Neural Networks**, in case someone is interested.

<!-- more -->

## Contents
- [Before We Start](#before-we-start)
- [Motivation](#motivation)
- [Revisit of AlphaGo Zero](#revisit-of-alphago-zero)
  - [Neural Network Architecture](#neural-network-architecture)

## Before We Start
Before we start, I would like to give you some insight on my master thesis. In short, my thesis is basically an extension of [AlphaGo Zero](https://arxiv.org/abs/1712.01815) inspired by [Imagination-Augmented Agents](https://arxiv.org/abs/1707.06203).

If you have already read the above two papers, you will find my thesis quite easy to follow. If not, well, I cannot guarantee anything. In the following sections I assume you have a general understanding of how AlphaGo Zero works, e.g.:
- A neural network with a policy head and a value head,
- Monte-Carlo tree search (MCTS) to find the best action,
- How neural network estimation is combined with MCTS in AlphaGo Zero,
- Selfplay to sample training data,
- Loss function in AlphaGo Zero and its optimization.

## Motivation
In the domain of artifcial intelligence, effective and efficient planning is a key factor to developing an adaptive agent which can solve tasks in complex environments. However, traditional planning algorithms only work properly in small domains:
- Global planners (e.g. value iteration), which can in theory give an accurate esimation of all states, suffers from the curse of dimensionality.
- Local planners (e.g. MCTS), which gives a coarse estimation of the current state, becomes less effective in large domains.

On the other hand, learning to plan, where an agent applies the knowledge learned from the past experience to planning, can scale up planning effectiveness in large domains.

Recent advances in deep learning widen the access to better learning techniques. Combining traditional planning algorithms with modern learning techniques in a proper way enables an agent to extract useful knowledge and thus show good performance in large domains. For instance, AlphaGo Zero achieved its success by combining a deep neural network with MCTS.

However, in the above example, the neural network does not fully leverage the local planner:
- Only the final output of the local planner (e.g. the searching probability distribution) is relevant to the agent's training.
- Some other valuable information of the local planner (e.g. the trajectory with the most visit counts) is nowhere to be used. 

Hereby we raise the following question: is it possible to design a neural network that can fully leverage the local planner to further improve the effectiveness and efficiency of planning?

## Revisit of AlphaGo Zero
Let's first revist some key parts of AlphaGo Zero.

###### Neural Network Architecture
{% include figure.html image="https://zhenkaishou.github.io/my-site/assets/My%20Master%20Thesis%20/Neural_Network_1.png" caption="Neural Network Architecture of AlphaGo Zero." width="60%" %}

The neural network in AlphaGo Zero can be formed as:

$$ V, P = f(s) $$

where $ s $ is the input state, $ V $ is output value, and $ P $ is the output policy. The figure above provides a more detailed description. The state $ s $ is first encoded into some feature $ x $, and then the network is split into two heads: the value head to predict value $ V $ and the policy head to predict the policy $ P $.
