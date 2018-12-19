---
title: My Master Thesis
categories: 
- Misc
---

In this blog I will make a brief summary about my master thesis: **Learning to Plan in Large Domains with Deep Neural Networks**, in case someone is interested.

<!-- more -->

## Contents
- [Before We Start](#before-we-start)
- [Motivation](#motivation)

## Before We Start
Before we start, I would like to give you some insight on my master thesis. In short, my thesis is basically an extension of [AlphaZero](https://arxiv.org/abs/1712.01815) inspired by [Imagination-Augmented Agents](https://arxiv.org/abs/1707.06203).

If you have already read the above two papers, you will find my thesis quite easy to follow. If not, well, I cannot guarantee anything. In the following I assume you have a rough understanding of how Alpha(Go) Zero works, e.g:
- A neural network with a policy head and a value head,
- Monte-Carlo tree search (MCTS) to find the best action,
- How neural network estimation is combined with MCTS in Alpha(Go) Zero,
- Selfplay to sample training data,
- Loss function in Alpha(Go) Zero and its optimization.

## Motivation
In the domain of artifcial intelligence, effective and efficient planning is a key factor to developing an adaptive agent which can solve tasks in complex environments. However, traditional planning algorithms only work properly in small domains:
- Global planners (e.g. value iteration), which can in theory give an accurate esimation of all states, suffers from the curse of dimensionality.
- Local planners (e.g. MCTS), which gives a coarse estimation of the current state, becomes less effective in large domains.

On the other hand, learning to plan, where an agent applies the knowledge learned from the past experience to planning, can scale up planning effectiveness in large domains.

Recent advances in deep learning widen the access to better learning techniques. Combining traditional planning algorithms with modern learning techniques in a proper way enables an agent to extract useful knowledge and thus show good performance in large domains. For instance, Alpha(Go)Zero achieved its success by combining a deep neural network with MCTS.

However, in the above example, the neural network does not fully leverage the local planner:
- Only
since some valuable information of the local planner (e.g. the principal variation in MCTS) is not informed to the network. Hereby we raise the following question: can we design a neural network that can fully leverage the local planner to further improve the effectiveness and efficiency of planning?
