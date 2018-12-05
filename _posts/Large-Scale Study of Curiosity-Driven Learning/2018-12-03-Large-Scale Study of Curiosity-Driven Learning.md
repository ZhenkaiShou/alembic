---
title: Large-Scale Study of Curiosity-Driven Learning
categories:
- Machine Learning
- Reinforcement Learning
---

In this blog I will talk about the [Large-Scale Study of Curiosity-Driven Learning](https://pathak22.github.io/large-scale-curiosity/resources/largeScaleCuriosity2018.pdf) paper developed by OpenAI, as well as giving some tips on how to reproduce it.

### What is curiosity?

First of all, we need to understand what curiosity means. Let's say, a baby may explore its surroundings without specific goals. It may open a drawer, or crawl under the bed ... of course aimlessly. A baby can be easily attracted by whatever looks new to it, until when the baby gets bored of it. Then what drives it to do such things? Yes, that's the power of curiosity! 

### What if we apply curiosity to the agent?

What will happen if we let the agent to explore the environments purely by curiosity? This is what that paper is all about. 

The extrinsic reward $ r_{ext} $, which can be sampled from the environment, is not used during training since the agent will only learn through its own curiosity. Instead, we need to define some intrinsic reward $ r_{int} $ which can reflect such kind of curiosity. Here we can define the intrinsic reward $ r_{int} $ as:
$$ r_{int} = ||f(\phi(o_{t}), a_{t}) - \phi(o_{t+1})||^{2} $$
where $ o_{t} $ is the observation at time step $ t $, $ a_{t} $ is the observation at time step $ t $, $ o_{t+1} $ is the next observation, $ \phi(\cdot) $ is a neural network that encodes the high dimensional observation into low dimensional feature, $ f(\cdot) $ is another neural network that predicts the next feature from the current feature and action.
