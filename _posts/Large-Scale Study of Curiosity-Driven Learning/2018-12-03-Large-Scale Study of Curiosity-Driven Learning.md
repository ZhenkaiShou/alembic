---
title: Large-Scale Study of Curiosity-Driven Learning
categories:
- Machine Learning
- Reinforcement Learning
---

In this blog I will talk about the [Large-Scale Study of Curiosity-Driven Learning](https://pathak22.github.io/large-scale-curiosity/resources/largeScaleCuriosity2018.pdf) paper developed by OpenAI, as well as giving some tips on how to reproduce it.

### What is curiosity?

First of all, we need to understand what curiosity means. Let's say, a baby may explore its surroundings without specific goals. It may open a drawer, or even crawl under the bed ... aimlessly. A baby can be easily attracted by whatever looks new to it, until when the baby gets bored of it. Then what drives it to do such things? Yes, that's the power of curiosity! 

### What if we incorporate curiosity into reinforcement learning?

In standard reinforcement learning, an agent will receive extrinsic reward from the environment after taking an action. However, such extrinsic reward requires manual engineering and may not even exist in some scenarios. Furthermore, classic reinforcement learning algorithms may not work well when the rewards are sparse. 

What will happen if we do not use extrinsic reward at all and let the agent generate its own intrinsic reward (e.g. an agent learns through its own curiosity)? This is what [this paper](https://pathak22.github.io/large-scale-curiosity/resources/largeScaleCuriosity2018.pdf) is all about. 

To train an agent driven by curiosity, the extrinsic reward $ r_{ext} $ will not be used during training. Instead, we need to define some intrinsic reward $ r_{int} $ which can reflect such kind of curiosity. Here we can define the intrinsic reward $ r_{int} $ as:

$$ r_{int, t} = ||f(\phi(o_{t}), a_{t}) - \phi(o_{t+1})||^{2} \label{eq: r_int} $$

where
- $ o_{t} $ is the observation at time step $ t $,
- $ a_{t} $ is the observation at time step $ t $,
- $ o_{t+1} $ is the next observation,
- $ \phi(\cdot) $ is a neural network that encodes high dimensional observation into low dimensional feature,
- $ f(\cdot) $ is also a neural network that predicts the next feature given the current feature and action.

We can see that **Equation \ref{eq: r_int}** is actually the prediction error. An agent that is trained to maximize this reward $ r_{int} $ will prefer transitions with high prediction errors. Curiosity, in this case, can be intepreted as the inability to predict future states.

In general, the reward $ r_{t} $ can be defined as a mixture of extrinsic and intrinsic reward:

$$ r_{t} = c_{r_{ext}} * r_{ext, t} + c_{r_{int}} * r_{int, t} \label{eq: r} $$

where $ c_{r_{ext}} = 0 $ and $ c_{r_{int}} = 1 $ are the coefficients. Later on we can set $ c_{r_{ext}} $ and $ c_{r_{int}} $ to other values for additional purposes.

### Feature learning

There is debate on how to learn features $ \phi $ in order to achieve good performance. Here are some possible choices:
- **Pixels**: We let $ \phi(o_{t}) = o_{t} $ so that the feature will be the same as the observation. 
- **Random Features**: Parameters in $ \phi(\cdot) $ is fixed and will not be changed during training. 
- **Inverse Dynamics Features (IDF)**: We use a network $ \hat{a_{t}} = \text{idf}(\phi(o_{t}), \phi(o_{t+1})) $ to predict the action given both the current and next features. Parameters in $ \phi(\cdot) $ will be trained along with $ \text{idf}(\cdot) $ to minimize the action prediction loss.
- **Variational autoencoders (VAE)**: We use a decoder network $ \hat{o_{t}} = \text{decode}(\text{sampled}(\phi(o_{t}))) $ to reconstruct the original observation. Parameters in $ \phi(\cdot) $ will be trained along with $ \text{decode}(\cdot) $ to minimize the VAE loss.

Each feature learning method has its own pros and cons. The figure below compares different feature learning methods across multiple environments. Note again that the extrinsic reward is only used for measuring the performance, not for training. It is difficult to tell which feature learning method is the best except for **Pixels** whose overall performance is bad.

{% include figure.html image="https://zhenkaishou.github.io/my_site/assets/Large-Scale%20Study%20of%20Curiosity-Driven%20Learning/feature_learning.png" caption="Performance of different feature learning methods across multiple environments. (Source: original paper)" width="90%" %}

### Training algorithm
[Clipped PPO algorithm](https://blog.openai.com/openai-baselines-ppo/) is applied to the policy training since it is a robost alogrithm which requires little hyperparameter tuning. For this algorithm to work, we need to create a policy network:

$$ \pi, v = \text{policy}(o) \label{eq: policy} $$

where $ \pi $ is the output policy, and $ v $ is the output value. The policy is trained by minimizing the following loss:

$$ loss_{policy} = loss_{pg} + loss_{vf} + c_{entropy} * loss_{entropy} $$

where $ loss_{pg} $ is the policy gradient loss, $ loss_{vf} $ is the value function loss, and $ loss_{entropy} $ is a regularization term to prevent policy overfitting. For more details on PPO algorithm as well as the concrete expression of loss functions, please refer to [PPO Algorithm](https://spinningup.openai.com/en/latest/algorithms/ppo.html) and [PPO Loss Functions](https://medium.com/aureliantactics/ppo-hyperparameters-and-ranges-6fc2d29bccbe).
