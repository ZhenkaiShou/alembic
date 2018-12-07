---
title: Large-Scale Study of Curiosity-Driven Learning
categories:
- Machine Learning
- Reinforcement Learning
---

In this blog I will talk about the [Large-Scale Study of Curiosity-Driven Learning](https://pathak22.github.io/large-scale-curiosity/resources/largeScaleCuriosity2018.pdf) paper developed by OpenAI, as well as giving some tips on how to reproduce it.

### What is Curiosity?

First of all, we need to understand what curiosity means. Let's say, a baby may explore its surroundings without specific goals. It may open a drawer, or even crawl under the bed ... aimlessly. A baby can be easily attracted by whatever looks new to it, until when the baby gets bored of it. Then what drives it to do such things? Yes, that's the power of curiosity! 

### Curiosity in Reinforcement Learning?

In standard reinforcement learning, an agent will receive extrinsic reward from the environment after taking an action. However, such extrinsic reward requires manual engineering and may not even exist in some scenarios. Furthermore, classic reinforcement learning algorithms may not work well when the rewards are sparse. 

What will happen if we do not use extrinsic reward at all and let the agent generate its own intrinsic reward (e.g. an agent learns through its own curiosity)? This is what [this paper](https://pathak22.github.io/large-scale-curiosity/resources/largeScaleCuriosity2018.pdf) is all about. 

To train an agent driven by curiosity, the extrinsic reward $ r_{\text{ext}} $ will not be used during training. Instead, we need to define some intrinsic reward $ r_{\text{int}} $ which can reflect such kind of curiosity. Here we can define the intrinsic reward $ r_{\text{int}} $ as:

$$ r_{\text{int}, t} = ||f(\phi(o_{t}), a_{t}) - \phi(o_{t+1})||^{2} \label{eq: r_int} $$

where
- $ o_{t} $ is the observation at time step $ t $,
- $ a_{t} $ is the observation at time step $ t $,
- $ o_{t+1} $ is the next observation,
- $ \phi(\cdot) $ is an encoding network that encodes high dimensional observation into low dimensional feature,
- $ f(\cdot) $ is a dynamic network that predicts the next feature given the current feature and action.

We can see that **Equation \ref{eq: r_int}** is actually the prediction error. An agent that is trained to maximize this reward $ r_{\text{int}} $ will prefer transitions with high prediction errors. Curiosity, in this case, can be intepreted as the inability to predict future states.

In general, the reward $ r_{t} $ can be defined as a mixture of extrinsic and intrinsic reward:

$$ r_{t} = c_{r_{\text{ext}}} * r_{\text{ext}, t} + c_{r_{\text{int}}} * r_{\text{int}, t} \label{eq: r} $$

where $ c_{r_{\text{ext}}} = 0 $ and $ c_{r_{\text{int}}} = 1 $ are the coefficients. Later on we can set $ c_{r_{\text{ext}}} $ and $ c_{r_{\text{int}}} $ to other values for additional purposes.

### Feature Learning

There is debate on how to learn features in order to achieve good performance. Here are some possible choices:
- **Pixels**: We let $ \phi(o_{t}) = o_{t} $ so that the feature will be the same as the observation. 
- **Random Features**: Parameters in $ \phi(\cdot) $ is fixed and will not be changed during training. 
- **Inverse Dynamics Features (IDF)**: We use a network $ \hat{a_{t}} = \text{idf}(\phi(o_{t}), \phi(o_{t+1})) $ to predict the action given both the current and next features. Parameters in $ \phi(\cdot) $ will be trained along with $ \text{idf}(\cdot) $ to minimize the action prediction loss.
- **Variational autoencoders (VAE)**: We use a decoder network $ \hat{o_{t}} = \text{decode}(\text{sampled}(\phi(o_{t}))) $ to reconstruct the original observation. Parameters in $ \phi(\cdot) $ will be trained along with $ \text{decode}(\cdot) $ to minimize the VAE loss.

{% include figure.html image="https://zhenkaishou.github.io/my_site/assets/Large-Scale%20Study%20of%20Curiosity-Driven%20Learning/feature_learning.png" caption="Performance of the agent across multiple environments with different feature learning methods. (Source: original paper)" width="90%" %}

Each feature learning method has its own pros and cons. The figure above compares different feature learning methods across multiple environments. It is difficult to tell which one is the best except for **Pixels** whose overall performance is bad.

### Training
OpenAI uses [Clipped PPO algorithm](https://blog.openai.com/openai-baselines-ppo/) to train the policy since it is a robost alogrithm which requires little hyperparameter tuning. For this algorithm to work, we need to create a policy network:

$$ \pi, v = \text{policy}(o) \label{eq: policy} $$

where $ \pi $ is the output policy, and $ v $ is the output value. The policy is trained by minimizing the following loss:

$$ \text{loss}_{1} = \text{loss}_{\text{pg}} + \text{loss}_{\text{vf}} + c_{\text{entropy}} * \text{loss}_{\text{entropy}} $$

where $ \text{loss}\_{\text{pg}} $ is the policy gradient loss, $ \text{loss}\_{\text{vf}} $ is the value function loss, and $ \text{loss}\_{\text{entropy}} $ is a regularization term to prevent policy overfitting. For more details of PPO algorithm as well as the concrete expression of those loss functions, please refer to [PPO Algorithm](https://spinningup.openai.com/en/latest/algorithms/ppo.html) and [PPO Loss Functions](https://medium.com/aureliantactics/ppo-hyperparameters-and-ranges-6fc2d29bccbe).

That is how the policy network is trained. But we are not done yet! Still remember that we have the dynamic network $ f(\cdot) $ to generate the intrinsic reward? The dynamic network is trained by minimizing the following loss:

$$ \text{loss}_{2} = \text{loss}_{\text{aux}} + \text{loss}_{\text{dyna}} $$

where
* $ \text{loss}\_{\text{aux}} $ is the auxiliary loss which depends on the chosen [feature learning method](#feature-learning),
  * In case of **Pixels** and **Random Features**, the auxiliary loss is set to 0.
* $ \text{loss}\_{\text{dyna}} = r_{\text{int}} $ is the dynamic loss with $ r_{\text{int}} $ defined in **Equation \ref{eq: r_int}**.

For more details regarding how to collect training data, please see this section.

### Large Scale = Better Performance

{% include figure.html image="https://zhenkaishou.github.io/my_site/assets/Large-Scale%20Study%20of%20Curiosity-Driven%20Learning/batch_size.png" caption="Performance of the agent in Mario with different batch sizes of environment. (Source: original paper)" width="50%" %}

An interesting finding is that the performance improves as the batch size of environments goes up. The figure above compares different batch sizes of environment in Mario. A large batch size results in better performance, at least in terms of extrinsic rewards. For more details regarding how to run multiple environments in parallel, please refer to this section.

### Curiosity with Extrinsic Reward
