---
title: Large-Scale Study of Curiosity-Driven Learning
categories:
- Machine Learning
- Reinforcement Learning
---

In this blog I will talk about the [Large-Scale Study of Curiosity-Driven Learning](https://pathak22.github.io/large-scale-curiosity/resources/largeScaleCuriosity2018.pdf) paper developed by OpenAI, as well as giving some [tips](#tips) on how to implement it.

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

$$ r_{t} = c_{r_{\text{ext}}} \cdot r_{\text{ext}, t} + c_{r_{\text{int}}} \cdot r_{\text{int}, t} \label{eq: r} $$

where $ c_{r_{\text{ext}}} = 0 $ and $ c_{r_{\text{int}}} = 1 $ are the coefficients. Later on we can set $ c_{r_{\text{ext}}} $ and $ c_{r_{\text{int}}} $ to other values for additional purposes.

### Feature Learning

There is debate on how to learn features in order to achieve good performance. Here are some possible choices:
- **Pixels**: We let $ \phi(o_{t}) = o_{t} $ so that the feature will be the same as the observation. 
- **Random Features**: Parameters in $ \phi(\cdot) $ is fixed and will not be changed during training. 
- **Inverse Dynamics Features (IDF)**: We use a network $ \hat{a}\_{t} = \text{idf}(\phi(o_{t}), \phi(o_{t+1})) $ to predict the action given both the current and next features. Parameters in $ \phi(\cdot) $ will be trained along with $ \text{idf}(\cdot) $ to minimize the action prediction loss.
- **Variational autoencoders (VAE)**: We use a decoder network $ \hat{o}\_{t} = \text{decode}(\text{sampled}(\phi(o_{t}))) $ to reconstruct the original observation. Parameters in $ \phi(\cdot) $ will be trained along with $ \text{decode}(\cdot) $ to minimize the VAE loss.

{% include figure.html image="https://zhenkaishou.github.io/my_site/assets/Large-Scale%20Study%20of%20Curiosity-Driven%20Learning/feature_learning.png" caption="Average reward across multiple environments with different feature learning methods. (Source: original paper)" width="90%" %}

Each feature learning method has its own pros and cons. The figure above compares different feature learning methods across multiple environments. It is difficult to tell which one is the best except for **Pixels** whose overall performance is bad.

### Training
OpenAI uses [Clipped PPO algorithm](https://blog.openai.com/openai-baselines-ppo/) to train the policy since it is a robost alogrithm which requires little hyperparameter tuning. For this algorithm to work, we need to create a policy network:

$$ \pi, v = \text{policy}(o) \label{eq: policy} $$

where $ \pi $ is the output policy, and $ v $ is the output value. The policy is trained by minimizing the following loss:

$$ L_{1} = L_{\text{pg}} + L_{\text{vf}} + c_{\text{ent}} \cdot L_{\text{ent}} $$

where $ L_{\text{pg}} $ is the policy gradient loss, $ L_{\text{vf}} $ is the value function loss, and $ L_{\text{ent}} $ is a regularization term (negative entropy) to prevent policy overfitting. For more details of PPO algorithm as well as the concrete expression of those loss functions, please refer to [PPO Algorithm](https://spinningup.openai.com/en/latest/algorithms/ppo.html) and [PPO Loss Functions](https://medium.com/aureliantactics/ppo-hyperparameters-and-ranges-6fc2d29bccbe).

That is how the policy network is trained. But we are not done yet! Still remember that we have the dynamic network $ f(\cdot) $ to generate the intrinsic reward? The dynamic network is trained by minimizing the following loss:

$$ L_{2} = L_{\text{aux}} + L_{\text{dyna}} $$

where
- $ L_{\text{aux}} $ is the auxiliary loss which depends on the chosen [feature learning method](#feature-learning),
  - In case of **Pixels** and **Random Features**, the auxiliary loss is set to 0.
- $ L_{\text{dyna}} = r_{\text{int}} $ is the dynamic loss with $ r_{\text{int}} $ defined in **Equation \ref{eq: r_int}**.

For more details regarding how to collect training data, please refer to [Rollout](#rollout).

### Large Scale = Better Performance

{% include figure.html image="https://zhenkaishou.github.io/my_site/assets/Large-Scale%20Study%20of%20Curiosity-Driven%20Learning/batch_size.png" caption="Average reward in Mario with different batch sizes of environment. (Source: original paper)" width="40%" %}

One interesting finding is that the performance improves as the batch size of environments goes up. The figure above compares different batch sizes of environment in Mario. A large batch size results in better performance. 

For more details regarding how to run multiple environments in parallel, please refer to [Parallel Environment](#parallel-environment).

### Curiosity with Extrinsic Reward

Sometimes we want an agent to learn skills for some particular task of interest. In that case, we can adjust the coefficient values in **Equation \ref{eq: r}**, let's say, we set $ c_{r_{\text{ext}}} = 1 $ and $ c_{r_{\text{int}}} = 0.01 $. This setting may come in handy especially when the extrinsic reward is **sparse**. For example, in navigation tasks, an agent needs to reach the target position in order to get a positive extrinsic reward (+1 reward for reaching the goal, 0 reward otherwise).

{% include figure.html image="https://zhenkaishou.github.io/my_site/assets/Large-Scale%20Study%20of%20Curiosity-Driven%20Learning/curiosity_with_extrinsic_reward.png" caption="Average reward in Unity maze with combined extrinsic and intrinsic reward. (Source: original paper)" width="40%" %}

The figure above shows the average extrinsic reward obtained by the agent in a Unity maze. Training with extrinsic reward only completely fails in this environment (the curve with "extrinsic only" label, which sits constantly at zero), while training with combined extrinsic and intrinsic reward enables the agent to reach the target position.

### Tips

In this section I will give some tips on the implementation. Feel free to skip this part if you are experienced in this field.

#### Parallel Environment

To run multiple environments in parallel, we can create a **ParallelEnvironment** class. Below is the main framework:
```python
import multiprocessing as mp

class ParallelEnvironment(object):
  def __init__(self, envs):
    self.envs = envs
    self.pipe_parents = []
    
    # Create a subprocess for each environment.
    for env in envs:
      pipe_parent, pipe_child = mp.Pipe()
      process = mp.Process(target = run_environment_process, args = (pipe_child, env))
      process.start()
      pipe_child.close()
      self.pipe_parents.append(pipe_parent)
  
  def reset(self):
    # Reset all environments.
    ...
  
  def step(self, action):
    # Take an action for each environment.
    ...
  
  def close(self):
    # Close all environments.
    ...
```

#### Rollout

#### Overall Network Architecture

#### Unmentioned Things
