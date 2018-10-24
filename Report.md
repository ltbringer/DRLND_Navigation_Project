# Learning Algorithm

## Objective
![banana_env_gif](https://github.com/AmreshVenugopal/DRLND_Navigation_Project/blob/master/trained_banana_collector.gif?raw=true?raw=true "Banana environment solved")

A reward of +1 is provided for collecting a yellow banana, and a reward of -1 is provided for collecting a blue banana. Thus, the goal of your agent is to collect as many yellow bananas as possible while avoiding blue bananas.

A reward of +1 is provided for collecting a yellow banana, and a reward of -1 is provided for collecting a blue banana. Thus, the goal of your agent is to collect as many yellow bananas as possible while avoiding blue bananas.

The state space has 37 dimensions and contains the agent's velocity, along with ray-based perception of objects around the agent's forward direction. Given this information, the agent has to learn how to best select actions. Four discrete actions are available, corresponding to:

- 0 - move forward.
- 1 - move backward.
- 2 - turn left.
- 3 - turn right.

The task is episodic, and in order to solve the environment, your agent must get an average score of +13 over 100 consecutive episodes.


## Implementation
- A `Deep Q-learning` algorithm was implemented to perform the given task using `pytorch`.
- The `Q-network` architecture is as follows:
    - Three fully connected layers with reLU activation function, where `state_size` is the number of states
    in the environment, here `37`. `action_size` is the number of actions possible for the agent; here `4`.
    `fc1_units` and `fc2_units` are both taken as `64` in this project.
    ```
        ('fc1', nn.Linear(state_size, fc1_units))
        ('reLU', nn.ReLU())
        ('fc2', nn.Linear(fc1_units, fc2_units))
        ('reLU', nn.ReLU())
        ('fc3', nn.Linear(fc2_units, action_size))
    ```

    The Q-network is tasked with the objective to find the optimal policy π* for the agent.
    ```
    Q(s,a) = Q(s, a) + α[R(s, a) + γmaxQ'(s',a')-Q(s,a)]
    ```
- Experience Replay: is implemented by providing the agent with a memory
    to hold `buffer_size` number of samples from which a random sample can be used for learning.
    This helps re-using the experiences as the agent gets trained, also prevents reinforcement of a
    certain kind/kinds of actions due to consecutive tuples containing same/similar experience.

- Fixed target implementation
    ∆w = α[(R + γmaxQ(s',a,w)) - Q(s,a,w)]∇Q(s,a,w)
    The target and the Q value are using the same approximation function 
    (weights of the network) hence, updating the weights in the q_network and 
    estimating the error between itself and the target_network would be inaccurate 
    as both keep diverging away, unless one of them stays relatively stable it would 
    be difficult to reduce the loss or ∆w, also known as the moving target problem, as 
    the target and the parameters to be changed are coupled.

    Here, to counter the moving target problem, small updates are made to the weights of the 
    target network by introducing a hyperparameter tau. 
    
    Imagine tau to be a very small number, then the effect of multiplying tau with the 
    q_network's weights will reduce the degree of change that will be brought upon in the 
    target_network's weights, also (1 - tau) will be a larger number, retaining majority of the
    target networks previous weights.
    
    This causes the target network to update much slower than the q_network solving 
    the moving target problem.

## Hyperparameters
- **BUFFER_SIZE   = 10000**     : The number of samples to be saved in the memory buffer
- **BATCH_SIZE    = 64**        : Training the Q network with samples in a batch of 64
- **GAMMA         = 0.99**      : Discount factor while considering future rewards
- **TAU           = 1e-3**      : Impact of target network over the main network
- **LEARNING_RATE = 5e-4**      : The learning rate
- **EPSILON_START = 1.0**       : For epsilon greedy
- **EPSILON_DECAY = 0.995**     : For epsilon greedy
- **EPSILON_END   = 0.01**      : For epsilon greedy
- **UPDATE_EVERY  = 4**         : The frequency of time-step in which the Q-network must be updated



# Plot of Rewards
![banana_agent_rewards](https://github.com/AmreshVenugopal/DRLND_Navigation_Project/blob/master/DRLND_agent_scores.png?raw=true?raw=true "Agent scores")


# Ideas for future work
- Running the models with the `play.py` script shows that there are occurances where the agent
frantically moves back and forth in an attempt to prevent collision with the blue bananas.
This can be solved by increasing the memory buffer or allowing for adaptive epsilon.

![banana_agent_rewards](https://github.com/AmreshVenugopal/DRLND_Navigation_Project/blob/master/banana_agent_fails.gif?raw=true?raw=true "Agent scores")

[Back to Readme](https://github.com/AmreshVenugopal/DRLND_Navigation_Project/blob/master/Readme.md)