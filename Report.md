# Learning Algorithm

## Objective
~[banana_env_gif](banana.gif?raw=true "Banana environment")
To train an agent to collect yellow bananas while strictly avoiding blue bananas.

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
    This helps re-using the experiences as the agent gets trained, also prevents reinforcement of same actions
    which may happen if there is little variation in the next state for a given action.

# Plot of Rewards
~[banana_agent_rewards](DRLND_agent_scores.png?raw=true "Agent scores")


# Ideas for future work
- Running the models with the `play.py` script shows that there are occurances where the agent
frantically moves back and forth in an attempt to prevent collision with the blue bananas.
This can be solved by increasing the memory buffer or allowing for adaptive epsilon.

