# DRLND Navigation Project

## Install
1. Unzip the environment for your machine:
    Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Linux.zip)
    Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana.app.zip)
    Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Windows_x86.zip)
    Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Windows_x86_64.zip)
2. Create a virtual environment:
    - `virtualenv -p /usr/bin/python<version> <project>`
    - `conda create -n <project> python=<version>`
3. Install dependencies `$ pip install -r requirements.txt`

## Objective

![banana_env_gif](https://github.com/AmreshVenugopal/DRLND_Navigation_Project/blob/master/banana.gif?raw=true?raw=true "Banana environment")

To train an agent to collect yellow bananas while strictly avoiding blue bananas.


## Usage

### Simple Training Example
```
$ python main.py --env-path=/path/to/unzipped_banana_env
```

### Simple Test Example

This repo contains a `checkpoint.pth` which contains weights
that can be loaded right into the model like so:
```
$ python play.py --env-path=/path/to/unzipped_banana_env --model-path=/path/to/checkpoint.pth
```

### Flags
The agent training supports many flags which **if not provided** or
**are of incorrect type** the **defaults would be used**.

```
usage: main.py [-h] [--env-path ENV_PATH] [--model-path MODEL_PATH]
               [--episodes EPISODES] [--time-steps TIME_STEPS]
               [--qualify-score QUALIFY_SCORE] [--score-window SCORE_WINDOW]
               [--buffer-size BUFFER_SIZE] [--batch-size BATCH_SIZE]
               [--gamma GAMMA] [--lr LR] [--eps-start EPS_START]
               [--eps-decay EPS_DECAY] [--eps-end EPS_END] [--tau TAU]
               [--update-every UPDATE_EVERY] [--fc1_units FC1_UNITS]
               [--fc2_units FC2_UNITS] [--seed SEED]

Teach an agent to pick up yellow bananas from blue implementing Deep-Q
Learning

optional arguments:
  -h, --help            Show this help message and exit

  --env-path ENV_PATH   Path to the unity environment

  --model-path MODEL_PATH
                        Path to a trained q-network's checkpoint(.pth) file

  --episodes EPISODES   Number of episodes for which the agent must be trained

  --time-steps TIME_STEPS
                        Number of steps to be taken in an episode

  --qualify-score QUALIFY_SCORE
                        Score at which the training must stop

  --score-window SCORE_WINDOW
                        Number of episodes for which the qualify-score should
                        be maintained as average

  --buffer-size BUFFER_SIZE
                        Number of episodes to keep in memory (for experience
                        replay)

  --batch-size BATCH_SIZE
                        Number of samples/batch for training the Q network

  --gamma GAMMA         Discount factor of the rewards

  --lr LR               Learning rate

  --eps-start EPS_START
                        Initial epsilon for epsilon greedy

  --eps-decay EPS_DECAY
                        The value by which the initial epsilon must decay
                        over-time

  --eps-end EPS_END     The minimum value of epsilon beyond which there should
                        be no decay

  --tau TAU             The degree of influence the target network has on the main/local network

  --update-every UPDATE_EVERY
                        Number of time-steps in an episode after which the Q
                        network should be updated

  --fc1_units FC1_UNITS
                        Neurons in the first fully connected layer

  --fc2_units FC2_UNITS
                        Neurons in the second fully connected layer

  --seed SEED           Random seed to ensure same results

```

## Report
The experiment conducted has a detailed report [here](https://github.com/AmreshVenugopal/DRLND_Navigation_Project/blob/master/Report.md)