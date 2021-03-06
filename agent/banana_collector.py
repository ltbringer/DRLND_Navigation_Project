import numpy as np
import random
import torch
import torch.nn.functional as F
import torch.optim as optim
from q_network.q_network import build_q_net
from q_network.memory import ReplayBuffer
from utils.cli import defaults


class Agent():
    """
    The agent must perform one of these actions:
        1. 0 - move forward.
        2. 1 - move backward.
        3. 2 - turn left.
        4. 3 - turn right.

    The agent must take into account the current state and identify the most valuable action
    that can be performed.

    This agent will implement Deep-Q learning to assess the optimal policy for the navigation project
    """
    def __init__(
            self,
            state_size,
            action_size,
            seed,
            fc1_units=defaults.FC1_UNITS,
            fc2_units=defaults.FC2_UNITS,
            buffer_size=defaults.BUFFER_SIZE,
            batch_size=defaults.BATCH_SIZE,
            gamma=defaults.GAMMA,
            tau=defaults.TAU,
            lr=defaults.LEARNING_RATE,
            update_every=defaults.UPDATE_EVERY
    ):
        """
        :param state_size: state space or dimensions for a given state
        :param action_size: actions space or number of actions that are possible
        :param seed: To be able to obtain same results
        -----------------
        hyper-parameters
        -----------------
        :param buffer_size: Number of episodes to keep in memory for experience replay
        :param batch_size: Number of samples for training the Q network
        :param gamma: Discount factor
        :param lr: learning rate
        :param update_every: number of iterations after which an update is required
        """
        self.state_size = state_size
        self.action_size = action_size

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.q_network = build_q_net(state_size, action_size, fc1_units=fc1_units, fc2_units=fc1_units)\
            .to(self.device)

        self.target_network = build_q_net(state_size, action_size, fc1_units=fc1_units, fc2_units=fc2_units).\
            to(self.device)

        self.optimizer = optim.Adam(self.q_network.parameters(), lr=lr)
        self.memory = ReplayBuffer(action_size, buffer_size, batch_size, seed)
        self.t_step = 0
        self.fc1_units = fc1_units
        self.fc2_units = fc2_units
        self.update_every = update_every
        self.batch_size = batch_size
        self.buffer_size = buffer_size
        self.gamma = gamma
        self.tau = tau

    def load_saved_model(self, model_path):
        self.q_network = build_q_net(
            self.state_size,
            self.action_size,
            fc1_units=self.fc1_units,
            fc2_units=self.fc2_units
        ).to(self.device)

        model_state_dict = torch.load(model_path)
        self.q_network.load_state_dict(model_state_dict)

    def step(self, state, action, reward, next_state, done):
        """
        A step taken by the agent is modeled in this method
        Since, the implementation also includes experience replay,
        the arguments are all added to the agents memory, to be
        learned from at a later stage. Here, when there are enough
        experiences (equal to the batch_size hyper-parameter)

        :param state: list
        current state

        :param action: int
        the action performed

        :param reward: float
        the reward corresponding to the action in the current state

        :param next_state: list
        the resultant state of the environment after the action

        :param done: bool
        is the task complete?

        :return: None
        """
        self.memory.add(state, action, reward, next_state, done)
        self.t_step = (self.t_step + 1) % self.update_every
        if self.t_step == 0:
            if len(self.memory) > self.batch_size:
                experiences = self.memory.sample()
                self.learn(experiences, self.gamma)

    def act(self, state, eps=defaults.EPSILON_END):
        """
        Epsilon greedy implementation, for a given state
        should the action be a random or the best known
        :param state: list
        :param eps: float
        :return: int
        """
        state = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
        self.q_network.eval()
        with torch.no_grad():
            action_values = self.q_network(state)
        self.q_network.train()
        if random.random() > eps:
            return np.argmax(action_values.cpu().data.numpy())
        else:
            return np.random.choice(np.arange(self.action_size))

    def learn(self, experiences, gamma):
        """
        - Experiences are the samples from the memory buffer to implement experience replay
        - The target network tries to predict the TD target
        - The q network tries to predict the current Q value
        :param experiences: list
        :param gamma: float
        discount factor
        :return:
        """
        states, actions, rewards, next_states, dones = experiences

        Q_targets_next = self.target_network(next_states).detach().max(1)[0].unsqueeze(1)
        Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))
        Q_expected = self.q_network(states).gather(1, actions)

        loss = F.mse_loss(Q_expected, Q_targets)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.soft_update(self.q_network, self.target_network)

    def soft_update(self, q_network_model, target_model):
        """
        Fixed target implementation

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

        :param q_network_model:
        :param target_model:
        :return:
        """
        for target_param, q_network_param in zip(target_model.parameters(), q_network_model.parameters()):
            target_param.data.copy_(self.tau * q_network_param.data + (1 - self.tau) * target_param.data)