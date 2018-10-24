import numpy as np
import random
import torch
from collections import namedtuple, deque


class ReplayBuffer():
    """
    The data structure that holds an agents buffer_size number of episode data
    to be used for experience replay
    """
    def __init__(self, action_size, buffer_size, batch_size, seed):
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.seed = seed

    def add(self, state, action, reward, next_state, done):
        """
        Add to experience memory
        :param state: list
        :param action: int
        :param reward: float
        :param next_state: list
        :param done: bool
        :return: None
        """
        experience = self.experience(state, action, reward, next_state, done)
        self.memory.append(experience)

    def sample(self):
        """
        Return a sample from the list of collected episodes for learning
        :return: tuple
        """
        experiences = random.sample(self.memory, k=self.batch_size)
        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(self.device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).long().to(self.device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(self.device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(
            self.device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(
            self.device)
        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        return len(self.memory)