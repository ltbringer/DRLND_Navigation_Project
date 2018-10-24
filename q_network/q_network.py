import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
from utils.cli import defaults


def q_network(state_size, action_size, fc1_units=defaults.FC1_UNITS, fc2_units=defaults.FC2_UNITS):
    """
    The Neural network that optimizes the Q values for the navigation project.
    starting with the input layer having `state_size` number of neurons
    the first hidden layer has fc1_units number of neurons
    the second hidden layer has fc2_units number of neurons
    the output layer has `action_size` number of neurons

    :param state_size: int
    :param action_size: int
    :param fc1_units: int
    :param fc2_units: int
    :return:
    """
    return nn.Sequential(
        OrderedDict([
            ('fc1', nn.Linear(state_size, fc1_units)),
            ('reLU', nn.ReLU()),
            ('fc2', nn.Linear(fc1_units, fc2_units)),
            ('reLU', nn.ReLU()),
            ('fc3', nn.Linear(fc2_units, action_size)),
        ])
    )
