import numpy as np
import torch
from collections import deque
from utils.cli import defaults
from utils.timestamp import get_current_timestamp

def dqn(
        agent,
        env,
        episodes=defaults.EPISODES,
        max_t=defaults.TIME_STEPS,
        qualify_score=defaults.QUALIFY_SCORE,
        eps_start=defaults.EPSILON_START,
        eps_end=defaults.EPSILON_END,
        eps_decay=defaults.EPSILON_DECAY
):
    """
    trains a given agent for the navigation project.

    :param agent: Agent
    The banana collecting agent

    :param env:
    The environment

    :param episodes: int
    maximum number of training episodes

    :param max_t: int
    maximum number of time-steps per episode

    :param eps_start: float
    starting value of epsilon, for epsilon-greedy action selection

    :param eps_end: float
    minimum value of epsilon

    :param eps_decay: float
    multiplicative factor (per episode) for decreasing epsilon

    :return: list
    """
    scores = []  # list containing scores from each episode
    scores_window = deque(maxlen=100)  # last 100 scores
    eps = eps_start  # initialize epsilon
    for i_episode in range(1, episodes + 1):
        state = env.reset_to_initial_state().get_state_snapshot()
        score = 0
        for t in range(max_t):
            action = agent.act(state, eps)
            next_state, reward, done = env.step(action).reaction()
            agent.step(state, action, reward, next_state, done)
            state = next_state
            score += reward
            if done:
                break
        scores_window.append(score)  # save most recent score
        scores.append(score)  # save most recent score
        eps = max(eps_end, eps_decay * eps)  # decrease epsilon
        print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)), end="")
        if i_episode % 100 == 0:
            print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)))
        if np.mean(scores_window) >= qualify_score:
            print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(i_episode - 100,
                                                                                         np.mean(scores_window)))
            torch.save(agent.qnetwork_local.state_dict(), 'checkpoint_{}.pth'.format(get_current_timestamp()))
            break
    return scores
