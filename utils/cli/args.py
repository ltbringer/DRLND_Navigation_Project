import argparse
from utils.cli import defaults


def fetch_args():
    parser = argparse.ArgumentParser(
        description='Teach an agent to pick up yellow bananas from blue implementing Deep-Q Learning'
    )
    parser.add_argument(
        '--env-path',
        help='Path to the unity environment',
        type=str
    )
    parser.add_argument(
        '--model-path',
        help='Path to a trained q-network\'s checkpoint(.pth) file',
        type=str
    )
    parser.add_argument(
        '--episodes',
        help='Number of episodes for which the agent must be trained',
        type=int,
        default=defaults.EPISODES
    )
    parser.add_argument(
        '--time-steps',
        help='Number of steps to be taken in an episode',
        type=int,
        default=defaults.TIME_STEPS
    )
    parser.add_argument(
        '--qualify-score',
        help='Score at which the training must stop',
        type=int,
        default=defaults.QUALIFY_SCORE
    )
    parser.add_argument(
        '--score-window',
        help='Number of episodes for which the qualify-score should be maintained as average',
        type=int,
        default=defaults.SCORE_WINDOW
    )
    parser.add_argument(
        '--buffer-size',
        help='Number of episodes to keep in memory (for experience replay)',
        type=int,
        default=defaults.BUFFER_SIZE
    )
    parser.add_argument(
        '--batch-size',
        help='Number of samples/batch for training the Q network',
        type=int,
        default=defaults.BATCH_SIZE
    )
    parser.add_argument(
        '--gamma',
        help='Discount factor of the rewards',
        type=float,
        default=defaults.GAMMA
    )
    parser.add_argument(
        '--lr',
        help='Learning rate',
        type=float,
        default=defaults.LEARNING_RATE
    )
    parser.add_argument(
        '--eps-start',
        help='initial epsilon for epsilon greedy',
        type=float,
        default=defaults.EPSILON_START
    )
    parser.add_argument(
        '--eps-decay',
        help='The value by which the initial epsilon must decay over-time',
        type=float,
        default=defaults.EPSILON_DECAY
    )
    parser.add_argument(
        '--eps-end',
        help='The minimum value of epsilon beyond which there should be no decay',
        type=float,
        default=defaults.EPSILON_END
    )
    parser.add_argument(
        '--tau',
        help='The degree of influence the target network has on the main/local network',
        type=float,
        default=defaults.TAU
    )
    parser.add_argument(
        '--update-every',
        help='Number of time-steps in an episode after which the Q network should be updated',
        type=int,
        default=defaults.UPDATE_EVERY
    )
    parser.add_argument(
        '--fc1_units',
        help='neurons in the first fully connected layer',
        type=int,
        default=defaults.FC1_UNITS
    )
    parser.add_argument(
        '--fc2_units',
        help='neurons in the second fully connected layer',
        type=int,
        default=defaults.FC2_UNITS
    )
    parser.add_argument(
        '--seed',
        help='Random seed to ensure same results',
        type=float,
        default=defaults.SEED
    )

    return parser.parse_args()
