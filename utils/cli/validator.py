from utils.cli import defaults


class ValidatedArgs(object):

    def __init__(self, args, has_model=False):
        self.__args = args
        self.__has_model = has_model

    def env_path(self):
        __ERROR_ENV_PATH = '--env-path must be a str, which should be a path where the unity model is saved.'
        if not self.__args.env_path:
            raise TypeError(__ERROR_ENV_PATH)
        return self.__args.env_path

    def model_path(self):
        if self.__has_model:
            __ERROR_ENV_PATH = '--model-path must be a str, which should be a path where the Q-network\s weights are saved.'
            if not self.__args.model_path:
                raise TypeError(__ERROR_ENV_PATH)
            return self.__args.model_path

    def episodes(self):
        return self.__args.episodes \
            if type(self.__args.episodes) is int \
            else defaults.EPISODES

    def time_steps(self):
        return self.__args.time_steps \
            if type(self.__args.time_steps) is int \
            else defaults.TIME_STEPS

    def qualify_score(self):
        return self.__args.qualify_score \
            if type(self.__args.qualify_score) is int \
            else defaults.QUALIFY_SCORE

    def score_window(self):
        return self.__args.score_window \
            if type(self.__args.score_window) is int \
            else defaults.SCORE_WINDOW

    def buffer_size(self):
        return self.__args.buffer_size \
            if type(self.__args.buffer_size) is int \
            else defaults.BUFFER_SIZE

    def batch_size(self):
        return self.__args.batch_size \
            if type(self.__args.batch_size) is int \
            else defaults.BATCH_SIZE

    def gamma(self):
        return self.__args.gamma \
            if type(self.__args.gamma) is float and \
               self.__args.gamma >= 0 and \
               self.__args.gamma <= 1 \
            else defaults.GAMMA

    def tau(self):
        return self.__args.tau \
            if type(self.__args.tau) is float and \
               self.__args.tau >= 0 and \
               self.__args.tau <= 1 \
            else defaults.TAU

    def lr(self):
        return self.__args.lr \
            if type(self.__args.lr) is float and \
               self.__args.lr >= 0 and \
               self.__args.lr <= 1 \
            else defaults.LEARNING_RATE

    def eps_start(self):
        return self.__args.eps_start \
            if type(self.__args.eps_start) is float and \
               self.__args.eps_start >= 0 and \
               self.__args.eps_start <= 1 \
            else defaults.EPSILON_START

    def eps_decay(self):
        return self.__args.eps_decay \
            if type(self.__args.eps_decay) is float and \
               self.__args.eps_decay >= 0 and \
               self.__args.eps_decay <= 1 \
            else defaults.EPSILON_DECAY

    def eps_end(self):
        return self.__args.eps_end \
            if type(self.__args.eps_end) is float and \
               self.__args.eps_end >= 0 and \
               self.__args.eps_end <= 1 \
            else defaults.EPSILON_END

    def update_every(self):
        return self.__args.update_every \
            if type(self.__args.update_every) is int \
            else defaults.UPDATE_EVERY

    def fc1_units(self):
        return self.__args.fc1_units \
            if type(self.__args.fc1_units) is int \
            else defaults.FC1_UNITS

    def fc2_units(self):
        return self.__args.fc2_units \
            if type(self.__args.fc2_units) is int \
            else defaults.FC2_UNITS

    def seed(self):
        return self.__args.seed \
            if type(self.__args.seed) is float and \
               self.__args.seed <= 0 and \
               self.__args.seed >= 1\
            else defaults.SEED
