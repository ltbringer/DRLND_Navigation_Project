import numpy as np
from unityagents import UnityEnvironment



class BananaEnv(object):
    """
    Helper class wrapped around the Navigation project environment
    to simplify and increase readability.
    """
    def __init__(self, unity_env_path):
        """
        :param unity_env_path: str
        path where the environment is loaded
        """
        self.__env = UnityEnvironment(file_name=unity_env_path)
        self.__brain_name = None
        self.__brain = None
        self.__env_info = None
        self.__set_brain_name()
        self.__set_brain()

    def __get_env_brain_name(self):
        """
        @private
        get the brain name to access the state information
        :return:
        """
        return self.__env.brain_names[0]

    def __get_brain_by_name(self):
        """
        @private
        return the environment-brain
        :return:
        """
        if self.__brain_name is None:
            self.__set_brain_name()
        return self.__env.brains[self.__brain_name]

    def __set_brain_name(self):
        """
        @private
        @chainable
        set the environment's brain name
        :return:
        """
        self.__brain_name = self.__get_env_brain_name()
        return self

    def __set_brain(self):
        """
        @private
        @chainable
        set the environment brain
        :return:
        """
        self.__brain = self.__get_brain_by_name()
        return self

    def get_action_size(self):
        """
        The action space set
        :return: int
        """
        if self.__brain is None:
            self.__set_brain()
        return self.__brain.vector_action_space_size

    def get_state_size(self):
        """
        The state space set
        :return: int
        """
        return len(
            self.reset_to_initial_state()
                .get_state_snapshot()
        )

    def get_env_info(self):
        """
        wrapper to provide the value of
        private __env_info
        :return:
        """
        return self.__env_info

    def reset_to_initial_state(self, train_mode=True):
        """
        @chainable
        If the environment should be reset
        :return:
        """
        self.__env_info = self.__env.reset(train_mode=train_mode)[self.__brain_name]
        return self

    def step(self, action):
        """
        Change in the environment
        @chainable
        :param action: int
        :return:
        """
        self.__env_info = self.__env.step(action)[self.__brain_name]
        return self

    def get_state_snapshot(self):
        """
        Get the state change after performing an action in the environment
        :return:
        """
        return self.__env_info.vector_observations[0]

    def get_reward_snapshot(self):
        """
        Get the reward after performing an action in the environment
        :return:
        """
        return self.__env_info.rewards[0]

    def get_done_snapshot(self):
        """
        Get if the task was completed after performing an action in the environment
        :return:
        """
        return self.__env_info.local_done[0]

    def reaction(self):
        """
        A tuple containing state, reward and done information
        :return:
        """
        return (self.get_state_snapshot(), self.get_reward_snapshot(), self.get_done_snapshot())
