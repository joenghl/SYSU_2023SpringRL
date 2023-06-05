import numpy as np
from utils.make_env import make_env


ENV = make_env('simple_spread', discrete_action=True)
N_AGENT = 3
N_ACTION = ENV.action_space[0].n


class Agents:
    def __init__(self):
        pass

    @staticmethod
    def act(obs):
        actions = []
        for i in range(N_AGENT):
            action = ENV.action_space[i].sample()
            actions.append(np.eye(N_ACTION, dtype=np.float32)[action])
        return actions
