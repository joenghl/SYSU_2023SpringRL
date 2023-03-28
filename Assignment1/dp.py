import copy
import numpy as np
from env.grid_scenarios import MiniWorld


# Hypar-parameters that could be helpful.
GAMMA = 0.9
EPSILON = 0.001
BLOCKS = [14, 15, 21, 27]
R = [
    0, 0, 0, 0, 0, 0,
    0, 0, 0, 1, 0, 0,
    0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, -1,
    0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0,
]


def policy_iteration():
    pass


def value_iteration():
    pass


if __name__ == "__main__":
    env = MiniWorld()
    n_state = env.observation_space.n
    n_action = env.action_space.n

    ######################################################
    # write your code to get a convergent value table v. #
    ######################################################

    env.update_r(v)
    for _ in range(2000):
        env.render()
