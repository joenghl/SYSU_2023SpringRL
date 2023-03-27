"""
Toy environment examples.
"""

from gym import spaces

from env.grid_core import GridWorldEnv


class LargeGridWorld(GridWorldEnv):
    """
    10*10 grid world, refer to https://cs.stanford.edu/people/karpathy/reinforcejs/gridworld_td.html.
    """
    def __init__(self):
        super().__init__(u_size=40,
                         n_width=10,
                         n_height=10,
                         default_reward=0,
                         default_type=0,
                         windy=False)
        self.start = (0, 9)
        self.ends = [(5, 4)]
        self.types = [
            (4, 2, 1), (4, 3, 1), (4, 4, 1), (4, 5, 1), (4, 6, 1), (4, 7, 1),
            (1, 7, 1), (2, 7, 1), (3, 7, 1), (4, 7, 1), (6, 7, 1), (7, 7, 1),
            (8, 7, 1)
        ]
        self.rewards = [
            (3, 2, -1), (3, 6, -1), (5, 2, -1), (6, 2, -1), (8, 3, -1),

            (8, 4, -1), (5, 4, 1), (6, 4, -1), (5, 5, -1), (6, 5, -1)
        ]
        self.refresh_setting()


class SimpleGridWorld(GridWorldEnv):
    """
    10*7 grid world.
    """
    def __init__(self):
        super().__init__(u_size=60,
                         n_width=10,
                         n_height=7,
                         default_reward=-1,
                         default_type=0,
                         windy=False)
        self.start = (0, 3)
        self.ends = [(7, 3)]
        self.rewards = [(7, 3, 20)]
        self.max_step = 100
        self.refresh_setting()


class MiniWorld(GridWorldEnv):
    """
    6*6 grid world.
    """
    def __init__(self):
        super().__init__(u_size=60,
                         n_width=6,
                         n_height=6,
                         default_reward=0,
                         default_type=0,
                         windy=False)
        self.start = (0, 5)
        self.ends = [(5, 3), (3, 1)]
        self.types = [
            (2, 2, 1), (3, 2, 1), (3, 3, 1), (3, 4, 1),
        ]
        self.rewards = [(3, 1, 1), (5, 3, -1)]
        self.max_step = 100
        self.refresh_setting()

    def update_r(self, values):
        """
        This function is only used for rendering env when using dynamic programming method.
        """
        v_min = values.min()
        v_max = values.max()
        v_norm = (2 * (values - v_min) / (v_max - v_min) - 1) * 8.0
        for i in range(len(v_norm)):
            x, y = self._state_to_xy(i)
            self.grids.set_reward(x, y, v_norm[i])


class WindyGridWorld(GridWorldEnv):
    """
    10*7 grid world with wind.
    """
    def __init__(self):
        super().__init__(u_size=60,
                         n_width=10,
                         n_height=7,
                         default_reward=-1,
                         default_type=0,
                         windy=True)
        self.start = (0, 3)
        self.ends = [(7, 3)]
        self.rewards = [(7, 3, 1)]
        self.refresh_setting()


class RandomWalk(GridWorldEnv):
    """
    Random walk demo.
    """
    def __init__(self):
        super().__init__(u_size=80,
                         n_width=7,
                         n_height=1,
                         default_reward=0,
                         default_type=0,
                         windy=False)
        self.action_space = spaces.Discrete(2)
        self.start = (3, 0)
        self.ends = [(6, 0), (0, 0)]
        self.rewards = [(6, 0, 1)]
        self.refresh_setting()


class CliffWalk(GridWorldEnv):
    """
    cliff walk demo.
    """
    def __init__(self):
        super().__init__(u_size=60,
                         n_width=12,
                         n_height=4,
                         default_reward=-1,
                         default_type=0,
                         windy=False)
        self.action_space = spaces.Discrete(4)
        self.start = (0, 0)
        self.ends = [(11, 0)]
        for i in range(10):
            self.rewards.append((i+1, 0, -100))
            self.ends.append((i+1, 0))
        self.refresh_setting()


class SkullAndTreasure(GridWorldEnv):
    """
    Example of Skull and Money explained the necessity and effectiveness.
    """
    def __init__(self):
        super().__init__(u_size=60,
                         n_width=5,
                         n_height=2,
                         default_reward=-1,
                         default_type=0,
                         windy=False)
        self.action_space = spaces.Discrete(4)
        self.start = (0, 1)
        self.ends = [(2, 0)]
        self.rewards = [(0, 0, -100), (2, 0, 100), (4, 0, -100)]
        self.types = [(1, 0, 1), (3, 0, 1)]
        self.refresh_setting()


if __name__ == "__main__":
    # You can render the env here if you want to know what the env looks like.
    env = MiniWorld()
    env.reset()
    n_state = env.observation_space
    n_action = env.action_space
    print("nfs:%s; nfa:%s" % (n_state, n_action))
    env.render()

    for _ in range(20000):
        env.render()
        # a = env.action_space.sample()
        # state, reward, done, info = env.step(a)
        # print("{0}, {1}, {2}, {3}".format(a, reward, done, info))

    print("env closed")
