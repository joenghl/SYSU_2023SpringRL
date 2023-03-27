import numpy as np
import copy
from env.grid_scenarios import MiniWorld


GAMMA = 0.8
EPSILON = 0.001
BLOCKS = [14, 15, 21, 27]


def prob(v):
    max_indices = np.where(v == v.max())[0]
    p = np.array([0 for _ in range(len(v))], dtype=np.float32)
    p[max_indices] = 1.0 / len(max_indices)
    return p


def policy_evaluation(env):
    n_state = env.observation_space.n
    v_last = [0 for _ in range(n_state)]
    v_old = [0 for _ in range(n_state)]
    v = [0 for _ in range(n_state)]
    t = 0
    while True:
        t += 1
        for i in range(n_state):
            # left
            if i % env.n_width == 0 or (i - 1) in BLOCKS:
                i_left = i
            else:
                i_left = i - 1

            # right
            if (i + 1) % env.n_width == 0 or (i + 1) in BLOCKS:
                i_right = i
            else:
                i_right = i + 1

            # up
            if (i + env.n_width) >= n_state or (i + env.n_width) in BLOCKS:
                i_up = i
            else:
                i_up = i + env.n_width

            # down
            if (i - env.n_width) < 0 or (i - env.n_width) in BLOCKS:
                i_down = i
            else:
                i_down = i - env.n_width

            s_next = np.array([i_left, i_right, i_up, i_down])
            v_next = np.array([v_old[i_left], v_old[i_right], v_old[i_up], v_old[i_down]])
            v_tmp = 0.0
            v_list = []
            for j in range(4):
                v_tmp = rewards[s_next[j]] + GAMMA * v_next[j]
                v_list.append(v_tmp)
            v[i] = max(v_list)

        if max(abs(np.array(v)-np.array(v_old))) < EPSILON:
            break
        else:
            v_old = copy.deepcopy(v)
        print(v)

    print(f'done after {t} iteration.')
    # print(v)
    for _ in range(2000):
        env.update_r(np.array(v))
        env.render()
    return v


def policy_improvement(env, v):
    n_state = env.observation_space.n
    p = np.empty((n_state, 4), dtype=np.float32)
    for i in range(n_state):
        # left
        if i % env.n_width == 0 or (i - 1) in BLOCKS:
            i_left = i
        else:
            i_left = i - 1

        # right
        if (i + 1) % env.n_width == 0 or (i + 1) in BLOCKS:
            i_right = i
        else:
            i_right = i + 1

        # up
        if (i + env.n_width) >= n_state or (i + env.n_width) in BLOCKS:
            i_up = i
        else:
            i_up = i + env.n_width

        # down
        if (i - env.n_width) < 0 or (i - env.n_width) in BLOCKS:
            i_down = i
        else:
            i_down = i - env.n_width

        probs = prob(np.array([v[i_left], v[i_right], v[i_up], v[i_down]]))
        p[i] = probs

    return p


def softmax(x):
    x -= np.max(x)
    return np.exp(x) / np.sum(np.exp(x))



if __name__ == "__main__":
    env = MiniWorld()
    env.reset()
    old_policy_v = np.array([0 for _ in range(env.observation_space.n)])
    rewards = [0 for _ in range(env.observation_space.n)]
    rewards[9] = 1
    rewards[23] = -1
    # old_policy = policy_improvement(env, old_policy_v)
    # tmp = 0
    # while True:
    #     tmp += 1
    #     v_policy = policy_evaluation(env, old_policy)
    #     policy = policy_improvement(env, v_policy)
    #     if old_policy.all() == policy.all():
    #         print(tmp)
    #         for _ in range(2000):
    #             env.render()
    #         break
    #     else:
    #         old_policy = copy.deepcopy(policy)
    policy_evaluation(env)
