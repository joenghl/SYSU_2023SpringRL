import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from utils.make_env import make_env


ENV = make_env('simple_spread', discrete_action=True)
N_AGENT = 3
N_ACTION = ENV.action_space[0].n  # 5
N_OBS = ENV.observation_space[0].shape[0]  # 18


class Actor(nn.Module):
    """
    MLP network for example (can be used as value or policy)
    """
    def __init__(self, input_dim, out_dim, hidden_dim=64, nonlin=F.relu, out_fn=F.softmax):
        """
        Inputs:
            input_dim (int): Number of dimensions in input
            out_dim (int): Number of dimensions in output
            hidden_dim (int): Number of hidden dimensions
            nonlin (PyTorch function): Nonlinearity to apply to hidden layers
        """
        super(Actor, self).__init__()

        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, out_dim)
        self.nonlin = nonlin
        self.out_fn = out_fn

    def forward(self, X):
        """
        Inputs:
            X (PyTorch Matrix): Batch of observations
        Outputs:
            out (PyTorch Matrix): Output of network (actions, values, etc)
        """
        h1 = self.nonlin(self.fc1(X))
        h2 = self.nonlin(self.fc2(h1))
        out = self.out_fn(self.fc3(h2))
        return out


class Agents:
    def __init__(self, n_obs, n_action):
        # this is an example of different agents params, the shared params is also allowed
        agent1 = Actor(n_obs, n_action)
        agent1.load_state_dict(torch.load(r'Assignment2\agents\random_network\agent1.pth'))

        agent2 = Actor(n_obs, n_action)
        agent2.load_state_dict(torch.load(r'Assignment2\agents\random_network\agent2.pth'))

        agent3 = Actor(n_obs, n_action)
        agent3.load_state_dict(torch.load(r'Assignment2\agents\random_network\agent3.pth'))

        self.agents =[agent1, agent2, agent3]

    def act(self, obs):
        torch_obs = [Variable(torch.Tensor(obs[i]).view(1, -1), requires_grad=False)
                     for i in range(N_AGENT)]
        actions = []
        for i in range(N_AGENT):
            action = self.agents[i](torch_obs[i]).argmax()
            action = np.eye(N_ACTION)[action]
            actions.append(action)
        return actions
