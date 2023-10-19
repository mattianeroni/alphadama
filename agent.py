import torch 
import torch.nn as nn
import torch.optim as optim

import random 
import collections
import numpy as np 

from model import decode, encode


class DamaAgent:

    def __init__(self, model, optimizer=None, criterion=None, lr=0.001, max_memory=10_000, 
            batch_size=100, gamma=0.9, randomness=0.01):
        self.randomness = randomness
        self.gamma = gamma
        self.model = model
        self.n_games = 0
        self.lr = lr 
        self.batch_size = batch_size
        self.max_memory = max_memory
        self.memory = collections.deque(maxlen=max_memory)
        self.optimizer = optim.Adam(model.parameters(), lr=lr) if optimizer is None else optimizer
        self.criterion = nn.MSELoss() if criterion is None else criterion
        
    def move (self, grid):
        """ Method to make an AI movement """
        if random.random() < self.randomness:
            random_output = torch.rand((grid.shape[0], 4, 8, 8))
            return decode(grid, random_output)
        in_tensor = encode(grid)
        out_tensor = self.model.__call__(in_tensor)
        return decode(grid, out_tensor)

    def train (self, state, action, reward, next_state, done):
        """ Short training step """
        # conversions and type checks
        action = torch.tensor(action, dtype=torch.int)
        reward = torch.tensor(reward, dtype=torch.float)
        done = torch.tensor(done, dtype=torch.bool)
        # estimate next Q-values
        out_tensor = self.model.__call__(state)
        target = out_tensor.clone()
        target[:, action[:,0], action[:,1], action[:,2]] = torch.where(done, reward, 
                                reward + self.gamma * torch.max(self.model.__call__(next_state)))
        # reset gradients and backpropagate the error
        self.optimizer.zero_grad()
        loss = self.criterion(target, out_tensor)
        loss.backward()
        self.optimizer.step()