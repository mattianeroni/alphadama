import torch 
import torch.nn as nn
import torch.optim as optim

import random 
import collections

from model import decode, encode


class DamaAgent:

    def __init__(self, model, optimizer=None, criterion=None, lr=0.001, max_memory=10_000, 
            batch_size=100, randomness=0.01):
        self.randomness = randomness
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

    def train (self, grid, action, reward, next_grid, done):
        """ Short training step """
        grid = torch.tensor(grid, dtype=torch.float)
        next_grid = torch.tensor(next_grid, dtype=torch.float)
        action = torch.tensor(action, dtype=torch.long)
        reward = torch.tensor(reward, dtype=torch.float)

        # Convert single batch to multiple batches
        if len(grid.shape) == 3:
            grid = torch.unsqueeze(grid, 0)
            next_grid = torch.unsqueeze(next_grid, 0)
            action = torch.unsqueeze(action, 0)
            reward = torch.unsqueeze(reward, 0)
            done = (done, )

        # 1: predicted Q values with current state
        pred = self.model.__call__(state)

        target = pred.clone()
        for idx in range(len(done)):
            Q_new = reward[idx]
            if not done[idx]:
                Q_new = reward[idx] + self.gamma * torch.max(self.model(next_state[idx]))

            target[idx][torch.argmax(action[idx]).item()] = Q_new
    
        # 2: Q_new = r + y * max(next_predicted Q value) -> only do this if not done
        # pred.clone()
        # preds[argmax(action)] = Q_new
        self.optimizer.zero_grad()
        loss = self.criterion(target, pred)
        loss.backward()

        self.optimizer.step()

    def long_train (self):
        """ Long training step """
        pass