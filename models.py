# models.py

import torch.nn as nn
import torch.nn.functional as F

class DQNNetwork(nn.Module):
    def __init__(self, num_inputs, num_actions):
        super(DQNNetwork, self).__init__()
        self.fc1 = nn.Linear(num_inputs, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, num_actions)
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        q_values = self.fc3(x)
        return q_values

class DuelingDQNNetwork(nn.Module):
    def __init__(self, num_inputs, num_actions):
        super(DuelingDQNNetwork, self).__init__()
        self.fc1 = nn.Linear(num_inputs, 128)
        self.fc2 = nn.Linear(128, 128)
        
        # Value Stream
        self.value_fc = nn.Linear(128, 64)
        self.value = nn.Linear(64, 1)
        
        # Advantage Stream
        self.advantage_fc = nn.Linear(128, 64)
        self.advantage = nn.Linear(64, num_actions)
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        
        # Value Stream
        val = F.relu(self.value_fc(x))
        val = self.value(val)
        
        # Advantage Stream
        adv = F.relu(self.advantage_fc(x))
        adv = self.advantage(adv)
        
        # Combine streams
        q_vals = val + (adv - adv.mean(dim=1, keepdim=True))
        return q_vals
