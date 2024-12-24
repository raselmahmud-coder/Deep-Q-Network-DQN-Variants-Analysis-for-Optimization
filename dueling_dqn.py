# dueling_dqn.py
import random
import torch
import torch.nn as nn
import torch.optim as optim
from models import DuelingDQNNetwork
from memory import Memory
from data import Data

class DuelingDQN:
    def __init__(self, num_states, num_actions):
        self.num_actions = num_actions
        self.gamma = 0.98
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 1000
        self.batch_size = 64
        self.target_update = 10
        self.learn_step_counter = 0
        self.memory_counter = 0
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.eval_net = DuelingDQNNetwork(num_states, num_actions).to(self.device)
        self.target_net = DuelingDQNNetwork(num_states, num_actions).to(self.device)
        self.target_net.load_state_dict(self.eval_net.state_dict())
        self.target_net.eval()
        
        self.optimizer = optim.Adam(self.eval_net.parameters(), lr=0.00025)
        self.loss_func = nn.MSELoss()
        
        self.memory = Memory(capacity=10000)
    
    def choose_action(self, state):
        if random.random() > self.epsilon:
            state = torch.tensor(state, dtype=torch.float).unsqueeze(0).to(self.device)
            with torch.no_grad():
                q_values = self.eval_net(state)
            action = torch.argmax(q_values).item()
        else:
            action = random.randint(0, self.num_actions - 1)
        return action
    
    def store_transition(self, data):
        self.memory.push(data)
        self.memory_counter += 1
    
    def learn(self):
        if self.learn_step_counter % self.target_update == 0:
            self.target_net.load_state_dict(self.eval_net.state_dict())
        
        if len(self.memory) < self.batch_size:
            return
        
        batch = self.memory.sample(self.batch_size)
        # batch = Data(*zip(*batch))
        
        # Correctly extract each component from the batch of Data objects
        states = torch.tensor([d.state for d in batch], dtype=torch.float).to(self.device)
        actions = torch.tensor([d.action for d in batch], dtype=torch.long).unsqueeze(1).to(self.device)
        rewards = torch.tensor([d.reward for d in batch], dtype=torch.float).unsqueeze(1).to(self.device)
        next_states = torch.tensor([d.next_state for d in batch], dtype=torch.float).to(self.device)
        dones = torch.tensor([d.done for d in batch], dtype=torch.float).unsqueeze(1).to(self.device)
        
        # Current Q values
        q_eval = self.eval_net(states).gather(1, actions)
        
        # Current Q values
        q_eval = self.eval_net(states).gather(1, actions)
        
        # Compute target Q values
        with torch.no_grad():
            q_next = self.target_net(next_states).max(1)[0].unsqueeze(1)
            q_target = rewards + self.gamma * q_next * (1 - dones)
        
        # Compute loss
        loss = self.loss_func(q_eval, q_target)
        
        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # Update epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon -= (1.0 - self.epsilon_min) / self.epsilon_decay
            self.epsilon = max(self.epsilon, self.epsilon_min)
        
        self.learn_step_counter += 1
    
    def save_model(self, path):
        torch.save(self.eval_net.state_dict(), path)
    
    def load_model(self, path):
        self.eval_net.load_state_dict(torch.load(path, map_location=self.device))
        self.target_net.load_state_dict(self.eval_net.state_dict())
