# double_dqn.py

import torch
from base_dqn import BaseDQN
from models import DQNNetwork
from data import Data

class DoubleDQN(BaseDQN):
    def __init__(self, num_states, num_actions):
        super(DoubleDQN, self).__init__(num_states, num_actions, DQNNetwork)

    def learn(self):
        if self.learn_step_counter % self.target_update == 0:
            self.target_net.load_state_dict(self.eval_net.state_dict())

        if len(self.memory) < self.batch_size:
            return

        batch = self.memory.sample(self.batch_size)

        # Extract components from the batch
        states = torch.tensor([d.state for d in batch], dtype=torch.float).to(self.device)
        actions = torch.tensor([d.action for d in batch], dtype=torch.long).unsqueeze(1).to(self.device)
        rewards = torch.tensor([d.reward for d in batch], dtype=torch.float).unsqueeze(1).to(self.device)
        next_states = torch.tensor([d.next_state for d in batch], dtype=torch.float).to(self.device)
        dones = torch.tensor([d.done for d in batch], dtype=torch.float).unsqueeze(1).to(self.device)

        # Current Q values
        q_eval = self.eval_net(states).gather(1, actions)

        # Double DQN: action selection is from eval_net, action evaluation is from target_net
        with torch.no_grad():
            next_actions = self.eval_net(next_states).argmax(1).unsqueeze(1)
            q_next = self.target_net(next_states).gather(1, next_actions)
            q_target = rewards + self.gamma * q_next * (1 - dones)

        # Compute loss
        loss = self.loss_func(q_eval, q_target)

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Update epsilon
        self.update_epsilon()

        self.learn_step_counter += 1
