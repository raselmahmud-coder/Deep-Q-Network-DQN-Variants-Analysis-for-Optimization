import os
import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import deque
import random
from torch.utils.tensorboard import SummaryWriter

class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.gamma = 0.99  # discount rate
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.model = self._build_model()

    def _build_model(self):
        model = nn.Sequential(
            nn.Linear(self.state_size, 24),
            nn.ReLU(),
            nn.Linear(24, 24),
            nn.ReLU(),
            nn.Linear(24, self.action_size)
        )
        return model

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return np.random.randint(self.action_size)
        state = torch.tensor(state, dtype=torch.float).unsqueeze(0)
        act_values = self.model(state)
        return torch.argmax(act_values).item()

    def replay(self, batch_size):
        if len(self.memory) < batch_size:
            return
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            state = torch.tensor(state, dtype=torch.float)
            next_state = torch.tensor(next_state, dtype=torch.float)
            target = reward
            if not done:
                target += self.gamma * torch.max(self.model(next_state)).item()
            target_f = self.model(state).detach().clone()
            target_f[action] = target
            self.model.zero_grad()
            criterion = nn.MSELoss()
            loss = criterion(self.model(state), target_f)
            loss.backward()
            optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
            optimizer.step()
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

def run(is_training=True, render=False):
    env = gym.make('CartPole-v1', render_mode='human' if render else None)
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    agent = DQNAgent(state_size, action_size)
    episodes = 1000
    batch_size = 32

    rewards_per_episode = []

    # Initialize TensorBoard writer
    writer = SummaryWriter(log_dir='./tensorboard_logs')

    for e in range(episodes):
        state, _ = env.reset()
        total_reward = 0
        done = False
        while not done:
            action = agent.act(state)
            next_state, reward, done, truncated, _ = env.step(action)
            reward = reward if not done else -10
            agent.remember(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward
            if done:
                rewards_per_episode.append(total_reward)
                mean_rewards = np.mean(rewards_per_episode[-100:])
                print(f"Episode {e}/{episodes}, Reward: {total_reward}, Mean Reward: {mean_rewards:.2f}, Epsilon: {agent.epsilon:.2f}")
                writer.add_scalar('Reward', total_reward, e)  # Log reward to TensorBoard
                break
        if is_training:
            agent.replay(batch_size)
        if np.mean(rewards_per_episode[-100:]) > 195:
            print("Environment solved!")
            break

    writer.close()  # Close TensorBoard writer

    if is_training:
        with open('cartpole_dqn.pkl', 'wb') as f:
            pickle.dump(agent.model.state_dict(), f)

    plt.plot(rewards_per_episode)
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.savefig('cartpole_dqn_rewards.png')
    env.close()

if __name__ == '__main__':
    # Training phase
    run(is_training=True, render=False)

    # Testing phase
    run(is_training=False, render=True)
