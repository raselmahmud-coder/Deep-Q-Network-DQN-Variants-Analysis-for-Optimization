import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import collections
from torch.utils.tensorboard import SummaryWriter
import os

# Set random seed for reproducibility
SEED = 0
np.random.seed(SEED)
torch.manual_seed(SEED)

# Hyperparameters
ENV_NAME = "MountainCar-v0"
EPISODES = 1000
LEARNING_RATE = 0.001
GAMMA = 0.99
BATCH_SIZE = 64
MEMORY_CAPACITY = 10000
EPSILON_START = 1.0
EPSILON_END = 0.01
EPSILON_DECAY = 500  # Decay over 500 episodes
TARGET_UPDATE_INTERVAL = 10
SAVE_PATH = "./mountaincar_dqn"
os.makedirs(SAVE_PATH, exist_ok=True)

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define Replay Memory
class ReplayMemory:
    def __init__(self, capacity):
        self.buffer = collections.deque(maxlen=capacity)

    def add(self, transition):
        self.buffer.append(transition)

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            torch.tensor(states, dtype=torch.float).to(device),
            torch.tensor(actions, dtype=torch.long).to(device),
            torch.tensor(rewards, dtype=torch.float).to(device),
            torch.tensor(next_states, dtype=torch.float).to(device),
            torch.tensor(dones, dtype=torch.float).to(device),
        )

    def __len__(self):
        return len(self.buffer)

# Define Neural Network
class DQNNetwork(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DQNNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, output_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Define DQN Agent
class DQNAgent:
    def __init__(self, env):
        self.env = env
        self.state_dim = env.observation_space.shape[0]
        self.action_dim = env.action_space.n
        self.eval_net = DQNNetwork(self.state_dim, self.action_dim).to(device)
        self.target_net = DQNNetwork(self.state_dim, self.action_dim).to(device)
        self.target_net.load_state_dict(self.eval_net.state_dict())
        self.optimizer = optim.Adam(self.eval_net.parameters(), lr=LEARNING_RATE)
        self.memory = ReplayMemory(MEMORY_CAPACITY)
        self.epsilon = EPSILON_START
        self.global_step = 0

    def choose_action(self, state):
        if np.random.rand() < self.epsilon:
            return self.env.action_space.sample()
        else:
            state = torch.tensor(state, dtype=torch.float).unsqueeze(0).to(device)
            with torch.no_grad():
                action_values = self.eval_net(state)
            return torch.argmax(action_values).item()

    def update_target_network(self):
        self.target_net.load_state_dict(self.eval_net.state_dict())

    def train(self):
        if len(self.memory) < BATCH_SIZE:
            return

        states, actions, rewards, next_states, dones = self.memory.sample(BATCH_SIZE)

        # Compute Q-values for current states
        q_values = self.eval_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)

        # Compute target Q-values
        with torch.no_grad():
            max_next_q_values = self.target_net(next_states).max(1)[0]
            target_q_values = rewards + GAMMA * max_next_q_values * (1 - dones)

        # Compute loss and update
        loss = nn.MSELoss()(q_values, target_q_values)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()

    def store_transition(self, state, action, reward, next_state, done):
        self.memory.add((state, action, reward, next_state, done))

# Main Training Loop
def train_dqn():
    env = gym.make(ENV_NAME)
    env.reset(seed=SEED)
    env.action_space.seed(SEED)
    agent = DQNAgent(env)
    writer = SummaryWriter(log_dir=SAVE_PATH)

    global_step = 0
    for episode in range(EPISODES):
        state, _ = env.reset()
        episode_reward = 0
        while True:
            global_step += 1

            # Choose action and step in environment
            action = agent.choose_action(state)
            next_state, reward, done, _, _ = env.step(action)
            agent.store_transition(state, action, reward, next_state, done)

            # Train the agent
            loss = agent.train()

            # Accumulate reward
            episode_reward += reward
            state = next_state

            if done:
                writer.add_scalar("Episode Reward", episode_reward, episode)
                if loss:
                    writer.add_scalar("Loss", loss, episode)
                print(
                    f"Episode: {episode}, Reward: {episode_reward:.2f}, Epsilon: {agent.epsilon:.2f}"
                )
                break

        # Decay epsilon
        agent.epsilon = max(
            EPSILON_END, EPSILON_START - episode / EPSILON_DECAY
        )

        # Update target network
        if episode % TARGET_UPDATE_INTERVAL == 0:
            agent.update_target_network()

    writer.close()
    env.close()

    # Save model
    torch.save(agent.eval_net.state_dict(), os.path.join(SAVE_PATH, "dqn_model.pth"))
    print("Training complete and model saved.")

    return agent

# Record Agent Following Gymnasium's Documentation
def record_agent(agent, video_path="./videos"):
    env = gym.make(ENV_NAME, render_mode="rgb_array")
    env = gym.wrappers.RecordVideo(
        env, video_path, episode_trigger=lambda x: True, disable_logger=True
    )
    state, _ = env.reset()
    while True:
        action = agent.choose_action(state)
        state, reward, done, _, _ = env.step(action)
        env.render()
        if done:
            break
    env.close()
    print(f"Video recorded at {video_path}")

if __name__ == "__main__":
    dqn_agent = train_dqn()
    record_agent(dqn_agent, video_path="./videos/mountaincar")
