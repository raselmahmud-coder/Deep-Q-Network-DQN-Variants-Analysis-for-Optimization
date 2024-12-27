# main.py

import os
import torch
import gymnasium as gym
import argparse
from dqn import DQN
from double_dqn import DoubleDQN
from dueling_dqn import DuelingDQN
from data import Data
from utils.logger import setup_logger
from utils.video_recorder import record_video

# Hyperparameters
EPISODES = 2000
BATCH_SIZE = 64
LR = 0.00025
GAMMA = 0.98
SAVING_ITERATION = 1000
MEMORY_CAPACITY = 10000
MIN_CAPACITY = 500
TARGET_UPDATE = 10
EPSILON_MIN = 0.01
EPSILON_DECAY = 1000
SEED = 42

def set_seed(env, seed=SEED):
    env.reset(seed=seed)
    env.action_space.seed(seed)
    torch.manual_seed(seed)
    import numpy as np
    import random
    np.random.seed(seed)
    random.seed(seed)

def get_agent(algorithm, num_states, num_actions):
    if algorithm == 'DQN':
        return DQN(num_states, num_actions)
    elif algorithm == 'DoubleDQN':
        return DoubleDQN(num_states, num_actions)
    elif algorithm == 'DuelingDQN':
        return DuelingDQN(num_states, num_actions)
    else:
        raise ValueError(f"Unknown algorithm: {algorithm}")

def main():
    parser = argparse.ArgumentParser(description='DQN, Double DQN, and Dueling DQN across Environments')
    parser.add_argument('--algorithm', type=str, required=True, choices=['DQN', 'DoubleDQN', 'DuelingDQN'], help='Algorithm to use')
    parser.add_argument('--environment', type=str, required=True, choices=['CartPole-v1', 'MountainCar-v0', 'LunarLander-v3'], help='Gym environment name')
    parser.add_argument('--test', action='store_true', help='Set to test the agent instead of training')
    args = parser.parse_args()
    
    # Initialize environment
    if args.environment == 'LunarLander-v3':
        env = gym.make("LunarLander-v3", continuous=False)
    else:
        env = gym.make(args.environment)
    
    set_seed(env, SEED)
    
    num_actions = env.action_space.n if isinstance(env.action_space, gym.spaces.Discrete) else env.action_space.shape[0]
    num_states = env.observation_space.shape[0]
    
    # Initialize agent
    agent = get_agent(args.algorithm, num_states, num_actions)
    
    # Setup directories
    save_dir = f"./results/weights/{args.algorithm}/{args.environment}/"
    os.makedirs(save_dir, exist_ok=True)
    
    video_dir = f"./results/videos/{args.algorithm}/{args.environment}"
    os.makedirs(video_dir, exist_ok=True)
    
    # Setup logger
    log_path = f"./results/weights/{args.algorithm}/{args.environment}/training.log"
    logger = setup_logger(log_path)
    
    best_reward = -float('inf')
    
    if args.test:
        # Load the best model
        model_path = f"{save_dir}/best.pth"
        if not os.path.exists(model_path):
            print(f"No model found at {model_path}. Train the agent first.")
            return
        agent.load_model(model_path)
        agent.epsilon = 0.0  # Greedy policy
        
        # Record video
        video_path = f"{video_dir}/best_strategy.mp4"
        record_video(env, agent, video_path)
        return
    
    # Training loop
    for episode in range(1, EPISODES + 1):
        state, info = env.reset()
        ep_reward = 0
        done = False
        
        while not done:
            action = agent.choose_action(state)
            next_state, reward, done, truncated, info = env.step(action)
            agent.store_transition(Data(state, action, reward, next_state, done))
            ep_reward += reward
            state = next_state
            
            if agent.memory_counter >= MIN_CAPACITY:
                agent.learn()
        
        # Logging
        logger.info(f"Episode: {episode}, Reward: {ep_reward}, Epsilon: {agent.epsilon:.4f}")
        
        # Save the best model
        if ep_reward > best_reward:
            best_reward = ep_reward
            torch.save(agent.eval_net.state_dict(), f"{save_dir}/best.pth")
            logger.info(f"New best reward: {best_reward} at episode {episode}")
        
        # Periodic checkpoint saving
        if episode % SAVING_ITERATION == 0:
            torch.save(agent.eval_net.state_dict(), f"{save_dir}/{episode}.pth")
            logger.info(f"Model saved at episode {episode}")
        
        # Print progress every 100 episodes
        if episode % 100 == 0:
            print(f"Algorithm: {args.algorithm}, Environment: {args.environment}, Episode: {episode}, Reward: {ep_reward}, Epsilon: {agent.epsilon:.4f}")
    
    env.close()
    
    # After training, record video of the best agent
    agent.load_model(f"{save_dir}/best.pth")
    video_path = f"{video_dir}/best_strategy.mp4"
    record_video(env, agent, video_path)

if __name__ == '__main__':
    main()
