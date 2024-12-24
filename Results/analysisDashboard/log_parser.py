# log_parser.py

import re
import pandas as pd

def parse_training_log(log_path):
    """
    Parses a training.log file to extract episode number, reward, and epsilon.
    
    Args:
        log_path (str): Path to the training.log file.
    
    Returns:
        pd.DataFrame: DataFrame containing the parsed data.
    """
    episode_pattern = re.compile(r"Episode: (\d+), Reward: ([\d\.]+), Epsilon: ([\d\.]+)")
    best_reward_pattern = re.compile(r"New best reward: ([\d\.]+) at episode (\d+)")
    
    episodes = []
    rewards = []
    epsilons = []
    best_rewards = []
    best_reward_episode = []
    
    current_best = -float('inf')
    
    with open(log_path, 'r') as file:
        for line in file:
            episode_match = episode_pattern.search(line)
            if episode_match:
                episode = int(episode_match.group(1))
                reward = float(episode_match.group(2))
                epsilon = float(episode_match.group(3))
                
                episodes.append(episode)
                rewards.append(reward)
                epsilons.append(epsilon)
                
                # Check for new best reward
                if reward > current_best:
                    current_best = reward
                    best_rewards.append(current_best)
                    best_reward_episode.append(episode)
                else:
                    best_rewards.append(current_best)
                    best_reward_episode.append(None)  # No new best in this episode
    
    data = pd.DataFrame({
        'Episode': episodes,
        'Reward': rewards,
        'Epsilon': epsilons,
        'Best_Reward': best_rewards,
        'Best_Reward_Episode': best_reward_episode
    })
    
    return data


# log_parser.py (continued)

import os

def aggregate_all_logs(base_dir='weights'):
    """
    Traverses the directory structure to parse all training.log files.
    
    Args:
        base_dir (str): Base directory containing algorithm subdirectories.
    
    Returns:
        pd.DataFrame: Combined DataFrame with all parsed log data, including algorithm and environment labels.
    """
    combined_data = pd.DataFrame()
    
    for algorithm in os.listdir(base_dir):
        algo_path = os.path.join(base_dir, algorithm)
        if os.path.isdir(algo_path):
            for environment in os.listdir(algo_path):
                env_path = os.path.join(algo_path, environment)
                if os.path.isdir(env_path):
                    log_path = os.path.join(env_path, 'training.log')
                    if os.path.exists(log_path):
                        df = parse_training_log(log_path)
                        df['Algorithm'] = algorithm
                        df['Environment'] = environment
                        combined_data = pd.concat([combined_data, df], ignore_index=True)
                    else:
                        print(f"Warning: No training.log found in {env_path}")
    return combined_data


# Example usage
if __name__ == "__main__":
    combined_df = aggregate_all_logs(base_dir='weights')
    print(combined_df.head())
    # Save to CSV if needed
    combined_df.to_csv('aggregated_training_logs.csv', index=False)
