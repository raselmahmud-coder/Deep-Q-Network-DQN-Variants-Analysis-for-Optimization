# plots.py

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

sns.set(style="darkgrid")

def plot_rewards(df, algorithm, environment, save_path=None):
    plt.figure(figsize=(12, 6))
    sns.lineplot(x='Episode', y='Reward', data=df, label='Reward')
    sns.lineplot(x='Episode', y='Best_Reward', data=df, label='Best Reward', linestyle='--')
    plt.title(f'Reward over Episodes - {algorithm} on {environment}')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.legend()
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()

def plot_epsilon(df, algorithm, environment, save_path=None):
    plt.figure(figsize=(12, 6))
    sns.lineplot(x='Episode', y='Epsilon', data=df, label='Epsilon', color='orange')
    plt.title(f'Epsilon Decay - {algorithm} on {environment}')
    plt.xlabel('Episode')
    plt.ylabel('Epsilon')
    plt.legend()
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()

def plot_best_reward(df, algorithm, environment, save_path=None):
    plt.figure(figsize=(12, 6))
    sns.lineplot(x='Episode', y='Best_Reward', data=df, label='Best Reward', linestyle='--', color='green')
    plt.scatter(df['Episode'], df['Best_Reward'], where=df['Best_Reward_Episode'].notnull(),
                color='red', label='New Best Reward', s=50)
    plt.title(f'Best Reward Progression - {algorithm} on {environment}')
    plt.xlabel('Episode')
    plt.ylabel('Best Reward')
    plt.legend()
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()


# Example usage
if __name__ == "__main__":
    import log_parser
    import plots
    import os
    
    combined_df = log_parser.aggregate_all_logs(base_dir='weights')
    
    algorithms = combined_df['Algorithm'].unique()
    environments = combined_df['Environment'].unique()
    
    for algo in algorithms:
        for env in environments:
            subset = combined_df[(combined_df['Algorithm'] == algo) & (combined_df['Environment'] == env)]
            if not subset.empty:
                env_path = os.path.join('weights', algo, env)
                os.makedirs(env_path, exist_ok=True)
                
                # Plot Reward
                plots.plot_rewards(
                    subset, algo, env, 
                    save_path=os.path.join(env_path, 'reward_plot.png')
                )
                
                # Plot Epsilon
                plots.plot_epsilon(
                    subset, algo, env, 
                    save_path=os.path.join(env_path, 'epsilon_plot.png')
                )
                
                # Plot Best Reward
                plots.plot_best_reward(
                    subset, algo, env, 
                    save_path=os.path.join(env_path, 'best_reward_plot.png')
                )
            else:
                print(f"No data for {algo} on {env}")
