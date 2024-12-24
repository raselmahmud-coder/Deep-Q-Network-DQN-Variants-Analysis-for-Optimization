import os
import re
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set(style="darkgrid")

def parse_log_file(log_path):
    pattern = re.compile(
        r"(?P<timestamp>[\d\-:\s,]+) - Episode: (?P<episode>\d+), Reward: (?P<reward>[\d\.]+), Epsilon: (?P<epsilon>[\d\.]+)"
    )
    data = []
    with open(log_path, 'r') as file:
        for line in file:
            match = pattern.search(line)
            if match:
                entry = match.groupdict()
                data.append({
                    'episode': int(entry['episode']),
                    'reward': float(entry['reward']),
                    'epsilon': float(entry['epsilon'])
                })
    return pd.DataFrame(data)

def aggregate_data(weights_dir):
    all_data = []
    for algo in os.listdir(weights_dir):
        algo_path = os.path.join(weights_dir, algo)
        if os.path.isdir(algo_path):
            env_dir = os.path.join(algo_path, 'CartPole-v1')
            if os.path.isdir(env_dir):
                log_path = os.path.join(env_dir, 'training.log')
                if os.path.exists(log_path):
                    df = parse_log_file(log_path)
                    df['algorithm'] = algo
                    df['environment'] = 'CartPole-v1'
                    all_data.append(df)
                else:
                    print(f"Warning: No training.log found at {log_path}")
    if all_data:
        combined_df = pd.concat(all_data, ignore_index=True)
        return combined_df
    else:
        print("No data found to aggregate.")
        return pd.DataFrame()

def compute_metrics(df, window=50):
    df_sorted = df.sort_values(['algorithm', 'episode'])
    df_sorted['reward_ma'] = df_sorted.groupby('algorithm')['reward'].transform(lambda x: x.rolling(window, min_periods=1).mean())
    df_sorted['reward_std'] = df_sorted.groupby('algorithm')['reward'].transform(lambda x: x.rolling(window, min_periods=1).std())
    return df_sorted

def plot_rewards(df, save_path=None):
    plt.figure(figsize=(12, 8))
    sns.lineplot(data=df, x='episode', y='reward_ma', hue='algorithm', ci='sd')
    plt.title('Comparison of Reward Moving Average Across Algorithms')
    plt.xlabel('Episode')
    plt.ylabel('Reward (Moving Average)')
    plt.legend(title='Algorithm')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
        print(f"Reward comparison plot saved to {save_path}")
    else:
        plt.show()
    plt.close()

def plot_epsilon(df, save_path=None):
    plt.figure(figsize=(12, 8))
    sns.lineplot(data=df, x='episode', y='epsilon', hue='algorithm', ci=None)
    plt.title('Epsilon Decay Across Algorithms')
    plt.xlabel('Episode')
    plt.ylabel('Epsilon')
    plt.legend(title='Algorithm')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
        print(f"Epsilon decay plot saved to {save_path}")
    else:
        plt.show()
    plt.close()

def plot_convergence(df, threshold=200, save_path=None):
    convergence = df[df['reward'] >= threshold].groupby('algorithm')['episode'].min().reset_index()
    plt.figure(figsize=(10, 6))
    sns.barplot(data=convergence, x='algorithm', y='episode')
    plt.title(f'Episode to Reach Reward Threshold of {threshold}')
    plt.xlabel('Algorithm')
    plt.ylabel('Episode')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
        print(f"Convergence plot saved to {save_path}")
    else:
        plt.show()
    plt.close()

def plot_stability(df, window=50, save_path=None):
    plt.figure(figsize=(12, 8))
    for algo in df['algorithm'].unique():
        algo_df = df[df['algorithm'] == algo].sort_values('episode')
        algo_df = algo_df.copy()
        algo_df['reward_ma'] = algo_df['reward'].rolling(window, min_periods=1).mean()
        algo_df['reward_std'] = algo_df['reward'].rolling(window, min_periods=1).std()
        plt.plot(algo_df['episode'], algo_df['reward_std'], label=algo)
    plt.title(f'Stability of Rewards Over Episodes (Window Size = {window})')
    plt.xlabel('Episode')
    plt.ylabel('Reward Standard Deviation')
    plt.legend(title='Algorithm')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
        print(f"Stability plot saved to {save_path}")
    else:
        plt.show()
    plt.close()

def plot_combined_metrics(df, save_path=None):
    fig, axes = plt.subplots(2, 2, figsize=(20, 15), constrained_layout=True)

    # Plot Reward Moving Average
    sns.lineplot(data=df, x='episode', y='reward_ma', hue='algorithm', ax=axes[0, 0], ci='sd')
    axes[0, 0].set_title('Reward Moving Average')
    axes[0, 0].set_xlabel('Episode')
    axes[0, 0].set_ylabel('Reward (Moving Average)')

    # Plot Epsilon Decay
    sns.lineplot(data=df, x='episode', y='epsilon', hue='algorithm', ax=axes[0, 1], ci=None)
    axes[0, 1].set_title('Epsilon Decay')
    axes[0, 1].set_xlabel('Episode')
    axes[0, 1].set_ylabel('Epsilon')

    # Plot Convergence Speed
    convergence = df[df['reward'] >= 200].groupby('algorithm')['episode'].min().reset_index()
    sns.barplot(data=convergence, x='algorithm', y='episode', ax=axes[1, 0])
    axes[1, 0].set_title('Convergence Speed (Episode to Reach Reward ≥ 200)')
    axes[1, 0].set_xlabel('Algorithm')
    axes[1, 0].set_ylabel('Episode')

    # Plot Stability
    for algo in df['algorithm'].unique():
        algo_df = df[df['algorithm'] == algo].sort_values('episode')
        algo_df = algo_df.copy()
        algo_df['reward_ma'] = algo_df['reward'].rolling(50, min_periods=1).mean()
        algo_df['reward_std'] = algo_df['reward'].rolling(50, min_periods=1).std()
        axes[1, 1].plot(algo_df['episode'], algo_df['reward_std'], label=algo)
    axes[1, 1].set_title('Stability of Rewards Over Episodes (Window Size = 50)')
    axes[1, 1].set_xlabel('Episode')
    axes[1, 1].set_ylabel('Reward Standard Deviation')
    axes[1, 1].legend(title='Algorithm')

    if save_path:
        plt.savefig(save_path)
        print(f"Combined metrics plot saved to {save_path}")
    else:
        plt.show()
    plt.close()

def main():
    # Path to the weights directory
    weights_dir = 'weights'

    # Step 1: Aggregate data from all training logs
    df = aggregate_data(weights_dir)

    if df.empty:
        print("No data available for plotting.")
        return

    # Step 2: Compute moving averages and standard deviations
    df = compute_metrics(df, window=50)

    # Step 3: Generate and save plots
    plot_rewards(df, save_path='comparison_rewards.png')
    plot_epsilon(df, save_path='comparison_epsilon_decay.png')
    plot_convergence(df, threshold=200, save_path='convergence_speed.png')
    plot_stability(df, window=50, save_path='stability_rewards.png')
    plot_combined_metrics(df, save_path='combined_metrics.png')

    print("All plots have been generated and saved successfully.")

if __name__ == "__main__":
    main()
