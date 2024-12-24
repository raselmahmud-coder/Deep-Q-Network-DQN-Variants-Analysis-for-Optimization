# dashboard.py

import streamlit as st
import pandas as pd
import os
import log_parser
import plots
import model_analyzer
import matplotlib.pyplot as plt
import seaborn as sns

# Set Seaborn style
sns.set(style="darkgrid")

# Streamlit app starts here
def main():
    st.title("Reinforcement Learning Training Dashboard")
    
    # Sidebar for selection
    st.sidebar.header("Configuration")
    algorithms = ['DQN', 'DoubleDQN', 'DuelingDQN']
    environments = ['CartPole-v1', 'LunarLander-v3', 'MountainCar-v0']
    
    selected_algorithm = st.sidebar.selectbox("Select Algorithm", algorithms)
    selected_environment = st.sidebar.selectbox("Select Environment", environments)
    selected_checkpoint = st.sidebar.selectbox(
        "Select Checkpoint",
        ['1000.pth', '2000.pth', 'best.pth']
    )
    
    # Paths
    base_dir = '../weights'
    algo_path = os.path.join(base_dir, selected_algorithm, selected_environment)
    log_path = os.path.join(algo_path, 'training.log')
    checkpoint_path = os.path.join(algo_path, selected_checkpoint)
    
    # Display selections
    st.header(f"{selected_algorithm} on {selected_environment}")
    
    # Parse and display training metrics
    if os.path.exists(log_path):
        st.subheader("Training Metrics")
        df = log_parser.parse_training_log(log_path)
        st.write(df.tail())  # Show last few entries
        
        # Display plots
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.pyplot(generate_plot(plots.plot_rewards, df, selected_algorithm, selected_environment))
        with col2:
            st.pyplot(generate_plot(plots.plot_epsilon, df, selected_algorithm, selected_environment))
        with col3:
            st.pyplot(generate_plot(plots.plot_best_reward, df, selected_algorithm, selected_environment))
    else:
        st.warning(f"No training.log found at {log_path}")
    
    # Analyze and display model parameters
    if os.path.exists(checkpoint_path):
        st.subheader(f"Model Parameters - {selected_checkpoint}")
        # Assuming fc1.weight is a common layer
        try:
            num_states = 4 if selected_environment == 'CartPole-v1' else 8 if selected_environment == 'LunarLander-v3' else 6
            num_actions = 2 if selected_environment in ['CartPole-v1', 'MountainCar-v0'] else 4
            
            model = model_analyzer.load_model(
                selected_algorithm, selected_environment, 
                checkpoint_path, num_states, num_actions
            )
            
            # Extract weights from 'fc1.weight'
            weights = model_analyzer.extract_weights(model, 'fc1.weight')
            
            # Display weight distribution
            st.markdown("**Weight Distribution (fc1.weight)**")
            fig1, ax1 = plt.subplots(figsize=(6,4))
            sns.histplot(weights.flatten(), bins=50, kde=True, color='skyblue', ax=ax1)
            ax1.set_title("Weight Distribution")
            st.pyplot(fig1)
            
            # Optionally, display heatmap
            st.markdown("**Weight Heatmap (fc1.weight)**")
            fig2, ax2 = plt.subplots(figsize=(12,6))
            sns.heatmap(weights, cmap='viridis', ax=ax2)
            ax2.set_title("Weight Heatmap")
            st.pyplot(fig2)
        except ValueError as e:
            st.error(str(e))
    else:
        st.warning(f"No checkpoint found at {checkpoint_path}")
    
    # Optionally, allow users to compare multiple checkpoints
    st.sidebar.header("Comparison Options")
    compare_checkpoints = st.sidebar.checkbox("Compare Multiple Checkpoints")
    
    if compare_checkpoints:
        selected_checkpoints = st.sidebar.multiselect(
            "Select Checkpoints to Compare",
            ['1000.pth', '2000.pth', 'best.pth'],
            default=['1000.pth', 'best.pth']
        )
        
        if len(selected_checkpoints) >= 2:
            st.subheader("Model Parameters Comparison")
            num_states = 4 if selected_environment == 'CartPole-v1' else 8 if selected_environment == 'LunarLander-v3' else 6
            num_actions = 2 if selected_environment in ['CartPole-v1', 'MountainCar-v0'] else 4
            
            # Load selected models
            models = []
            for chk in selected_checkpoints:
                chk_path = os.path.join(algo_path, chk)
                if os.path.exists(chk_path):
                    model = model_analyzer.load_model(
                        selected_algorithm, selected_environment, 
                        chk_path, num_states, num_actions
                    )
                    models.append((chk, model))
                else:
                    st.warning(f"Checkpoint {chk} not found in {algo_path}")
            
            # Compare weights between the first two selected checkpoints
            if len(models) >= 2:
                chk1, model1 = models[0]
                chk2, model2 = models[1]
                
                weights1 = model_analyzer.extract_weights(model1, 'fc1.weight')
                weights2 = model_analyzer.extract_weights(model2, 'fc1.weight')
                
                difference = weights2 - weights1
                
                # Plot weight differences
                st.markdown(f"**Weight Differences between {chk1} and {chk2} (fc1.weight)**")
                fig_diff, ax_diff = plt.subplots(figsize=(12,6))
                sns.heatmap(difference, cmap='bwr', center=0, ax=ax_diff)
                ax_diff.set_title(f"Weight Differences: {chk2} - {chk1}")
                st.pyplot(fig_diff)
    else:
        pass  # No comparison

def generate_plot(plot_func, df, algorithm, environment):
    """
    Generates a plot using the specified plotting function.
    
    Args:
        plot_func (function): Plotting function from plots.py.
        df (pd.DataFrame): DataFrame with training metrics.
        algorithm (str): Algorithm name.
        environment (str): Environment name.
    
    Returns:
        matplotlib.figure.Figure: Generated plot.
    """
    fig, ax = plt.subplots(figsize=(12, 6))
    if plot_func == plots.plot_rewards:
        sns.lineplot(x='Episode', y='Reward', data=df, label='Reward', ax=ax)
        sns.lineplot(x='Episode', y='Best_Reward', data=df, label='Best Reward', linestyle='--', ax=ax)
        ax.set_title(f'Reward over Episodes - {algorithm} on {environment}')
        ax.set_xlabel('Episode')
        ax.set_ylabel('Reward')
        ax.legend()
    elif plot_func == plots.plot_epsilon:
        sns.lineplot(x='Episode', y='Epsilon', data=df, label='Epsilon', color='orange', ax=ax)
        ax.set_title(f'Epsilon Decay - {algorithm} on {environment}')
        ax.set_xlabel('Episode')
        ax.set_ylabel('Epsilon')
        ax.legend()
    elif plot_func == plots.plot_best_reward:
        sns.lineplot(x='Episode', y='Best_Reward', data=df, label='Best Reward', linestyle='--', color='green', ax=ax)
        sns.scatterplot(x='Episode', y='Best_Reward', data=df[df['Best_Reward_Episode'].notnull()],
                        color='red', label='New Best Reward', s=50, ax=ax)
        ax.set_title(f'Best Reward Progression - {algorithm} on {environment}')
        ax.set_xlabel('Episode')
        ax.set_ylabel('Best Reward')
        ax.legend()
    return fig

if __name__ == "__main__":
    main()
