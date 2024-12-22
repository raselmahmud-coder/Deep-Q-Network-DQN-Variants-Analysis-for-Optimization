import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load TensorBoard CSV data
data = pd.read_csv('csv.csv')

# Plot using Seaborn
sns.lineplot(data=data, x='Step', y='Value')
plt.xlabel('Episode')
plt.ylabel('Total Reward')
plt.title('Classical Control CartPole Training Rewards')
plt.show()
