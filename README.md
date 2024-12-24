# Project outline

![ DQN VS DDQN of CartPole](./Results/cartpole%20dqn%20vs%20ddqn.png)

Observed Differences from the Plots:
Stability:

The reward curve for DQN shows high variability, with frequent spikes and dips, indicating instability in the learning process.
The Double DQN reward curve appears more consistent, with a smoother decline in exploration-based rewards early on.
Performance:

Both algorithms failed to solve the environment (CartPole is solved at an average reward of 195 over 100 episodes).
Double DQN demonstrates more stable behavior but with a slightly lower peak in rewards compared to DQN.
Convergence Speed:

DQN's erratic behavior shows that it struggles to converge effectively.
Double DQN, although smoother, appears to require additional tuning to reach optimal performance.
Explanation of the Differences:
Overestimation Bias in DQN:

In DQN, Q-values are updated using the maximum estimated Q-value from the same network, leading to an overestimation bias that destabilizes learning.
Double DQN mitigates this bias by using the evaluation network to select actions and the target network to estimate their values, resulting in better stability.
Exploration vs Exploitation:

The smoother reward curve in Double DQN suggests that the algorithm balances exploration and exploitation more effectively, reducing the impact of random actions on the learning process.
Hyperparameters:

The differences may also stem from suboptimal hyperparameter tuning for both methods. Adjusting epsilon, learning rate, and replay buffer size could improve convergence for both algorithms.
