import numpy as np
import pandas as pd
import gym
import gym_anytrading
from stable_baselines3 import DQN
from matplotlib import pyplot as plt

# Example data
data = {
    'Open': np.random.rand(300),
    'High': np.random.rand(300),
    'Low': np.random.rand(300),
    'Close': np.random.rand(300),
    'Volume': np.random.randint(1, 100, size=300)
}
df = pd.DataFrame(data)
from gym.envs.registration import register

try:
    register(
        id='stocks-v0',
        entry_point='gym_anytrading.envs:StocksEnv',
    )
except Exception as e:
    print(f"Environment might already be registered: {e}")

env = gym.make('stocks-v0', df=df, frame_bound=(201, 250), window_size=5)

# Initialize or load a model
model = DQN('MlpPolicy', env, verbose=1)

# Define reward function
def compute_reward(portfolio_value, previous_portfolio_value, risk_penalty_factor=0.1, reward_factor=1.0):
    profit = portfolio_value - previous_portfolio_value
    risk_penalty = risk_penalty_factor * abs(profit)
    return reward_factor * profit - risk_penalty

obs = env.reset()

# Track portfolio value
previous_portfolio_value = env.initial_balance  # Assuming you have an initial balance attribute

while True:
    obs = obs[np.newaxis, ...]  # Add batch dimension
    action, _states = model.predict(obs)
    obs, rewards, done, info = env.step(action)

    # Assuming you have a way to get the current portfolio value
    portfolio_value = info['portfolio_value']  # Update with your actual method of getting portfolio value

    # Compute reward
    reward = compute_reward(portfolio_value, previous_portfolio_value)
    previous_portfolio_value = portfolio_value

    if done:
        print("info", info)
        break

# Plot the results
plt.figure(figsize=(15, 6))
plt.cla()
env.render_all()
plt.show()
