# Stock-Predicttion-using-RL

## Description

This project implements a custom stock trading environment using gym-anytrading for reinforcement learning. The goal is to create an agent that can maximize profits by buying and selling stocks based on historical price data. I have used GME Stocks data from MarketWatch of the previous 1 year(June 2023- June 2024). You can also import given datasets in the respective envirnoments you are using.

## Installation

Requirements are: 
tensorflow-gpu, 
tensorflow,
stable-baselines3,
gym-anytrading,
gym,
numpy, pandas, matplotlib

```
import gymnasium as gym
import gym_anytrading



from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3 import A2C

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
```
## DataFrame

Use pandas to read the csv file, and we need to change the dtype of 'Date' columnn from object to datetime and also set it as index.
Visit this link to download the dataframe: https://www.marketwatch.com/investing/stock/gme/download-data?startDate=11/1/2019&endDate=03/12/2021
```
df = pd.read_csv("data/gme-date.csv")
df['Date'] =pd.to_datetime(df['Date'])
df.set_index('Date', inplace=True)
```
## Custom Environment

The new gym package does not allow you to directly use 'stock-v0' or 'forex-v0'. You have to register the environment using the code below, it also handles error if our environment is already registered. After registering the Env, set the dataframe, frame_bound(datapoint starting and ending point), window_size(size of sliding window)

```
from gym.envs.registration import register

try:
    register(
        id='stocks-v0',
        entry_point='gym_anytrading.envs:StocksEnv',
    )
except Exception as e:
    print(f"Environment might already be registered: {e}")

env = gym.make('stocks-v0',df=df, frame_bound= (5,200), window_size=5)
```

##Training the model

Reset the Env, Take a random action from the action space, use the action to get next_state, reward and other info. Then we are making a Dummy vectorized environment to loop through the environment multiple time if we want but here we are going to do it once. Then we are going to use A2C algorithm from the gym, and we are using a MlpPolicy(which tries to predict a probability distribution of actions to be taken from that step).

```
state= env.reset()

while True:
    action = env.action_space.sample()
    next_state, reward,terminated, truncated, info = env.step(action)
    done=terminated or truncated

    if done:
        print("info", info)
        break

env_maker=lambda: gym.make('stocks-v0', df=df, frame_bound=(5,200), window_size=5)
env= DummyVecEnv([env_maker])

model=A2C('MlpPolicy', env, verbose=1)
model.learn(total_timesteps=100000)

```
## Evaluation

Create an Env using the left datapoints to test our model. We want our explained_variance to be in the range of 0 to 1. You will see as training progresses our model will start to learn the variances and starts to understand the pattern. 
```
env = gym.make('stocks-v0', df=df, frame_bound=(201,250), window_size=5)
obs = env.reset()
if isinstance(obs, tuple):
    obs = obs[0]  # Adjust based on your specific observation structure

# Make sure obs is a numpy array
obs = np.array(obs)

while True:
    obs = obs[np.newaxis, ...]  # Add batch dimension
    action, _states = model.predict(obs)
    obs, rewards,terminated,truncated, info = env.step(action)
    done = terminated or truncated
    
    if isinstance(obs, tuple):
        obs = obs[0]  # Adjust based on your specific observation structure

    obs = np.array(obs)  # Ensure obs is a numpy array

    if done:
        print("info", info)
        break
plt.figure(figsize=(15,6))
plt.cla()
env.render_all()
plt.show()

```
The red points show the the points where we sell and green points show the points where we buy. The profit we get is (total_profit -1)*100 percent.
![image](https://github.com/Sarveshgits/Stock-Predicttion-using-RL/assets/139525935/8fa858cf-e8f2-4abb-a050-9e8925668f9b)


