import gymnasium as gym
import matplotlib.pyplot as plt

import register_env
import numpy as np
from utils import get_original_speed


# env = gym.make('ExtruderControl-v0')
# env = gym.make('ExtruderControl-v1')
# env = gym.make('ExtruderControl-v2')
# env = gym.make('ExtruderControl-v3')
# env = gym.make('ExtruderControl-v4')
# env = gym.make('ExtruderControl-v5')
env = gym.make('ExtruderControl-v6')
action_list = [10,
 10,
 9,
 10,
 10,
 10,
 10,
 10,
 10,
 0,
 0,
 10,
 0,
 0,
 0,
 0,
 0,
 0,
 0,
 0,
 0,
 0,
 0,
 0,
 0]

for i in range(1):
    step = 0
    state, _ = env.reset()
    temp_diff = []
    terminated, truncated = False, False
    while not terminated and not truncated:
        # action = action_list[step]
        step += 1
        # action = env.action_space.sample()
        state, reward, terminated, truncated, info = env.step(5)
        # temp_diff.append(env.state["temperature_diff"])
    print(f"Episode {i} reward: {env.cumulative_reward}")
    print(step)


# total_time = 0
# for ls in env.time_spent:
#     total_time += sum(ls)
# print(total_time)

