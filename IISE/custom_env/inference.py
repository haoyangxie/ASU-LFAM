import pathlib
import gymnasium as gym
import numpy as np
import torch
from ray.rllib.core.rl_module import RLModule
import register_env
import matplotlib.pyplot as plt
from custom_env import utils
from utils import *
import time

_action_to_speed = np.linspace(-0.05, 0.05, 11)


def get_action(action):
    return _action_to_speed[action]


env = gym.make('ExtruderControl-v5')
model_path = "../model_weight/v5_partition_acceleration_2"
# env = gym.make('ExtruderControl-v6')
# model_path = "../model_weight/v6_paper_case_study"


rl_module = RLModule.from_checkpoint(
    pathlib.Path(model_path) / "learner" / "module_state"/ "default_policy"
)
terminated = truncated = False
obs, info = env.reset()
action_list = [10,10,9,10,10,10,10,10,10,0,0,10,0,0,0,0,0,0,0,0,0,0,0,0,0]
while not terminated and not truncated:
    # Compute the next action from a batch (B=1) of observations.
    torch_obs_batch = torch.from_numpy(np.array([obs]))
    action_logits = rl_module.forward_inference({"obs": torch_obs_batch})[
        "action_dist_inputs"
    ]
    action = torch.argmax(action_logits[0]).numpy()
    obs, reward, terminated, truncated, info = env.step(5)
# v5_speed_history = env.speed_history
# temp_diff = []
# for i in range(25):
#     temp_diff.append(np.mean(env.temp_diff_history[i]))
# print(np.mean(temp_diff))

# objective = 0
# count = 0
# for i in range(25):
#     for td in env.temp_diff_history[i]:
#         count += 1
#         objective += (td - 120)**2
# print(objective)
print(f"Reached episode return of {env.cumulative_reward}.")
# time_spent = np.array(env.time_spent)
# layer_time_list = []
# for i in range(20):
#     layer_time = 0
#     for j in range(25):
#         layer_time += sum(time_spent[i][j])
#     layer_time_list.append(layer_time)
# print(np.mean(layer_time_list))
# plt.plot(list(range(20)), layer_time_list, color='blue')
# plt.xlabel('Layer Index', fontsize=20)
# plt.ylabel('Layer Time (s)', fontsize=20)
# plt.legend(frameon=False, fontsize=16)
# plt.xticks(fontsize=18)
# x_ticks = range(0, len(layer_time_list), 2)
# plt.xticks(ticks=x_ticks)
# plt.yticks(fontsize=18)
# plt.tight_layout()
# plt.savefig('../paper_plot/case_study_layer_time.png')
# plt.show()

# print(sum(time_spent[0][0]))

# layer_time = []
# for i in range(20):
#     total_time = 0
#     for j in range(25):
#         total_time += sum(env.time_spent[i][j])
#         layer_time.append(total_time)
# print(layer_time)

# total_time = 0
# for ls in env.time_spent:
#     total_time += sum(ls)
# print(total_time)

# temp_diff_list = []
# for i in range(20):
#     layer_time_diff = 0
#     for j in range(25):
#         layer_time_diff = np.mean(env.temp_diff_history[i][j])
#     temp_diff_list.append(layer_time_diff)
# print(np.mean(temp_diff_list))
# plt.plot(temp_diff_list, color='blue')
# plt.xlabel('Layer Index', fontsize=20)
# plt.ylabel('Temperature Difference (℃)', fontsize=20)
# plt.legend(frameon=False, fontsize=16)
# plt.xticks(fontsize=18)
# x_ticks = range(0, 20, 2)
# plt.xticks(ticks=x_ticks)
# plt.yticks(fontsize=18)
# plt.tight_layout()
# plt.savefig('../paper_plot/case_study_temp_difference.png')
# plt.show()
# print(np.mean(temp_diff_list))

# res = 0
# for i in range(54):
#     for j in range(660):
#         res += env.temp_diff_history[i][j]**2
# print(res)




# env = gym.make('ExtruderControl-v4')
# model_path = "../model_weight/v4_true_data_with_params"
#
# rl_module = RLModule.from_checkpoint(
#     pathlib.Path(model_path) / "learner" / "module_state"/ "default_policy"
# )
#
# terminated = truncated = False
# obs, info = env.reset()
# while not terminated and not truncated:
#     # Compute the next action from a batch (B=1) of observations.
#     torch_obs_batch = torch.from_numpy(np.array([obs]))
#     action_logits = rl_module.forward_inference({"obs": torch_obs_batch})[
#         "action_dist_inputs"
#     ]
#     action = torch.argmax(action_logits[0]).numpy()
#     obs, reward, terminated, truncated, info = env.step(action)
# v4_speed_history = env.speed_history
# print(f"Reached episode return of {env.cumulative_reward}.")
#
#
# plt.plot(v4_speed_history)
# plt.plot(v5_speed_history)
# plt.savefig('./env_speed_comparison.png')
# plt.show()