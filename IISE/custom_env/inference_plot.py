import pathlib
import gymnasium as gym
import numpy as np
import torch
from ray.rllib.core.rl_module import RLModule
import register_env
import itertools
import matplotlib.pyplot as plt
from custom_env import utils
from utils import *

_action_to_speed = np.linspace(-0.05, 0.05, 11)
model_path = "../model_weight/v4_120"


def get_action(action):
    return _action_to_speed[action]


env = gym.make('ExtruderControl-v4')
rl_module = RLModule.from_checkpoint(
    pathlib.Path(model_path) / "learner" / "module_state"/ "default_policy"
)


def run_policy_collect_data(policy='PPO'):
    '''
    return: policy temp diff, speed history, time used
    '''
    temp_diff = []
    terminated = truncated = False
    obs, info = env.reset()
    while not terminated and not truncated:
        torch_obs_batch = torch.from_numpy(np.array([obs]))
        action_logits = rl_module.forward_inference({"obs": torch_obs_batch})["action_dist_inputs"]
        if policy == 'PPO':
            action = torch.argmax(action_logits[0]).numpy()
        elif policy == 'Fixed':
            action = 5
        obs, reward, terminated, truncated, info = env.step(action)
        temp_diff.append(env.state["temperature_diff"])
    print(f"Reached episode return of {env.cumulative_reward}.")
    return temp_diff, env.temp_diff_history, env.speed_history, env.time_spent


def plot_speed_history(speed_history_dict):
    for speed_history in speed_history_dict:
        plt.plot(speed_history_dict[speed_history], label=speed_history)
    plt.xlabel('Location')
    plt.ylabel('Speed (mm/s)')
    plt.legend()
    plt.savefig('./meeting_plot/speed_histroy_true_data_120.png')
    plt.clf()


def plot_temp_diff(temp_diff_dict):
    for temp_diff in temp_diff_dict:
        length = len(temp_diff_dict[temp_diff][-54:])
        plt.scatter(range(length), temp_diff_dict[temp_diff][-54:], label=temp_diff)
    plt.legend()
    plt.xlabel('Location')
    plt.ylabel('Temperature Difference')
    plt.savefig("./meeting_plot/performance_comparison_true_data_120.png")
    plt.clf()


def plot_multilayer_temp_diff(temp_diff_dict):
    for temp_diff in temp_diff_dict:
        length = len(temp_diff_dict[temp_diff])
        plt.scatter(range(length), temp_diff_dict[temp_diff], label=temp_diff)
    plt.legend()
    plt.xlabel('Location')
    plt.ylabel('Temperature Difference')
    plt.savefig("./meeting_plot/performance_comparison_true_data.png")
    plt.clf()


def plot_policy_difference(policy_1, policy_2):
    # plt.scatter(range(54), [policy_2[i] - policy_1[i] for i in range(54)])
    improvement = [policy_2[i] - policy_1[i] for i in range(54)]
    plt.bar(range(54), improvement)

    # plt.plot([policy_2[i] - policy_1[i] for i in range(54)])
    # plt.legend()
    plt.xlabel('Location')
    plt.ylabel('Temperature Difference Improvement')
    plt.savefig("./meeting_plot/performance_improvement_true_data_120_value.png")
    plt.clf()


def temp_diff_value_comparison(temp_diff_fixed, temp_diff_ppo):
    plt.plot(temp_diff_fixed, label='Fixed')
    plt.plot(temp_diff_ppo, label='PPO')
    plt.ylim(-20, 20)
    plt.axhline(y=0, color='black', linestyle='-', linewidth=1)
    plt.xlabel('Location')
    plt.ylabel('Temperature Difference')
    plt.legend()
    plt.savefig("./meeting_plot/temp_diff_true_data_120.png")


abs_temp_diff_ppo, temp_diff_ppo, speed_history_ppo, time_spent_ppo = run_policy_collect_data()
abs_temp_diff_fixed, temp_diff_fixed, speed_history_fixed, time_spent_fixed = run_policy_collect_data('Fixed')
speed_history_dict = {"PPO": speed_history_ppo, "Fixed": speed_history_fixed}
temp_diff_dict = {"PPO": abs_temp_diff_ppo, "Fixed": abs_temp_diff_fixed}

import pickle

with open('../pkl_file/abs_temp_diff_ppo.pkl', 'wb') as file:
    pickle.dump(abs_temp_diff_ppo, file)

with open('../pkl_file/temp_diff_ppo.pkl', 'wb') as file:
    pickle.dump(temp_diff_ppo, file)

with open('../pkl_file/speed_history_ppo.pkl', 'wb') as file:
    pickle.dump(speed_history_ppo, file)

with open('../pkl_file/time_spent_ppo.pkl', 'wb') as file:
    pickle.dump(time_spent_ppo, file)

with open('../pkl_file/abs_temp_diff_fixed.pkl', 'wb') as file:
    pickle.dump(abs_temp_diff_fixed, file)

with open('../pkl_file/temp_diff_fixed.pkl', 'wb') as file:
    pickle.dump(temp_diff_fixed, file)

with open('../pkl_file/speed_history_fixed.pkl', 'wb') as file:
    pickle.dump(speed_history_fixed, file)

with open('../pkl_file/time_spent_fixed.pkl', 'wb') as file:
    pickle.dump(time_spent_fixed, file)




# plot_speed_history(speed_history_dict)
# plot_temp_diff(temp_diff_dict)
# plot_policy_difference(abs_temp_diff_ppo, abs_temp_diff_fixed)
# temp_diff_value_comparison(temp_diff_fixed, temp_diff_ppo)






