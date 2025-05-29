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
model_path = "../model_weight/v4_acceleration"


def get_action(action):
    return _action_to_speed[action]


env = gym.make('ExtruderControl-v5')
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
    return temp_diff, env.temp_diff_history, env.speed_history, env.time_spent, env.v_inside_partition

abs_temp_diff_ppo, temp_diff_ppo, speed_history_ppo, time_spent_ppo, v_inside = run_policy_collect_data()
abs_temp_diff_fixed, temp_diff_fixed, speed_history_fixed, time_spent_fixed, _ = run_policy_collect_data('Fixed')

# improvement = []
# for i in range(54):
#     part_improvement = 0
#     for j in range(660):
#         part_improvement += np.abs(temp_diff_fixed[i][j]) - np.abs(temp_diff_ppo[i][j])
#     improvement.append(part_improvement/660)
# plt.bar(range(len(improvement)), improvement)
# plt.savefig('./partition_improvement.png')
# plt.show()
v_history = []
for i in range(54):
    for j in range(660):
        v_history.append(v_inside[i][j])
plt.plot(v_history)
plt.savefig('./ppo_speed_history.png')
plt.show()
