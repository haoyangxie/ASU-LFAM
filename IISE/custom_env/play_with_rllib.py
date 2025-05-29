from ray.tune.registry import register_env
from custom_env.extruder_control_env_fixed_profiles import ExtruderControlEnvFixed
from custom_env.extruder_control_env_hard_constraint import ExtruderControlEnvHardConstraint
from custom_env.extruder_control_env_multiple_layer import ExtruderControlEnvMultipleLayer
from custom_env.extruder_control_env_acceleration import ExtruderControlEnvAcceleration
from custom_env.extruder_control_env_multiple_layer_case_study import ExtruderControlEnvAllLayers
from ray.rllib.algorithms import ppo
from pprint import pprint
from ray.rllib.algorithms.ppo import PPOConfig
from env_state_mlp import EnvStateMLP
from ray.rllib.models import ModelCatalog
from ray.tune.logger import TBXLoggerCallback
from ray import tune
import pickle
import time

ModelCatalog.register_custom_model("env_state_mlp", EnvStateMLP)


config = (
    PPOConfig()
    .api_stack(
        enable_rl_module_and_learner=True,
        enable_env_runner_and_connector_v2=True,
    )
    .environment(ExtruderControlEnvAllLayers)
    .env_runners(sample_timeout_s=None)
    .env_runners(num_env_runners=8, num_envs_per_env_runner=4)
    .training(train_batch_size=80000)
    # .resources()
    # .framework("torch")
    # .training(
    #     model={
    #     "custom_model": "env_state_mlp",
    # })
    # .sample_timeout_s(30)
)

algo = config.build()

mean_episode_rewards = []
max_episode_rewards = []
min_episode_rewards = []
policy_loss = []
vf_loss = []
total_loss = []

for i in range(201):
    print('epoch:', i)
    result = algo.train()
    result.pop("config")
    # pprint(result)
    # pprint(result['env_runners']['agent_episode_returns_mean'])
    pprint(result['env_runners']['episode_return_mean'])
    max_episode_rewards.append(result['env_runners']['episode_return_max'])
    mean_episode_rewards.append(result['env_runners']['episode_return_mean'])
    min_episode_rewards.append(result['env_runners']['episode_return_min'])
    policy_loss.append(result['learners']['default_policy']['policy_loss'])
    total_loss.append(result['learners']['default_policy']['total_loss'])
    vf_loss.append(result['learners']['default_policy']['vf_loss'])

    if i > 0 and i % 100 == 0:
        checkpoint_dir = algo.save("../model_weight/v6_paper_case_study")
        print(f"Checkpoint saved in directory {checkpoint_dir}")

with open('all_layers_policy_loss.pkl', 'wb') as f:
    pickle.dump(policy_loss, f)
with open('all_layers_total_loss.pkl', 'wb') as f:
    pickle.dump(total_loss, f)
with open('all_layers_vf_loss.pkl', 'wb') as f:
    pickle.dump(vf_loss, f)

with open('all_layers_mean.pkl', 'wb') as f:
    pickle.dump(mean_episode_rewards, f)
with open('all_layers_max.pkl', 'wb') as f:
    pickle.dump(max_episode_rewards, f)
with open('all_layers_min.pkl', 'wb') as f:
    pickle.dump(min_episode_rewards, f)
