import ray
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.models import ModelCatalog
import torch
import torch.nn as nn


class EnvStateMLP(TorchModelV2, nn.Module):
    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        TorchModelV2.__init__(self, obs_space, action_space, num_outputs, model_config, name)
        nn.Module.__init__(self)
        print(f"obs_space: {obs_space}")
        print(f"action_space: {action_space}")
        print(f"num_outputs: {num_outputs}")
        print(f"model_config: {model_config}")
        print(f"name: {name}")
        # self.main_network = nn.Sequential(
        #     nn.Linear(obs_space.shape[0], 64),
        #     nn.ReLU(),
        #     nn.Linear(64, num_outputs)
        # )
        #
        # self.mlp_network = nn.Sequential(
        #     nn.Linear(54*3, 256),
        #     nn.ReLU(),
        #     nn.Linear(256, 256),
        #     nn.ReLU(),
        #     nn.Linear(256, 256),
        # )

    def forward(self, input_dict, state, seq_lens):
        x = input_dict['obs']
        x = torch.tanh(self.fc1(x))
        x = torch.tanh(self.fc2(x))
        x = self.fc3(x)
        return x, state
