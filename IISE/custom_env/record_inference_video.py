import gymnasium as gym
import pygame.time
import moviepy.editor as mpy
import numpy as np
import register_env
import pathlib
import torch
from ray.rllib.core.rl_module import RLModule


env = gym.make('ExtruderControl-v1')
rl_module = RLModule.from_checkpoint(
    pathlib.Path("/var/folders/dy/k0313dz972q9cgnsmg_fypqm0000gn/T/tmpd3i2564u") / "learner" / "module_state"/ "default_policy"
)
frames = []

for i in range(1):
    obs, info = env.reset()
    terminated, truncated = False, False
    while not terminated and not truncated:
        torch_obs_batch = torch.from_numpy(np.array([obs]))
        action_logits = rl_module.forward_inference({"obs": torch_obs_batch})[
            "action_dist_inputs"
        ]
        action = torch.argmax(action_logits[0]).numpy()
        obs, reward, terminated, truncated, info = env.step(action)
        env.render()
        frame = pygame.surfarray.array3d(pygame.display.get_surface())
        frame = np.transpose(frame, (1, 0, 2))  # Transpose to match MoviePy's expected format
        frames.append(frame)
        pygame.time.wait(1)
    print(f"Episode {i} reward: {env.cumulative_reward}")

clip = mpy.ImageSequenceClip(frames, fps=5)  # Adjust fps as needed
clip.write_videofile("inference_simulation.mp4", codec="libx264")
