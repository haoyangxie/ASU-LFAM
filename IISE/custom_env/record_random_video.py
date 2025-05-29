import gymnasium as gym
import pygame.time
import moviepy.editor as mpy
import numpy as np
import register_env

env = gym.make('ExtruderControl-v1')
frames = []

for i in range(1):
    state, _ = env.reset()
    terminated, truncated = False, False
    step = 0
    while not terminated and not truncated:
        step += 1
        action = env.action_space.sample()
        state, reward, terminated, truncated, info = env.step(action)
        env.render()
        frame = pygame.surfarray.array3d(pygame.display.get_surface())
        frame = np.transpose(frame, (1, 0, 2))  # Transpose to match MoviePy's expected format
        frames.append(frame)
        pygame.time.wait(1)
    print(f"Episode {i} reward: {env.cumulative_reward}")
    print(f"Episode {i} step: {step}")

clip = mpy.ImageSequenceClip(frames, fps=5)  # Adjust fps as needed
clip.write_videofile("random_simulation.mp4", codec="libx264")
