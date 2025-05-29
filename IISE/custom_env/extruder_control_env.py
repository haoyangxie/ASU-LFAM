from typing import Optional, Tuple
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import pickle
import pygame
from custom_env import utils
from utils import *
import warnings
from generate_profiles import generate_profiles
warnings.filterwarnings("ignore")

# unit is mm/s
MAX_SPEED = 100
MIN_SPEED = 5
# unit is mm/s^2
MAX_ACCELERATION = 136
MIN_ACCELERATION = -136
TEMPERATURE_UPPER_BOUND = 140
IDEAL_TEMPERATURE = 110
TEMPERATURE_LOWER_BOUND = 80


class ExtruderControlEnv(gym.Env):
    '''
    ### Description

    This environment corresponds to the extruder control problem in LFAM descried by Haoyang Xie.
    The printing tool path is decided by slic3r, so it is deterministic. All we need to do is control the extruder's speed
    to make sure the temperature of every checkpoint is as close as ideal temperature possible when extruder passes.

    ### Action Space
    We are planning to use a 1-dimensional continuous space as the action space, the speed change at that time step.
    The range of speed change is from -5% to 5%. We will clip the speed if it is over maximum speed (100 mm/s) or
    under minimum speed (5 mm/s).

    ### Observation Space
    We use discrete space as Observation Space. Typically, 100 selected checkpoints on the printing tool path. The distance
    between two checkpoints is 35mm now. Each sate is a 6-dimensional vector: current checkpoint, current speed and
    the difference between ideal temperature in this checkpoint, the profile of next checkpoint and all checkpoint temperature profile
    that each is a 1x120 vector.

    ### Rewards
    After every step a reward is granted. The total reward of an episode is the
    sum of the rewards for all the steps within that episode.

    For each step, the reward:
        - is decreased by 0.1 points if the extruder needs to change the speed.
        - is decreased by 0.1 * the time it takes between two checkpoints.
        - is increased 2 -  (|difference from ideal temperature| / half temperature interval (upper bound - lower bound))^2

    ### Episode End
    The episode finishes if:
    1) Pass all the check points, done

    **Note**: We decide to use 30 mm/s as the first layer speed, the first layer time would be 117s (59 frame).
    Now, we only consider the layer after the first layer.
    We clip the speed to be in the range of 10 to 360 mm/s.
    '''

    def __init__(self, env_config=None):
        super(ExtruderControlEnv, self).__init__()
        self.env_config = env_config if env_config is not None else {}
        # action space is from -1 to 1, step is 0.1
        self.action_space = spaces.Discrete(21)
        self._action_to_speed = np.linspace(-1, 1, 21)
        self.current_reward = 0
        self.cumulative_reward = 0
        self.observation_space = spaces.Dict({
            "current_checkpoint": spaces.Discrete(100),
            "current_speed": spaces.Box(low=0, high=1, shape=(), dtype=np.float32),
            "current_acceleration": spaces.Discrete(21),
            "temperature_diff": spaces.Box(low=-np.inf, high=np.inf, shape=(), dtype=np.float32),
            "next_profile": spaces.Box(low=-np.inf, high=np.inf, shape=(120,), dtype=np.float32),
            "all_profiles": spaces.Box(low=-np.inf, high=np.inf, shape=(100, 120), dtype=np.float32)
        })
        # Initialize state
        self.state = None
        self.checkpoints = 100
        self.current_checkpoint = 0
        self.time_spent = []
        self.profile_list = None
        # with open('profile_dict.pkl', 'rb') as f:
        #     profile_list = pickle.load(f)
        # self.profile_list = profile_list
        # assert len(self.profile_list) == 100, "wrong number of profiles"

    def get_acceleration(self, action):
        return self._action_to_speed[action]

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        self.current_checkpoint = 0
        self.time_spent = []
        self.current_reward = 0
        self.cumulative_reward = 0
        self.profile_list = generate_profiles()
        assert len(self.profile_list) == 100, "wrong number of profiles"
        self.state = {
            "current_checkpoint": 0,
            "current_speed": utils.get_normalized_speed(MIN_SPEED),
            "current_acceleration": 0.0,
            "temperature_diff": 0.0,  # Initialize to some value
            "next_profile": self.profile_list[0],  # Initialize to first checkpoint profile
            "all_profiles": self.profile_list
        }
        obs = self.state
        info = {}
        return obs, info

    def step(self, action):
        reward = 0
        terminated = False
        truncated = False
        # change speed will decrease reward
        if action != self.state["current_acceleration"]:
            reward -= 0.1
        current_original_acceleration = utils.get_original_acceleration(self.get_acceleration(action))
        current_original_speed = utils.get_original_speed(self.state["current_speed"])

        # get next state
        t = -1
        temp_difference = -1
        if current_original_acceleration == 0.0:
            t = 35 / current_original_speed
            self.time_spent.append(t)
            index_in_profile = int((99 - self.state["current_checkpoint"]) * 0.59) + int(sum(self.time_spent) / 2)
            if index_in_profile >= 120:
                temp_difference = np.abs(self.state["next_profile"][-1] - IDEAL_TEMPERATURE)
            else:
                temp_difference = np.abs(self.state["next_profile"][index_in_profile] - IDEAL_TEMPERATURE)
        else:
            u = current_original_speed
            a = current_original_acceleration
            s = 35
            t = utils.solve_for_t(u, a, s)
            self.time_spent.append(t)
            index_in_profile = int((99 - self.state["current_checkpoint"]) * 0.59) + int(sum(self.time_spent) / 2)
            if index_in_profile >= 120:
                temp_difference = np.abs(self.state["next_profile"][-1] - IDEAL_TEMPERATURE)
            else:
                temp_difference = np.abs(self.state["next_profile"][index_in_profile] - IDEAL_TEMPERATURE)
        reward -= 0.1 * t
        reward += 2 - (temp_difference / ((TEMPERATURE_UPPER_BOUND - TEMPERATURE_LOWER_BOUND) / 2)) ** 2

        clipped_spped = np.clip(current_original_speed + t * current_original_acceleration, MIN_SPEED, MAX_SPEED)
        next_normalized_speed = utils.get_normalized_speed(clipped_spped)
        next_normalized_acceleration = self.get_acceleration(action)

        self.current_checkpoint += 1
        terminated = self.current_checkpoint >= self.checkpoints
        next_temp_profile = self.profile_list[self.current_checkpoint] if not terminated else self.profile_list[0]

        self.state = {
            "current_checkpoint": self.current_checkpoint,
            "current_speed": next_normalized_speed,
            "current_acceleration": next_normalized_acceleration,
            "temperature_diff": temp_difference,
            "next_profile": next_temp_profile,
            "all_profiles": self.profile_list
        }
        obs = self.state
        self.current_reward = reward
        self.cumulative_reward += reward
        return obs, reward, terminated, truncated, {}

    def render(self, screen_width=800, screen_height=400, visible_checkpoints=10):
        if not pygame.get_init():
            pygame.init()
        screen = pygame.display.set_mode((screen_width, screen_height))
        pygame.display.set_caption("Extruder Control Visualization")

        # Load Mario image
        mario_image = pygame.image.load("mario.png")
        mario_image = pygame.transform.scale(mario_image, (30, 30))

        # Colors
        BLACK = (0, 0, 0)
        WHITE = (255, 255, 255)
        BLUE = (0, 0, 255)

        # Fill the screen
        screen.fill(WHITE)

        # Make sure checkpoints include the final point (101 checkpoints for 100 segments)
        total_checkpoints = self.checkpoints + 1

        # Calculate the range of checkpoints to display
        start_index = max(0, self.current_checkpoint - visible_checkpoints // 2)
        end_index = min(total_checkpoints, start_index + visible_checkpoints)

        # Ensure that the last checkpoint is included
        if end_index - start_index < visible_checkpoints:
            start_index = max(0, end_index - visible_checkpoints)

        # Draw the path as a line (only for visible checkpoints)
        checkpoints_y = screen_height // 1.5
        path_start = (screen_width // 10, checkpoints_y)
        path_end = (screen_width * 9 // 10, checkpoints_y)
        pygame.draw.line(screen, BLACK, path_start, path_end, 2)

        # Calculate spacing for the visible checkpoints
        num_visible_checkpoints = end_index - start_index
        checkpoint_spacing = (screen_width * 8 // 10) // (num_visible_checkpoints - 1)
        font = pygame.font.SysFont(None, 24)

        # Draw each visible checkpoint as a small circle
        for i in range(num_visible_checkpoints):
            x_pos = screen_width // 10 + i * checkpoint_spacing
            pygame.draw.circle(screen, BLUE, (x_pos, checkpoints_y), 5)
            index_text = font.render(f"{start_index + i}", True, BLACK)
            screen.blit(index_text, (x_pos - 10, checkpoints_y + 20))

        # Draw the Mario image at the current extruder position
        extruder_relative_index = self.current_checkpoint - start_index
        extruder_x = screen_width // 10 + extruder_relative_index * checkpoint_spacing
        screen.blit(mario_image, (extruder_x - 15, checkpoints_y - 40))

        # Display additional information
        speed_text = font.render(f"Speed: {utils.get_original_speed(self.state['current_speed']):.2f} mm/s", True,
                                 BLACK)
        acceleration_text = font.render(
            f"Acceleration: {utils.get_original_acceleration(self.state['current_acceleration']):.2f} mm/s²", True,
            BLACK)
        temp_diff_text = font.render(f"Temperature Difference: {self.state['temperature_diff']:.2f}°C", True, BLACK)
        current_reward_text = font.render(f"Current Reward: {self.current_reward:.2f}", True, BLACK)
        cumulative_reward_text = font.render(f"Cumulative Reward: {self.cumulative_reward:.2f}", True, BLACK)

        screen.blit(speed_text, (10, 10))
        screen.blit(acceleration_text, (10, 40))
        screen.blit(temp_diff_text, (10, 70))
        screen.blit(current_reward_text, (10, 100))
        screen.blit(cumulative_reward_text, (10, 130))

        # Update the display
        pygame.display.flip()

        # Handle window close events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()

    def close(self):
        pass
