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
INITIAL_SPEED = 20
# unit is mm/s^2
MAX_ACCELERATION = 136
MIN_ACCELERATION = -136
TEMPERATURE_UPPER_BOUND = 140
IDEAL_TEMPERATURE = 110
TEMPERATURE_LOWER_BOUND = 80


class ExtruderControlEnvMultipleLayer(gym.Env):
    '''
    ### Description
    This is v3 version, in this version, we consider all the layers of the printed part.
    '''

    def __init__(self, env_config=None):
        super(ExtruderControlEnvMultipleLayer, self).__init__()
        self.env_config = env_config if env_config is not None else {}
        # action space is from -0.05 to 0.05, step is 0.01
        self.action_space = spaces.Discrete(11)
        self._action_to_speed = np.linspace(-0.05, 0.05, 11)
        self.current_reward = 0
        self.cumulative_reward = 0
        # current index of checkpoint, current speed, temp difference
        self.observation_space = spaces.Box(
            low=np.array([0, 0, -np.inf]),
            high=np.array([53, 1, np.inf]),
            dtype=np.float32
        )
        # Initialize state
        self.state = None
        self.num_layers = 20
        self.checkpoints_per_layer = 54
        self.current_checkpoint = 0
        self.current_layer = 0
        self.time_spent = []
        self.profile_list = None
        self.speed_history = []
        with open('true_data.pkl', 'rb') as f:
            profile_list = pickle.load(f)
        self.profile_list = profile_list
        # assert len(self.profile_list) == 100, "wrong number of profiles"

    def get_action(self, action):
        return self._action_to_speed[action]

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        self.current_layer = 0
        self.current_checkpoint = 0
        self.time_spent = [[] for _ in range(self.num_layers)]
        self.current_reward = 0
        self.cumulative_reward = 0
        self.speed_history = [[] for _ in range(self.num_layers)]
        self.speed_history[self.current_layer].append(INITIAL_SPEED)
        self.state = {
            "current_layer": 0,
            "current_checkpoint": 0,
            "current_speed": utils.get_normalized_speed(INITIAL_SPEED),
            "temperature_diff": 0.0,  # Initialize to some value
            "next_profile": self.profile_list[0],  # Initialize to first checkpoint profile
            "all_profiles": self.profile_list
        }
        obs = np.array([
            self.state["current_layer"],
            self.state["current_speed"],
            self.state["temperature_diff"]
        ], dtype=np.float32)
        info = {}
        return obs, info

    def step(self, action):
        action = self.get_action(action)
        reward = 0
        terminated = False
        truncated = False

        current_original_speed = utils.get_original_speed(self.state["current_speed"]) * (1 + action)
        clipped_speed = np.clip(current_original_speed, MIN_SPEED, MAX_SPEED)
        next_normalized_speed = utils.get_normalized_speed(clipped_speed)
        self.speed_history[self.current_layer].append(clipped_speed)

        t = 66 / clipped_speed
        self.time_spent[self.current_layer].append(t)
        if self.current_layer == 0:
            index_in_profile = (int((53 - self.state["current_checkpoint"]) * (66.0/INITIAL_SPEED))
                                + int(sum(self.time_spent[self.current_layer])))
            # assert 99 - self.state["current_checkpoint"] + len(self.time_spent[self.current_layer]) == 100, "wrong index in profile when current layer = 0"
        else:
            index_in_profile = (int(sum(self.time_spent[self.current_layer-1][self.current_checkpoint+1:]))
                                + int(sum(self.time_spent[self.current_layer])))
            # assert len(self.time_spent[self.current_layer-1][self.current_checkpoint+1:]) + len(self.time_spent[self.current_layer]) == 100, "wrong index in profile when current layer > 0"

        if index_in_profile >= 200:
            temp_difference = np.abs(self.state["next_profile"][-1] - IDEAL_TEMPERATURE)
        else:
            temp_difference = np.abs(self.state["next_profile"][index_in_profile] - IDEAL_TEMPERATURE)

        reward += 5 - self.softplus(temp_difference)

        self.current_checkpoint += 1
        self.current_checkpoint %= self.checkpoints_per_layer
        if self.current_checkpoint == 0:
            self.current_layer += 1

        terminated = self.current_layer >= self.num_layers
        next_temp_profile = self.profile_list[self.current_checkpoint]

        self.state = {
            "current_layer": self.current_layer,
            "current_checkpoint": self.current_checkpoint,
            "current_speed": next_normalized_speed,
            "temperature_diff": temp_difference,
            "next_profile": next_temp_profile,
            "all_profiles": self.profile_list
        }
        obs = np.array([
            self.state["current_checkpoint"],
            self.state["current_speed"],
            self.state["temperature_diff"]
        ], dtype=np.float32)
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
        total_checkpoints = self.checkpoints_per_layer + 1

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

    @staticmethod
    def softplus(x):
        return np.log(1 + np.exp(x))

    def close(self):
        pass
