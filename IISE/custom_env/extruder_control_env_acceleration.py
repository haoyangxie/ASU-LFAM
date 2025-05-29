from typing import Optional, Tuple
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import pickle
import pygame
from custom_env import utils
from utils import *
import warnings
import sympy as sp
from generate_profiles import generate_profiles
warnings.filterwarnings("ignore")

# unit is mm/s
MAX_SPEED = 100
MIN_SPEED = 5
INITIAL_SPEED = 20
# unit is mm/s^2
# MAX_ACCELERATION = 136
# MIN_ACCELERATION = -136
ACCELERATION = 50
TEMPERATURE_UPPER_BOUND = 150
IDEAL_TEMPERATURE = 120
TEMPERATURE_LOWER_BOUND = 90
T_env = 24
hashmap = {0: (66, 0), 1: (132, 2), 2: (264, 6), 3: (132, 8), 4: (264, 12), 5: (132, 14), 6: (198, 17), 7: (66, 18), 8: (264, 22),
           9: (198, 25), 10: (66, 26), 11: (330, 31), 12: (66, 32), 13: (132, 34), 14: (66, 35), 15: (66, 36), 16: (132, 38), 17: (66, 39),
           18: (66, 40), 19: (132, 42), 20: (132, 44), 21: (264, 48), 22: (66, 49), 23: (132, 51), 24: (132, 53)}


def get_temperature_from_params(t, b0, b1, T0):
    return T_env - (b0/b1) + (T0 - T_env + (b0/b1))*np.exp(b1*t)


class ExtruderControlEnvAcceleration(gym.Env):
    '''
    In this version, we use actual partitions where each partition has a different length.
    '''
    def __init__(self, env_config=None, constraint='soft'):
        super(ExtruderControlEnvAcceleration, self).__init__()
        self.env_config = env_config if env_config is not None else {}
        self.constraint = constraint
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
        self.checkpoints = 25
        self.current_checkpoint = 0
        partition_length, self.partition_index = hashmap[self.current_checkpoint]
        self.segments = partition_length * 10

        self.time_spent = [[] for _ in range(self.checkpoints)]
        self.temp_diff_history = [[] for _ in range(self.checkpoints)]
        self.v_inside_partition = [[] for _ in range(self.checkpoints)]
        self.speed_history = []
        self.acceleration_distance = []

        with open('../pkl_file/true_data_params.pkl', 'rb') as f:
            true_data_params = pickle.load(f)
        self.true_data_params = true_data_params

    def get_action(self, action):
        return self._action_to_speed[action]

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        self.current_checkpoint = 0
        partition_length, self.partition_index = hashmap[self.current_checkpoint]
        self.segments = partition_length * 10
        self.current_reward = 0
        self.cumulative_reward = 0
        self.time_spent = [[] for _ in range(self.checkpoints)]
        self.temp_diff_history = [[] for _ in range(self.checkpoints)]
        self.v_inside_partition = [[] for _ in range(self.checkpoints)]
        self.speed_history = []
        self.acceleration_distance = []
        self.state = {
            "current_checkpoint": 0,
            "current_speed": utils.get_normalized_speed(INITIAL_SPEED),
            "temperature_diff": 0.0
        }
        obs = np.array([
            self.state["current_checkpoint"],
            self.state["current_speed"],
            self.state["temperature_diff"]
        ], dtype=np.float32)
        info = {}
        return obs, info

    def step(self, action):
        # each action is for one partition
        action = self.get_action(action)
        reward = 0
        terminated = False
        truncated = False
        # determine accelerate or decelerate
        if action < 0:
            accelerate_state = -1
        elif action == 0:
            accelerate_state = 0
        else:
            accelerate_state = 1

        previous_speed = utils.get_original_speed(self.state["current_speed"])
        current_original_speed = previous_speed * (1 + action)
        clipped_speed = np.clip(current_original_speed, MIN_SPEED, MAX_SPEED)
        next_normalized_speed = utils.get_normalized_speed(clipped_speed)
        self.speed_history.append(clipped_speed)

        # deal with whole partition
        if accelerate_state == -1:
            distance = (previous_speed**2 - clipped_speed**2) / (2*ACCELERATION)
        elif accelerate_state == 0:
            distance = 0
        else:
            distance = (clipped_speed**2 - previous_speed**2) / (2*ACCELERATION)
        self.acceleration_distance.append(distance)

        if accelerate_state == 0:
            self.v_inside_partition[self.current_checkpoint] = [clipped_speed] * self.segments
        else:
            v0 = previous_speed
            v_new = 0
            for i in range(self.segments):
                if 0.1*(i+1) <= distance:
                    if accelerate_state == 1:
                        v_new = np.sqrt(v0**2 + 2*ACCELERATION*0.1)
                    elif accelerate_state == -1:
                        v_new = np.sqrt(v0 ** 2 - 2*ACCELERATION*0.1)
                    else:
                        raise ValueError("Invalid accelerate state.")
                else:
                    v_new = clipped_speed
                self.v_inside_partition[self.current_checkpoint].append(v_new)
                v0 = v_new

        for i in range(self.segments):
            if i == 0:
                t = 0.1 / ((previous_speed + self.v_inside_partition[self.current_checkpoint][0]) / 2)
            else:
                t = 0.1 / ((self.v_inside_partition[self.current_checkpoint][i] + self.v_inside_partition[self.current_checkpoint][i-1]) / 2)
            self.time_spent[self.current_checkpoint].append(t)

        time_in_profile = (53 - self.partition_index) * (66.0/INITIAL_SPEED)
        time_in_profile += sum(sum(inner_list) for inner_list in self.time_spent[:self.current_checkpoint])
        # print(sum(sum(inner_list) for inner_list in self.time_spent[:self.current_checkpoint]))
        for i in range(self.segments):
            time_in_profile += self.time_spent[self.current_checkpoint][i]
            temperature_in_profile = get_temperature_from_params(time_in_profile,
                                                                 self.true_data_params[self.partition_index][0],
                                                                 self.true_data_params[self.partition_index][1],
                                                                 self.true_data_params[self.partition_index][2])
            self.temp_diff_history[self.current_checkpoint].append(temperature_in_profile - IDEAL_TEMPERATURE)
            temp_difference = np.abs(temperature_in_profile - IDEAL_TEMPERATURE)
            reward += 5 - 10*(temp_difference / ((TEMPERATURE_UPPER_BOUND - TEMPERATURE_LOWER_BOUND) / 2))
        self.current_checkpoint += 1
        if self.current_checkpoint < self.checkpoints:
            partition_length, self.partition_index = hashmap[self.current_checkpoint]
            self.segments = partition_length * 10
        if self.constraint == 'soft':
            terminated = self.current_checkpoint >= self.checkpoints
        elif self.constraint == 'hard':
            terminated = self.current_checkpoint >= self.checkpoints or temp_difference > (TEMPERATURE_UPPER_BOUND - TEMPERATURE_LOWER_BOUND) / 2

        self.state = {
            "current_checkpoint": self.current_checkpoint,
            "current_speed": next_normalized_speed,
            "temperature_diff": temp_difference,
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