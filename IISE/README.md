# Deep Reinforcement Learning for Dynamic Extruder Speed Control in Large Format Additive Manufacturing

This repository contains the complete implementation of a deep reinforcement learning system for controlling extruder speed in LFAM. The system uses Enhanced Environment Aware Proximal Policy Optimization (EAPPO) to dynamically adjust extruder speed based on spatiotemporal temperature predictions.

## Repository Structure

### Core Directories

#### `custom_env/` - Reinforcement Learning Environment
Contains the custom OpenAI Gymnasium environments that simulate the LFAM printing process:

- **`extruder_control_env.py`** - Main RL environment with discrete action space for acceleration control
- **`extruder_control_env_acceleration.py`** - Environment supporting acceleration constraints and variable partition lengths
- **`extruder_control_env_multiple_layer.py`** - Multi-layer printing environment for complex geometries
- **`extruder_control_env_multiple_layer_case_study.py`** - Complete environment for the hexagon case study (20 layers, 25 partitions)
- **`extruder_control_env_hard_constraint.py`** - Environment with hard temperature constraints
- **`extruder_control_env_fixed_profiles.py`** - Environment using pre-computed temperature profiles
- **`register_env.py`** - Environment registration for Ray RLlib integration
- **`utils.py`** - Utility functions for speed normalization and physics calculations

#### `optimization/` - Benchmarking and Optimization
Mathematical optimization baselines and comparison studies:

- **`dynamic_programming.ipynb`** - Dynamic programming approach for speed optimization
- **`model_with_acceleration.py`** - Mixed Integer Programming (MIP) model with acceleration constraints
- **`model_without_acceleration.py`** - Simplified MIP model without acceleration
- **`model_without_acceleration_discrete.py`** - Discrete version of MIP model
- **`test_model_without_a.ipynb`** - Performance testing and validation

### Training and Inference

#### `custom_env/` Training Files
- **`play_with_rllib.py`** - Main training script using Ray RLlib and PPO algorithm
- **`env_state_mlp.py`** - Custom neural network architecture for state representation
- **`inference.py`** - Model inference and policy evaluation
- **`inference_plot.py`** - Performance visualization and comparison plots
- **`v5_inference_plot.py`** - Specific inference plotting for version 5 models
- **`record_inference_video.py`** - Generate videos of inference results
- **`record_random_video.py`** - Generate videos of random policy baseline

#### `model_weight/` - Trained Models
Stores checkpoints and trained model weights from different experimental configurations.

### Analysis and Visualization

#### `paper_plot/` - Publication Figures
Contains all figures and plots used in the research paper:
- **Training performance curves**: `training_reward.png`, `training_loss.png`
- **Temperature analysis**: `case_study_temp_difference.png`, `clean_profile.png`, `dirty_profile.png`
- **Speed and acceleration patterns**: `acceleration_policy.png`, `simplified_policy.png`
- **Case study results**: `case_study_compare_temp_diff.png`, `case_study_layer_time.png`
- **3D visualizations**: `prediction_3d.png`, `partition.png`
- **Comparative analysis**: `different_layertime.png`, `cycle_pattern.png`

#### `src/` - Analysis Tools
- **`linear_regression.py`** - Statistical analysis of printing parameters
- **`find_distribution.py`** - Distribution analysis of temperature data
- **`test.ipynb`** - Experimental notebooks and data exploration
- **`perspective_image.py`** - Generate perspective view images
- **`perspective_video.py`** - Create perspective view videos
- **`verify_gcode_contour_profile.py`** - Validate G-code against contour profiles

### Data and Configuration

#### `pkl_file/` - Serialized Data
Pickled data files containing:
- Temperature profiles and parameters
- Training results and metrics
- Experimental configurations

#### `input/` & `output/` - Data Pipeline
- **`input/`** - Raw data and configuration files
- **`output/`** - Generated profiles, results, and processed data

#### `cyclic_pattern/` - Pattern Analysis
Analysis of cyclical patterns in temperature behavior during multi-layer printing.

#### `layertime_comparison/` - Performance Metrics
Comparative analysis of layer printing times between different control strategies.

## Usage

### Training a New Model
```bash
cd custom_env/
python play_with_rllib.py
```

### Running Inference
```bash
cd custom_env/
python inference.py
```

### Benchmark Comparison with MIP
```bash
cd optimization/
python model_with_acceleration.py
```

### Creating Visualizations
```bash
cd custom_env/
python inference_plot.py
```