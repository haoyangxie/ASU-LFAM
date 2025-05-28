# Transformer-Based Offline Printing Strategy Design for Large Format Additive Manufacturing

This repository contains the implementation code for the paper "Transformer-Based Offline Printing Strategy Design for Large Format Additive Manufacturing". The project uses Transformer architecture to predict temperature distributions during Large Format Additive Manufacturing (LFAM) processes and optimize printing strategies.

## Project Structure

### Core Code Files

- **`train.py`** - Main training script containing Transformer model training pipeline, data loading, loss computation, and model saving
- **`model.py`** - Transformer model architecture definition including encoder, decoder, attention mechanisms, and positional encoding components
- **`dataset.py`** - Dataset class definition for processing coordinate sequences to temperature distribution mapping, including data preprocessing and mask generation
- **`config.py`** - Configuration file defining model hyperparameters, training parameters, and data paths
- **`inference_3_cases.py`** - Inference script for temperature prediction on three geometries (planters, table, totems)
- **`optimization.py`** - Optimization algorithm implementation for printing time optimization based on predicted temperature distributions

### Data Folders

- **`csv/`** - Contains training data for three geometries (download required - see below)
  - `planters/` - Layer-wise temperature data for planter geometry
  - `table/` - Layer-wise temperature data for table geometry  
  - `totems/` - Layer-wise temperature data for totem geometry
- **`weights/`** - Stores trained model weight files (download required - see below)
- **`Inference_source/`** - Inference result files
  - `planters_inference.txt` - Planter inference results
  - `table_inference.txt` - Table inference results
  - `totems_inference.txt` - Totem inference results

### Validation Image Folders

- **`planters_validate_image/`** - Validation images for planter geometry
- **`table_validate_image/`** - Validation images for table geometry
- **`totems_validate_image/`** - Validation images for totem geometry
- **`paper_figures/`** - Figures used in the paper

### Jupyter Notebooks

- **`inference_3_cases.ipynb`** - Inference analysis and visualization for three case studies
- **`optimization.ipynb`** - Optimization algorithm analysis and results visualization
- **`inference.ipynb`** - General inference analysis
- **`plot_for_paper.ipynb`** - Paper figure generation
- **`regression_result.ipynb`** - Regression analysis results

### Utility Scripts

- **`plot_for_optimization.py`** - Optimization results visualization
- **`plot_position.py`** - Position information visualization
- **`myjob_io-4.sh`** - Cluster job submission script

### Configuration and Data Files

- **`tokenizer.json`** - Tokenizer configuration file for coordinate sequences
- **`planter_cycle.csv`** - Planter cycle data
- **`.gitignore`** - Git ignore file configuration
- **`runs/`** - TensorBoard training logs

## Data Download

### CSV Data
Download the CSV data from Google Drive and extract it to the ASME folder:
```
https://drive.google.com/file/d/16OnP7tH07d9bI1EuD3SwkWtYMMTE5_ip/view?usp=drive_link
```
After downloading, extract the `csv.zip` file directly in the ASME directory.

### Model Weights
Download the pre-trained model weights from Google Drive:
```
https://drive.google.com/file/d/16o6z43KM-ozpM2XCYXzsXZmaq6Pdpu8B/view?usp=drive_link
```
After downloading, extract the `weights.zip` file directly in the ASME directory.

## Key Features

1. **Temperature Prediction**: Uses Transformer model to predict temperature distributions based on printing path coordinates
2. **Printing Optimization**: Optimizes inter-layer timing based on temperature prediction results to improve print quality
3. **Multi-Geometry Support**: Supports three different complexity geometries: planters, table, and totems
4. **Visualization Analysis**: Provides rich visualization tools for analyzing prediction results and optimization effects

## Usage

1. **Train Model**: Run `python train.py` to start training
2. **Inference Prediction**: Run `python inference_3_cases.py` for temperature prediction
3. **Optimization Analysis**: Run `python optimization.py` for printing time optimization
4. **Results Visualization**: Use Jupyter notebooks for detailed analysis

