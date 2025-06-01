# Video-Diffusion-based-Digital-Twin
This code is about the paper "Video Diffusion based Digital Twin for Large Format Additive Manufacturing", which was accepted and is going to be published on the IEEE CASE 2025 Conference. We propose an adaptive Digital Twin framework based on the Video Diffusion Transformer (VDT). Our approach leverages Generative AI to dynamically simulate future temperature distributions when layer time or other printing parameters change.

Paper: **`IEEE_CASE_2025.pdf`**

## Input data
The input format is 256*320 csv files. Model is tested to train on Pytorch using parallel GPU A100 with the IR thermal data for Hexagon geometry with different layer time. 

## Model training and inference
1. **Train Model**: Provide root path of data files in **`model_training.py`**. Under the root path, we have two folders "hex3min" and "hex4min" with all .csv files. Adjust parameters and run **`model_training.py`**: 
  ````python
# nproc_per_node: number of GPU on one node
torchrun --nproc_per_node=2 model_training.py 
  ````
2. **Inference Prediction**: Run **`model_inference.py`** for temperature prediction.

## Code files
- **`attention.py`** - Define the cross attention function. 
- **`data_tensor_import.py`** - Process .csv data files into format: [[sequence images], class label] with image size 256x256. 
- **`mask_generator.py`** - Split sequence of images into past frames and future frames. Mask the sequence of future frames for prediction reason. Please determine number of frames to predict in this script (default = 8).
- **`models.py`** - Transformer based diffusion model architecture.
- **`utils.py`** - Define helper functions, like learning rate and model loading. 
- **`model_training.py`** - Main training script containing the diffusion ransformer model training pipeline, data loading, loss computation, and model saving. Pretrained state-of-art encoder model is applied. 
- **`model_inference.py`** - Inference script for temperature prediction.
- Folder **`diffusion`** - Define the algorithm of diffusion model and loss function. 


## Diffusion Transformer Architecture
<img width="634" alt="architecture v2" src="https://github.com/user-attachments/assets/1679e7df-9583-4dbb-b46e-2b6a776c469f" />
