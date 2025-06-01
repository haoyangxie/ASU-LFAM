import torch
# the first flag below was False when we tested this script but True makes A100 training a lot faster:
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
import torch.distributed as dist
from torch.utils.data import DataLoader
from torchvision import transforms
import os
import pandas as pd
import re
from torch.utils.data import Dataset
import numpy as np
import cv2 
from matplotlib import pyplot as plt

class CSVImageDatasetSequence_Hexagon(Dataset):
    def __init__(self, root_dir, num_training = 100, first_frame = 0, frame_sequence = 16, transform = None, mix_label = False):
        """
        Args:
            root_dir (str): Parent directory containing 'hex3min' and 'hex4min'.
            num_training (int): Number of training samples to use.
            first_frame (int): Starting frame index.
            frame_sequence (int): Number of frames per sequence.
            transform (callable, optional): Transformations to apply.
            mix_label (bool): Whether to mix labels (swap label assignments) in inference for generation model 
        """
        self.root_dir = root_dir
        self.transform = transform
        self.sequences = []

        if not mix_label:
            # assign labels: folder_3min = 3, folder_4min = 4
            self.class_folders = {
                "hex3min": 0,  # Assign label 0
                "hex4min": 1   # Assign label 1
            }
        else:
            self.class_folders = {
                "hex3min": 1,  
                "hex4min": 0   
            }
            

        def extract_last_number(filename):
            # Use a regular expression to find the last number in the filename
            matches = re.findall(r'\d+', filename)
            return int(matches[-1]) if matches else float('inf') 

        # Process both class folders
        for class_name, label in self.class_folders.items():
            # print(class_name)
            folder_path = os.path.join(root_dir, class_name)
            if not os.path.exists(folder_path):
                continue  # Skip if the folder doesn't exist

            # Get and sort files by frame number
            files = os.listdir(folder_path)
            sorted_files = sorted(files, key=extract_last_number)
            sorted_paths = [os.path.join(folder_path, f) for f in sorted_files]

            # Limit training samples
            if len(sorted_paths) >= num_training:
                sorted_paths = sorted_paths[first_frame: first_frame + num_training]

            # Split into sequences
            if len(sorted_paths) >= frame_sequence:
                for i in range(len(sorted_paths) - frame_sequence + 1):
                    self.sequences.append((sorted_paths[i:i + frame_sequence], label))
                    # if class_name == "hex4min":
                    #     print(self.sequences[-1])        


    # get the number of sequences
    def __len__(self):
        return len(self.sequences)

    # read csv files and convert to tensor, process data with transform
    def __getitem__(self, idx):
        file_paths, label = self.sequences[idx]
        images = []
        for file_path in file_paths:
            # Load CSV and convert to tensor
            image = pd.read_csv(file_path, header=None).values  # Load as NumPy array
            image = image[:, :256] # crop image to 256x256, cut off right side 
            image = torch.tensor(image, dtype=torch.float32)  # Convert to tensor
            image = image.unsqueeze(0)  # Add channel dimension (grayscale)
            image = image.repeat(3, 1, 1)  # Repeat the channel to create a 3-channel image
            image = image.permute(1, 2, 0).numpy().astype(np.uint8) # make the input shape fit for transforms.ToTensor(), which scale values in range of [0,1]

            if self.transform:
                image = self.transform(image)

            images.append(image)

        # Stack the sequence along the first dimension
        sequence_image = torch.stack(images, dim=0)  # Shape: (sequence_length, 3, H, W)

        return sequence_image, label





