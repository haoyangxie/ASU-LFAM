# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
A minimal training script for VDT using PyTorch DDP.
"""
import os
import torch
# the first flag below was False when we tested this script but True makes A100 training a lot faster:
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torchvision.datasets import ImageFolder
from torchvision import transforms
import numpy as np
from collections import OrderedDict
from PIL import Image
from copy import deepcopy
from glob import glob
from time import time
import argparse
import logging
import os
import torch.nn.functional as F
from torchvision.utils import save_image
import pandas as pd
import matplotlib.pyplot as plt
import torch.optim.lr_scheduler as lr_scheduler

from models import VDT_models
from diffusion import create_diffusion
from diffusers.models import AutoencoderKL
from data_tensor_import import CSVImageDatasetSequence_Hexagon

from mask_generator import VideoMaskGenerator
from utils import load_checkpoint, init_distributed_mode


print('pytorch_version: ', torch.__version__)
print(f"CUDA version: {torch.version.cuda}")
print('cuda availability: ', torch.cuda.is_available())
print('num_cuda device', torch.cuda.device_count())

import socket



#################################################################################
#                             Training Helper Functions                         #
#################################################################################

# @torch.no_grad()
def update_ema(ema_model, model, decay=0.9999):
    """
    Step the EMA model towards the current model.
    """
    ema_params = OrderedDict(ema_model.named_parameters())
    model_params = OrderedDict(model.named_parameters())

    for name, param in model_params.items():
        # TODO: Consider applying only to params that require_grad to avoid small numerical changes of pos_embed
        ema_params[name].mul_(decay).add_(param.data, alpha=1 - decay)


def requires_grad(model, flag=True):
    """
    Set requires_grad flag for all parameters in a model.
    """
    for p in model.parameters():
        p.requires_grad = flag


def cleanup():
    """
    End DDP training.
    """
    dist.destroy_process_group()



#################################################################################
#                                  Training Loop                                #
#################################################################################
def main(args):
    assert torch.cuda.is_available(), "Training currently requires at least one GPU."
    
    device = torch.device(args.device)
    init_distributed_mode(args)
    
    # # Setup DDP:
    # dist.init_process_group("nccl")
    assert args.batch_size % dist.get_world_size() == 0, f"Batch size must be divisible by world size."
    rank = dist.get_rank()
    device = rank % torch.cuda.device_count()
    seed = args.seed * dist.get_world_size() + rank
    torch.manual_seed(seed)
    torch.cuda.set_device(device)
    print(f"Starting rank={rank}, seed={seed}, world_size={dist.get_world_size()}.")

    
    # Create model:
    assert args.image_size % 8 == 0, "Image size must be divisible by 8 (for the VAE encoder). Assume default 256*256"
    latent_size = args.image_size // 8
    additional_kwargs = {'num_frames': args.num_frames,
    'mode': 'video'} 
    model = VDT_models[args.model](
        input_size=latent_size,
        num_classes=args.num_classes,
        **additional_kwargs
    ).to(device)
    
    # Note that parameter initialization is done within the VDT constructor
    ema = deepcopy(model).to(device)  # Create an EMA of the model for use after training
    requires_grad(ema, False)
    
    # assign model to DDP
    rank = dist.get_rank()
    model = DDP(model.to(device), device_ids=[rank])
    # diffusion = create_diffusion(timestep_respacing="")  # default: 1000 steps, linear noise schedule
    diffusion = create_diffusion(str(args.num_sampling_steps), training=True) 
    vae = AutoencoderKL.from_pretrained(f"stabilityai/sd-vae-ft-{args.vae}").to(device)
    
    # Setup optimizer (we used default Adam betas=(0.9, 0.999):
    opt = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=0)
    
    
    # Setup data:
    data_path = '/scratch/lliu112/CNN_LSTM/data' # root path of hexagon data folder "hex3min" and "hex4min"
    
    transform = transforms.Compose([
        transforms.ToTensor(), # scale to range [0,1]
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True) # normalization to range [-1,1], gray-scale one channel; otherwise, mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5] for three channels
    ])
    
    
    # read csv sequential images and split into sequences: (num_frames, channel, height, width), label 
    dataset = CSVImageDatasetSequence_Hexagon(root_dir=data_path, 
                                num_training = args.num_training, 
                                first_frame = args.first_frame, 
                                frame_sequence = args.num_frames,
                                transform=transform
                                )
    
    # Apply DistributedSampler for proper sharding of data across GPUs
    sampler = DistributedSampler(
            dataset,
            num_replicas=dist.get_world_size(),
            rank=rank,
            shuffle=False,
            seed=args.seed
        )
    
    # split sequences and corresponding labels into batches: [(batch_size, num_frames, channel, height, width), (batch_size, )]
    loader = DataLoader(
        dataset,
        batch_size=int(args.batch_size // dist.get_world_size()),
        shuffle=False,
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True
    )
    
    # for batch_idx, (sequences, labels) in enumerate(loader):
    #     print(f"Batch {batch_idx}: Sequences Shape: {sequences.shape}, Labels: {labels}")
    
    # Prepare models for training:
    update_ema(ema, model.module, decay=0)  # Ensure EMA is initialized with synced weights
    model.train()  # important! This enables embedding dropout for classifier-free guidance
    ema.eval()  # EMA model should always be in eval mode
    
    
    # Variables for monitoring/logging purposes:
    train_steps = 0
    log_steps = 0
    running_loss = 0
    start_time = time()
    checkpoint_dir = args.ckpt_path
    
    # Define warmup parameters
    warmup_epochs = 5
    warmup_lr = 1e-5  # Warmup starting LR
    # Define CosineAnnealingLR
    cosine_scheduler = lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs, eta_min=args.min_lr)


    training_losssave = []
    print("Start training")
    for epoch in range(args.epochs):
        sampler.set_epoch(epoch)
        
        # set decay learning rate
        if epoch < warmup_epochs:
            # Linear warmup
            lr = warmup_lr + (args.learning_rate - warmup_lr) * epoch / warmup_epochs
            for param_group in opt.param_groups:
                param_group['lr'] = lr
        else:
            # Cosine schedule
            cosine_scheduler.step()
            
            
        for x, y in loader:
            x = x.to(device) # sequence images
            y = y.to(device) # label
            B, T, C, W, H = x.shape # input as video sequences, (8,16,3,256,256)
            x = x.contiguous().view(-1, C, W, H) # (B*T, C, W, H) --> (N, C, W, H)
            
            # Encoding input image sequences
            with torch.no_grad():
                # Map input images to latent space + normalize latents: input shape (N, C, W, H), output shape (N, 4, 32, 32) for the pretrained VAE
                x = vae.encode(x).latent_dist.sample().mul_(0.18215)
                # print(x.shape)
    
            N, C, W, H = x.shape
            x = x.view(B, T, C, H, W) # reshape back to video sequences, (8,16,4,32,32)
            t = torch.randint(0, diffusion.num_timesteps, (x.shape[0],), device=device) # get diffusion timestep; shape should be (B,)

            
            
            # generate mask to seperate past frames and future frames, eg: default past 8, future 8
            generator = VideoMaskGenerator((x.shape[-4], x.shape[-2], x.shape[-1]))
            choice_idx = 0 # for video prediction
            mask = generator(B, device, idx=choice_idx) # (B,T,H,W): mask value = 0 for all past condition frames, mask value = 1 for all future prediction frames
            # print(mask.shape)
    
            
            # fit diffusion model
            model_kwargs = dict(y=y)
            loss_dict = diffusion.training_losses(model, x, t, model_kwargs, mask = mask) # input shape: (B,T,C,H,W), H=W
            loss = loss_dict["loss"].mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            update_ema(ema, model.module)
    
            # Log loss values:
            running_loss += loss.item()
            log_steps += 1
            train_steps += 1
            if train_steps % args.log_every == 0:
                # print(train_steps)
                # Measure training speed:
                torch.cuda.synchronize()
                end_time = time()
                steps_per_sec = log_steps / (end_time - start_time)
                
                # Reduce loss history over all processes:
                avg_loss = torch.tensor(running_loss / log_steps, device=device)
                dist.all_reduce(avg_loss, op=dist.ReduceOp.SUM)
                avg_loss = avg_loss.item() / dist.get_world_size()
                print(f"Epoch [{epoch+1}/{args.epochs}], Step [{train_steps:07d}], Train Loss: {avg_loss:.4f}, Train Steps/Sec: {steps_per_sec:.2f}, Learning Rate: {opt.param_groups[0]['lr']}")
                training_losssave.append(avg_loss)
                
                # Reset monitoring variables:
                running_loss = 0
                log_steps = 0
                start_time = time()
                
        
            # Save VDT checkpoint:
            if train_steps % args.ckpt_every == 0 and train_steps > 0:
                if rank == 0:
                    checkpoint = {
                        "model": model.module.state_dict(),
                        "ema": ema.state_dict(),
                        # "opt": opt.state_dict(),
                        "pos_embed": model.module.pos_embed
                        # "args": args
                    }
                    checkpoint_path = f"{checkpoint_dir}/{train_steps:07d}.pt"
                    torch.save(checkpoint, checkpoint_path)
                    # logger.info(f"Saved checkpoint to {checkpoint_path}")
                dist.barrier()
    
    
    print("Training complete")
    
    # save entir model
    dist.barrier()
    if rank == 0:
        torch.save(model.state_dict(), "/scratch/lliu112/hexagon/model_batch8/entire_model_batch8.pt")
    
    # save training loss
    save_loss = pd.DataFrame(training_losssave)
    save_loss.to_csv('/scratch/lliu112/hexagon/model_batch8/training_loss.csv', index=False)
    
    cleanup()




if __name__ == "__main__":
    # os.environ['MASTER_ADDR'] = '127.0.0.1'
    # os.environ['MASTER_PORT'] = '29500'
    os.environ['OPENBLAS_NUM_THREADS'] = '1'
    
    
    # Default args here will train DiT-XL/2 with the hyperparameters we used in our paper (except training iters).
    parser = argparse.ArgumentParser()
    # parser.add_argument("--data-path", type=str, required=True)
    parser.add_argument("--results-dir", type=str, default="results")
    
    parser.add_argument("--model", type=str, choices=list(VDT_models.keys()), default="VDT-S/2")
    parser.add_argument("--vae", type=str, choices=["ema", "mse"], default="ema")
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--f", type=str, default=None)
    
    parser.add_argument("--num-classes", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--learning_rate", type=int, default=1e-4)
    parser.add_argument("--min_lr", type=int, default=1e-6)
    parser.add_argument("--cfg-scale", type=float, default=4.0)
    
    parser.add_argument("--num-sampling-steps", type=int, default=500) # Set higher for better results! (max 1000)
    
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num_frames", type=int, default=16)
    parser.add_argument("--num_training", type=int, default=3000)
    parser.add_argument("--num_testing", type=int, default=200)
    parser.add_argument("--first_frame", type=int, default=0)
    
    parser.add_argument('--device', default='cuda')
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--dist_url", default = 'env://')
    
    parser.add_argument("--epochs", type=int, default=500) # default=1400
    parser.add_argument("--log-every", type=int, default=100)  
    parser.add_argument("--ckpt-every", type=int, default=30000) # default = 50_000
    parser.add_argument('--ckpt_path', default='/scratch/lliu112/hexagon/model_batch8')
    args = parser.parse_args()
    
    main(args)




