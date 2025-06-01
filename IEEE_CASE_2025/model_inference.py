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

from models import VDT_models
from diffusion import create_diffusion
from diffusers.models import AutoencoderKL
from data_tensor_import import CSVImageDatasetSequence_Hexagon

from mask_generator import VideoMaskGenerator
from utils import load_checkpoint, init_distributed_mode


#################################################################################
#                                  Sampling/Inference                                #
#################################################################################
parser = argparse.ArgumentParser()
parser.add_argument("--results-dir", type=str, default="results")

parser.add_argument("--model", type=str, choices=list(VDT_models.keys()), default="VDT-S/2")
parser.add_argument("--vae", type=str, choices=["ema", "mse"], default="ema")
parser.add_argument("--image-size", type=int, default=256)
parser.add_argument("--f", type=str, default=None)

parser.add_argument("--num-classes", type=int, default=1)
parser.add_argument("--batch_size", type=int, default=8)
parser.add_argument("--cfg-scale", type=float, default=4.0)

parser.add_argument("--num-sampling-steps", type=int, default=500) # Set higher for better results! (max 1000)

parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--num_frames", type=int, default=16)
parser.add_argument("--num_training", type=int, default=3000)
parser.add_argument("--num_testing", type=int, default=20)
parser.add_argument("--first_frame", type=int, default=0)


# parser.add_argument("--ckpt", type=str, default="model.pt",
#                     help="Optional path to a VDT checkpoint.")

parser.add_argument('--device', default='cuda')
parser.add_argument("--num-workers", type=int, default=2)
parser.add_argument("--dist_url", default = 'env://')

# parser.add_argument("--epochs", type=int, default=100) # default=1400
# parser.add_argument("--log-every", type=int, default=100)  
# parser.add_argument("--ckpt-every", type=int, default=5000) # default = 50_000
# parser.add_argument('--ckpt_path', default='/home/lliu112/Desktop/digital_twin/thermal_generative/VDT_result')
args = parser.parse_args()



os.environ['MASTER_ADDR'] = '127.0.0.1'
os.environ['MASTER_PORT'] = '29500'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
# os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

assert torch.cuda.is_available(), "Training currently requires at least one GPU."

device = torch.device(args.device)

# # Setup DDP:
init_distributed_mode(args)
assert args.batch_size % dist.get_world_size() == 0, f"Batch size must be divisible by world size."
rank = dist.get_rank()
device = rank % torch.cuda.device_count()
seed = args.seed * dist.get_world_size() + rank
torch.manual_seed(seed)
torch.cuda.set_device(device)
print(f"Starting rank={rank}, seed={seed}, world_size={dist.get_world_size()}.")

# Setup PyTorch:
torch.set_grad_enabled(False) # disable gradient for all parameters in the model; better for inference

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

diffusion = create_diffusion(str(args.num_sampling_steps), training=False) # training = False for inference
vae = AutoencoderKL.from_pretrained(f"stabilityai/sd-vae-ft-{args.vae}").to(device)


# Setup data:
data_path = '/scratch/lliu112/CNN_LSTM/data'

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True) # gray-scale one channel; otherwise, mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5] for three channels
])


########################################### testing data

########################## save frames for temperature compare ######################
import os, shutil
# clear directory files 
directory_list = ['/scratch/lliu112/hexagon/test_y_true',
                    '/scratch/lliu112/hexagon/test_y_pred'
                    ]

for directory in directory_list:
    folder = directory
    for filename in os.listdir(folder):
        file_path = os.path.join(folder, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print('Failed to delete %s. Reason: %s' % (file_path, e))
        

########################## sampling future images ######################
# # Load the saved training model
# state_dict = torch.load("/home/lliu112/Desktop/digital_twin/thermal_generative/VDT_result/entire_model_reduceLR.pt")

# # Remove the "module." prefix (fix the model saving error in DDP)
# new_state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}

# # Load the updated state dict into the non-DDP model
# model.load_state_dict(new_state_dict)
# # model.load_state_dict(torch.load("/home/lliu112/Desktop/digital_twin/thermal_generative/VDT_result/entire_model.pt"), strict=False)
# model.eval()



chekpoint = torch.load("/scratch/lliu112/hexagon/model_batch8/0180000.pt")
model.load_state_dict(chekpoint['model'])
# ema.load_state_dict(checkpoint['ema'])
model.eval()



dataset_testing = CSVImageDatasetSequence_Hexagon(root_dir=data_path, 
                            num_training = args.num_testing, 
                            first_frame = args.num_training, 
                            frame_sequence = args.num_frames,
                            transform=transform,
                            mix_label = True
                            )

loader_testing = DataLoader(
    dataset_testing,
    batch_size=int(args.batch_size // dist.get_world_size()),
    shuffle=False,
    # sampler=sampler,
    # num_workers=args.num_workers,
    pin_memory=True,
    drop_last=True
)


invTrans = transforms.Compose([ transforms.Normalize(mean = [ 0., 0., 0. ],
                                                     std = [ 1/0.5, 1/0.5, 1/0.5 ]),
                                transforms.Normalize(mean = [ -0.5, -0.5, -0.5 ],
                                                     std = [ 1., 1., 1. ]),
                               ])
n_future = int(args.num_frames/2)



for batch_idx, (x_testing, y_testing) in enumerate(loader_testing):
    
    ### save ground truth of prediction frames
    inv_tensor = invTrans(x_testing)*255
    torch.save(inv_tensor.clone(), "/scratch/lliu112/hexagon/test_y_true/y_true_batch" + str(batch_idx) +'.pt')
    del inv_tensor
    torch.cuda.empty_cache()
        
    # load data to gpu
    x_testing = x_testing.to(device) # sequence images
    y_testing = y_testing.to(device) # label
    # print(x_testing[0][0][0])
    B, T, C, W, H = x_testing.shape # input as video sequences, (8,16,3,256,256)
    x_testing = x_testing.contiguous().view(-1, C, W, H) # (B*T, C, W, H) --> (N, C, W, H)
    raw_x = x_testing
    
    # VAE encoder
    with torch.no_grad():
        # Map input images to latent space + normalize latents: input shape (N, C, W, H), output shape (N, 4, 32, 32) for the pretrained VAE
        x_testing = vae.encode(x_testing).latent_dist.sample().mul_(0.18215)
    
    x_testing = x_testing.view(-1, args.num_frames, 4, x_testing.shape[-2], x_testing.shape[-1])
    z = torch.randn(B, args.num_frames, 4, latent_size, latent_size, device=device)
    
    choice_idx = 0 # mask for prediction problem
    generator = VideoMaskGenerator((x_testing.shape[-4], x_testing.shape[-2], x_testing.shape[-1]))
    mask = generator(B, device, idx=choice_idx)

    sample_fn = model.forward
    # Sample images:
    z = z.permute(0, 2, 1, 3, 4)
    model_kwargs = dict(y=y_testing)
    samples = diffusion.p_sample_loop(
        sample_fn, z.shape, z, clip_denoised=False, progress=True, model_kwargs=model_kwargs, device=device,
        raw_x=x_testing, mask=mask
    )
    

    # abc->acb->bac
    # samples = samples.permute(0, 2, 1, 3, 4)
    samples = samples.permute(1, 0, 2, 3, 4) * mask + x_testing.permute(2, 0, 1, 3, 4) * (1-mask)
    samples = samples.permute(1, 2, 0, 3, 4) # 4, 16, 8, 32, 32 -> 16 8 4
    samples = samples.reshape(-1, 4, latent_size, latent_size) / 0.18215
    
    del x_testing, z, y_testing, model_kwargs
    torch.cuda.empty_cache()
    
    # decoder VAE
    decoded_chunks = []
    chunk_size = 256
    num_chunks = (samples.shape[0] + chunk_size - 1) // chunk_size
    for i in range(num_chunks):
        start_idx = i * chunk_size
        end_idx = min((i + 1) * chunk_size, samples.shape[0])
        chunk = samples[start_idx:end_idx]
        with torch.no_grad():
            decoded_chunk = vae.decode(chunk).sample
            decoded_chunks.append(decoded_chunk)
    
    samples = torch.cat(decoded_chunks, dim=0)
    samples = samples.reshape(-1, args.num_frames, samples.shape[-3], samples.shape[-2], samples.shape[-1])
    # print('samples shape', samples.shape)
    
    
    ### save sampling prediction frames
    inv_tensor_pred = invTrans(samples)*255
    torch.save(inv_tensor_pred.clone(), "/scratch/lliu112/hexagon/test_y_pred/y_pred_batch" + str(batch_idx) +'.pt')
    del inv_tensor_pred
    torch.cuda.empty_cache()
    
    mask = F.interpolate(mask.float(), size=(raw_x.shape[-2], raw_x.shape[-1]), mode='nearest')
    mask = mask.unsqueeze(0).repeat(3,1,1,1,1).permute(1, 2, 0, 3, 4) 
    
    raw_x = raw_x.reshape(-1, args.num_frames, raw_x.shape[-3], raw_x.shape[-2], raw_x.shape[-1])
    raw_x = raw_x * (1 - mask)
    
    # raw_x = raw_x[:, :n_future, :,:,:] # past frames 
    # samples = samples[:, n_future:, :,:,:] # predict frames
        
    # combine input sequence and output sequence    
    samples = torch.cat([raw_x, samples], dim=1)
    del raw_x, mask
    torch.cuda.empty_cache()
    # print(samples.shape)


    # print image for view
    save_image(samples.reshape(-1, samples.shape[-3], samples.shape[-2], samples.shape[-1]), "/scratch/lliu112/hexagon/output_img/output_" + str(batch_idx) + ".png", nrow=16,normalize=True, value_range=(-1, 1))
  

print("done")






























