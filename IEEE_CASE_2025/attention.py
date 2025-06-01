import torch
from torch import nn
from torch.nn import functional as F
import math


class CrossAttention(nn.Module):
    def __init__(self, d_embed, d_cross, num_heads,  qkv_bias=False, proj_bias=True):
        super().__init__()
        # assert d_embed % num_heads == 0, 'dim should be divisible by num_heads'
        self.q_proj   = nn.Linear(d_embed, d_embed, bias=qkv_bias)
        self.k_proj   = nn.Linear(d_cross, d_embed, bias=qkv_bias)
        self.v_proj   = nn.Linear(d_cross, d_embed, bias=qkv_bias)
        self.out_proj = nn.Linear(d_embed, d_embed, bias=proj_bias)
        self.num_heads = num_heads
        self.d_head = d_embed // num_heads
        self.scale = self.d_head ** -0.5
 
    
    def forward(self, x: torch.Tensor, y: torch.Tensor):
        # x (latent): # (B, N, C) - batch, sequence length, d_embed
        # y (embedding_label): # (B, Seq_Len_KV, Dim_KV), Dim_KV = d_cross
        # default input: d_cross = d_embed

        B, N, C = x.shape
        # Divide each embedding of Q into multiple heads such that d_heads * num_heads = Dim_Q
        interim_shape = (B, -1, self.num_heads, self.d_head)
        
        # Project queries from image features
        q = self.q_proj(x) # (B,N,C) -> (B,N,C)
        # Project keys and values from class label
        k = self.k_proj(y) # (B, Seq_Len_KV, Dim_KV) -> (B, Seq_Len_KV, C)
        v = self.v_proj(y) # (B, Seq_Len_KV, Dim_KV) -> (B, Seq_Len_KV, C)
        
        q = q.view(interim_shape).transpose(1, 2) # (B,N,C) -> (B,N, H, C / H) -> (B, H, N, C / H)
        k = k.view(interim_shape).transpose(1, 2) # (B, Seq_Len_KV, C) -> (B, Seq_Len_KV, H, C / H) -> (B, H, Seq_Len_KV, C / H)
        v = v.view(interim_shape).transpose(1, 2) # (B, Seq_Len_KV, C) -> (B, Seq_Len_KV, H, C / H) -> (B, H, Seq_Len_KV, C / H)
        
        # Compute attention scores
        weight = q @ k.transpose(-1, -2) # (B, H, N, C / H) @ (B, H, C / H, Seq_Len_KV) -> (B, H, N, Seq_Len_KV)
        weight *= self.scale # (B, H, N, Seq_Len_KV)
        weight = F.softmax(weight, dim=-1) # (B, H, N, Seq_Len_KV)
        
        # Compute attention output
        output = weight @ v # (B, H, N, Seq_Len_KV) @ (B, H, Seq_Len_KV, C / H) -> (Batch_Size, H, N, C / H)
        output = output.transpose(1, 2).reshape(B, N, C) # (B, H, N, C / H) -> (B, N, H, C / H) -> (B, N, C)
        output = self.out_proj(output) # (B, N, C) -> (B, N, C)

        return output
        
 