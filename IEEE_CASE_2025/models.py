"""
Originally inspired by impl at https://github.com/facebookresearch/DiT/blob/main/models.py

Modified by Haoyu Lu, for video diffusion transformer
"""
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# GLIDE: https://github.com/openai/glide-text2im
# MAE: https://github.com/facebookresearch/mae/blob/main/models_mae.py
# DiT: https://github.com/facebookresearch/DiT/blob/main/models.py
# 
# --------------------------------------------------------

import torch
import torch.nn as nn
import numpy as np
import math
from timm.models.vision_transformer import PatchEmbed, Attention, Mlp
from einops import rearrange, reduce, repeat
from attention import CrossAttention

def modulate(x, shift, scale, T):

    N, M = x.shape[-2], x.shape[-1]
    B = scale.shape[0]
    x = rearrange(x, '(b t) n m-> b (t n) m',b=B,t=T,n=N,m=M)
    x = x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)
    x = rearrange(x, 'b (t n) m-> (b t) n m',b=B,t=T,n=N,m=M)
    return x


#################################################################################
#               Embedding Layers for Timesteps and Class Labels                 #
#################################################################################

class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element.
                          These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """
        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb


class LabelEmbedder(nn.Module):
    """
    Embeds class labels into vector representations. Also handles label dropout for classifier-free guidance.
    """
    def __init__(self, num_classes, hidden_size, dropout_prob):
        super().__init__()
        use_cfg_embedding = dropout_prob > 0
        self.embedding_table = nn.Embedding(num_classes + use_cfg_embedding, hidden_size)
        self.num_classes = num_classes
        self.dropout_prob = dropout_prob

    def token_drop(self, labels, force_drop_ids=None):
        """
        Drops labels to enable classifier-free guidance.
        """
        if force_drop_ids is None:
            drop_ids = torch.rand(labels.shape[0], device=labels.device) < self.dropout_prob
        else:
            drop_ids = force_drop_ids == 1
        labels = torch.where(drop_ids, self.num_classes, labels)
        return labels

    def forward(self, labels, train, force_drop_ids=None):
        use_dropout = self.dropout_prob > 0
        if (train and use_dropout) or (force_drop_ids is not None):
            labels = self.token_drop(labels, force_drop_ids)
        embeddings = self.embedding_table(labels)
        return embeddings



def drop_path(x, drop_prob: float = 0., training: bool = False):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).
    This is the same as the DropConnect impl I created for EfficientNet, etc networks, however,
    the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for
    changing the layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use
    'survival rate' as the argument.
    """
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output

#################################################################################
#                                 Core VDT Model                                #
#################################################################################

class VDTBlock(nn.Module):
    """
    A VDT block with adaptive layer norm zero (adaLN-Zero) conditioning.
    """
    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0, mode='video', num_frames=16, **block_kwargs):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = Attention(hidden_size, num_heads=num_heads, qkv_bias=True, **block_kwargs)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.norm3 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        approx_gelu = lambda: nn.GELU(approximate="tanh")
        self.mlp = Mlp(in_features=hidden_size, hidden_features=mlp_hidden_dim, act_layer=approx_gelu, drop=0)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 9 * hidden_size, bias=True)
        )
        self.crossAtten1 = CrossAttention(hidden_size, hidden_size, num_heads=num_heads, qkv_bias=True)
        self.norm4 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.crossAtten2 = CrossAttention(hidden_size, hidden_size, num_heads=num_heads, qkv_bias=True)
        self.norm5 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.num_frames = num_frames
        
        self.mode = mode
        
        ## Temporal Attention Parameters
        if self.mode == 'video':
            self.temporal_norm1 = nn.LayerNorm(hidden_size)
            self.temporal_attn = Attention(
              hidden_size, num_heads=num_heads, qkv_bias=True) # multihead attention; learnable bias for query, key, value
            self.temporal_fc = nn.Linear(hidden_size, hidden_size)

    def forward(self, x, t, y):
        # x shape: # (B*T, T', D)
        # t shape: # (B, D)
        # y shape: # (B, D)
        
        shift_msa, scale_msa, gate_msa, shift_msa1, scale_msa1, gate_msa1, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(t).chunk(9, dim=1) # (B,D)
        T = self.num_frames
        K, N, M = x.shape 
        B = K // T 
        
        # # print('y shape: ', y.shape)
        # x1 = rearrange(x, '(b t) n m-> b (t n) m',b=B,t=T,n=N,m=M) # (B, T*T', D)
        # crossAtten1 = self.crossAtten1(self.norm4(x1), y.unsqueeze(1)) # (B, T*T', D)
        # crossAtten1 = rearrange(crossAtten1, 'b (t n) m-> (b t) n m',b=B,t=T,n=N,m=M) # (B*T, T', D)
        # x1 = x + crossAtten1
        # # print('x1 shape: ', x1.shape)


        if self.mode == 'video':
            x = rearrange(x, '(b t) n m -> (b n) t m',b=B,t=T,n=N,m=M) # (B*T, T', D) -> (B*T', T, D)
            res_temporal = self.temporal_attn(self.temporal_norm1(x)) # output: (B*T', T, D)
            res_temporal = rearrange(res_temporal, '(b n) t m -> (b t) n m',b=B,t=T,n=N,m=M) # (B*T', T, D) -> (B*T, T', D)
            res_temporal = self.temporal_fc(res_temporal)
            x = rearrange(x, '(b n) t m -> (b t) n m',b=B,t=T,n=N,m=M) # (B*T, T', D)
            
            # cross-attention
            x = rearrange(x, '(b t) n m-> b (t n) m',b=B,t=T,n=N,m=M) # (B, T*T', D)
            crossAtten1 = self.crossAtten1(self.norm4(x), y.unsqueeze(1)) # (B, T*T', D)
            crossAtten1 = rearrange(crossAtten1, 'b (t n) m-> (b t) n m',b=B,t=T,n=N,m=M) # (B*T, T', D)
            

            x = rearrange(x, 'b (t n) m-> (b t) n m',b=B,t=T,n=N,m=M) # (B*T, T', D)
            x = x + res_temporal # residual connection, shape (B*T, T', D)

        attn = self.attn(modulate(self.norm1(x), shift_msa, scale_msa, self.num_frames)) # (B*T, T', D)
        attn = rearrange(attn, '(b t) n m-> b (t n) m',b=B,t=T,n=N,m=M) # (B, T*T', D)
        attn = gate_msa.unsqueeze(1) * attn # (B, T*T', D)
        attn = rearrange(attn, 'b (t n) m-> (b t) n m',b=B,t=T,n=N,m=M) # (B*T, T', D)
        x = x + attn

        
        crossAtten2 = self.crossAtten2(modulate(self.norm2(x), shift_msa1, scale_msa1, self.num_frames), self.norm5(crossAtten1)) # (B*T, T', D)
        crossAtten2 = rearrange(crossAtten2, '(b t) n m-> b (t n) m',b=B,t=T,n=N,m=M) # (B, T*T', D)
        crossAtten2 = gate_msa1.unsqueeze(1) * crossAtten2 # (B, T*T', D)
        crossAtten2 = rearrange(crossAtten2, 'b (t n) m-> (b t) n m',b=B,t=T,n=N,m=M) # (B*T, T', D)
        x = x + crossAtten2


        mlp = self.mlp(modulate(self.norm3(x), shift_mlp, scale_mlp, self.num_frames)) # (B*T, T', D)
        mlp = rearrange(mlp, '(b t) n m-> b (t n) m',b=B,t=T,n=N,m=M) # (B, T*T', D)
        mlp = gate_mlp.unsqueeze(1) * mlp # (B, T*T', D)
        mlp = rearrange(mlp, 'b (t n) m-> (b t) n m',b=B,t=T,n=N,m=M) # (B*T, T', D)
        x = x + mlp # (B*T, T', D)


        return x

class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)

class FinalLayer(nn.Module):
    """
    The final layer of VDT.
    """
    def __init__(self, hidden_size, patch_size, out_channels, num_frames):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, patch_size * patch_size * out_channels, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=True)
        )
        self.num_frames = num_frames

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        x = modulate(self.norm_final(x), shift, scale, self.num_frames)
        x = self.linear(x)
        return x


class VDT(nn.Module):
    """
    Diffusion model with a Transformer backbone.
    """
    def __init__(
        self,
        input_size=32,
        patch_size=2,
        in_channels=4,
        hidden_size=1152, # transformer embedding dimension
        depth=28,
        num_heads=16, # for multi-head attention block
        mlp_ratio=4.0,
        class_dropout_prob=0.1,
        num_classes=2, # for the classification problems with labels or with text inputs
        learn_sigma=True, 
        mode='video',
        num_frames=16
    ):
        super().__init__()
        self.learn_sigma = learn_sigma
        self.in_channels = in_channels
        self.out_channels = in_channels * 2 if learn_sigma else in_channels
        self.patch_size = patch_size
        self.num_heads = num_heads

        self.x_embedder = PatchEmbed(input_size, patch_size, in_channels, hidden_size, bias=True) # in_channels: number of channels in input data; hidden_size: embedding dimension
        self.t_embedder = TimestepEmbedder(hidden_size)
        self.y_embedder = LabelEmbedder(num_classes, hidden_size, class_dropout_prob)
        num_patches = self.x_embedder.num_patches
        # Will use fixed sin-cos embedding:
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, hidden_size), requires_grad=False)
        
        # for videos, add another condition to learn temporal relationships between frames
        self.mode = mode
        if self.mode == 'video':
            self.num_frames = num_frames
            self.time_embed = nn.Parameter(torch.zeros(1, num_frames, hidden_size), requires_grad=False) # shape: (1, num_frames, hidden_size): hidden_size is the dimensionality of the embedding vector for each frame. Tensor non-learnable, meaning its values will not be updated during backpropagation.
            self.time_drop = nn.Dropout(p=0) # dropout probability of 0, meaning no dropout
        else:
            self.num_frames = 1

        self.blocks = nn.ModuleList([
            VDTBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio, mode=mode, num_frames=self.num_frames) for _ in range(depth)
        ])
        self.final_layer = FinalLayer(hidden_size, patch_size, self.out_channels, self.num_frames)
        self.initialize_weights()

    def initialize_weights(self):
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)

        # Initialize (and freeze) pos_embed by sin-cos embedding:
        pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], int(self.x_embedder.num_patches ** 0.5))
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        if self.mode == 'video':
            grid_num_frames = np.arange(self.num_frames, dtype=np.float32)
            time_embed = get_1d_sincos_pos_embed_from_grid(self.pos_embed.shape[-1], grid_num_frames)
            self.time_embed.data.copy_(torch.from_numpy(time_embed).float().unsqueeze(0))

        # Initialize patch_embed like nn.Linear (instead of nn.Conv2d):
        w = self.x_embedder.proj.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        nn.init.constant_(self.x_embedder.proj.bias, 0)

        # Initialize label embedding table:
        nn.init.normal_(self.y_embedder.embedding_table.weight, std=0.02)

        # Initialize timestep embedding MLP:
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

        # Zero-out adaLN modulation layers in VDT blocks:
        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        # Zero-out output layers:
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

    def unpatchify(self, x):
        """
        x: (N, T', patch_size**2 * C)
        # imgs: (N, H, W, C)
        imgs: (N, out_channels, H, W)
        """
        c = self.out_channels
        p = self.x_embedder.patch_size[0]
        h = w = int(x.shape[1] ** 0.5)
        assert h * w == x.shape[1]

        x = x.reshape(shape=(x.shape[0], h, w, p, p, c))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], c, h * p, h * p))
        return imgs

    def forward(self, x, t, y):
        """
        Forward pass of VDT.
        x: (B, T, C, H, W) tensor of spatial inputs (images or latent representations of images)
        t: (B,) tensor of diffusion timesteps
        y: (B,) tensor of class labels
        """
        
        B, T, C, W, H = x.shape # 32 16 4 8 8 --> input as video sequences
        x = x.contiguous().view(-1, C, W, H) # (B*T, C, W, H) --> (N, C, W, H)
  
        # Adding temporal and spatial embeddings to each patch
        x = self.x_embedder(x) + self.pos_embed  # Patch and positional embedding input video sequences. Output shape: (N, T', D), where num_patches (T') = H * W / patch_size ** 2; D = hidden_size; N = B*T
        if self.mode == 'video':
            # Temporal embed
            x = rearrange(x, '(b t) n m -> (b n) t m',b=B,t=T) # reshape (B*T, T', D) to (B*T',T, D)
            ## Resizing time embeddings in case they don't match
            x = x + self.time_embed # consider time relationships between frames in video sequences; shape of time_embed: (1, T, D)
            x = self.time_drop(x)
            x = rearrange(x, '(b n) t m -> (b t) n m',b=B,t=T) # reshape (B*T',T, D) to (B*T, T', D) --> (N, T', D)
        
        t = self.t_embedder(t)                   # (B, D): Timesteps in Diffusion Model
        y = self.y_embedder(y.long(), self.training)    # (B, D): Class labels (additional conditioning) 
        
        # c = t + y                             # (B, D): addition conbination for adaLN 
  
        for block in self.blocks:
            x = block(x, t, y)                      # (N, T', D)
            
        # print('output shape after blocks: ', x.shape)    
        x = self.final_layer(x, t)                # (N, T', patch_size ** 2 * out_channels)
        # print('final layer output shape: ', x.shape)

        x = self.unpatchify(x)                   # (N, out_channels, W, H)
        # print("upatchify output shape: ", x.shape)
        x = x.view(B, T, x.shape[-3], x.shape[-2], x.shape[-1]) # # (B, T, out_channels, W,H)
        # print("final model_output shape: ", x.shape)
        return x

    def forward_with_cfg(self, x, t, y, cfg_scale):
        """
        Forward pass of VDT, but also batches the unconditional forward pass for classifier-free guidance.
        """
        # https://github.com/openai/glide-text2im/blob/main/notebooks/text2im.ipynb
        half = x[: len(x) // 2]
        combined = torch.cat([half, half], dim=0)
        model_out = self.forward(combined, t, y)
        # For exact reproducibility reasons, we apply classifier-free guidance on only
        # three channels by default. The standard approach to cfg applies it to all channels.
        # This can be done by uncommenting the following line and commenting-out the line following that.
        # eps, rest = model_out[:, :self.in_channels], model_out[:, self.in_channels:]
        eps, rest = model_out[:, :3], model_out[:, 3:]
        cond_eps, uncond_eps = torch.split(eps, len(eps) // 2, dim=0)
        half_eps = uncond_eps + cfg_scale * (cond_eps - uncond_eps)
        eps = torch.cat([half_eps, half_eps], dim=0)
        return torch.cat([eps, rest], dim=1)


#################################################################################
#                   Sine/Cosine Positional Embedding Functions                  #
#################################################################################
# https://github.com/facebookresearch/mae/blob/main/util/pos_embed.py

def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False, extra_tokens=0):
    """
    grid_size: int of the grid height and width
    return:
    pos_embed: [grid_size*grid_size, embed_dim] or [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, grid_size, grid_size])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token and extra_tokens > 0:
        pos_embed = np.concatenate([np.zeros([extra_tokens, embed_dim]), pos_embed], axis=0)
    return pos_embed


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)

    emb = np.concatenate([emb_h, emb_w], axis=1) # (H*W, D)
    return emb


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.
    omega = 1. / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum('m,d->md', pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out) # (M, D/2)
    emb_cos = np.cos(out) # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb


#################################################################################
#                                   VDT Configs                                  #
#################################################################################

def VDT_L_2(**kwargs):
    return VDT(depth=28, hidden_size=1152, patch_size=2, num_heads=16, **kwargs)

def VDT_S_2(**kwargs):
    return VDT(depth=12, hidden_size=384, patch_size=2, num_heads=6, **kwargs)


VDT_models = {
    'VDT-L/2':  VDT_L_2,
    'VDT-S/2':  VDT_S_2,   
}
