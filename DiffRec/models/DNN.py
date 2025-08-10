import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np
import math

class DNN(nn.Module):
    """
    A deep neural network for the reverse diffusion process with conditional support.
    """
    def __init__(self, in_dims, out_dims, emb_size, time_type="cat", norm=True, dropout=0.5, use_conditionals=False, conditional_dim=2):
        super(DNN, self).__init__()
        self.in_dims = in_dims
        self.out_dims = out_dims
        self.time_emb_dim = emb_size
        self.time_type = time_type
        self.norm = norm
        self.use_conditionals = use_conditionals
        self.conditional_dim = conditional_dim
        
        # Time embedding layer
        self.emb_layer = nn.Linear(self.time_emb_dim, self.time_emb_dim)
        
        # Conditional embedding layer for user groups
        if self.use_conditionals:
            self.conditional_emb_layer = nn.Linear(self.conditional_dim, self.conditional_dim)
        
        # Input layers
        self.in_layers = nn.ModuleList()
        self.in_layers.append(nn.Linear(self.in_dims[0] + self.time_emb_dim + (self.conditional_dim if self.use_conditionals else 0), self.in_dims[1]))
        for i in range(1, len(self.in_dims) - 1):
            self.in_layers.append(nn.Linear(self.in_dims[i], self.in_dims[i + 1]))
        
        # Output layers
        self.out_layers = nn.ModuleList()
        for i in range(len(self.out_dims) - 1):
            self.out_layers.append(nn.Linear(self.out_dims[i], self.out_dims[i + 1]))
        
        self.drop = nn.Dropout(p=dropout)
        self.init_weights()
    
    def init_weights(self):
        for layer in self.in_layers:
            # Xavier Initialization for weights
            size = layer.weight.size()
            fan_out = size[0]
            fan_in = size[1]
            std = np.sqrt(2.0 / (fan_in + fan_out))
            layer.weight.data.normal_(0.0, std)

            # Normal Initialization for weights
            layer.bias.data.normal_(0.0, 0.001)
        
        for layer in self.out_layers:
            # Xavier Initialization for weights
            size = layer.weight.size()
            fan_out = size[0]
            fan_in = size[1]
            std = np.sqrt(2.0 / (fan_in + fan_out))
            layer.weight.data.normal_(0.0, std)

            # Normal Initialization for weights
            layer.bias.data.normal_(0.0, 0.001)
        
        size = self.emb_layer.weight.size()
        fan_out = size[0]
        fan_in = size[1]
        std = np.sqrt(2.0 / (fan_in + fan_out))
        self.emb_layer.weight.data.normal_(0.0, std)
        self.emb_layer.bias.data.normal_(0.0, 0.001)
        
        # Initialize conditional embedding layer if used
        if self.use_conditionals:
            size = self.conditional_emb_layer.weight.size()
            fan_out = size[0]
            fan_in = size[1]
            std = np.sqrt(2.0 / (fan_in + fan_out))
            self.conditional_emb_layer.weight.data.normal_(0.0, std)
            self.conditional_emb_layer.bias.data.normal_(0.0, 0.001)
    
    def forward(self, x, timesteps, conditionals=None):
        time_emb = timestep_embedding(timesteps, self.time_emb_dim).to(x.device)
        emb = self.emb_layer(time_emb)
        
        if self.norm:
            x = F.normalize(x)
        x = self.drop(x)
        
        # Concatenate input with time embedding
        h = torch.cat([x, emb], dim=-1)
        
        # Add conditionals if provided and enabled
        if self.use_conditionals and conditionals is not None:
            # Process conditionals through embedding layer
            conditional_emb = self.conditional_emb_layer(conditionals)
            h = torch.cat([h, conditional_emb], dim=-1)
        
        for i, layer in enumerate(self.in_layers):
            h = layer(h)
            h = torch.tanh(h)
        
        for i, layer in enumerate(self.out_layers):
            h = layer(h)
            if i != len(self.out_layers) - 1:
                h = torch.tanh(h)
        
        return h


def timestep_embedding(timesteps, dim, max_period=10000):
    """
    Create sinusoidal timestep embeddings.

    :param timesteps: a 1-D Tensor of N indices, one per batch element.
                      These may be fractional.
    :param dim: the dimension of the output.
    :param max_period: controls the minimum frequency of the embeddings.
    :return: an [N x dim] Tensor of positional embeddings.
    """

    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
    ).to(timesteps.device)
    args = timesteps[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding
