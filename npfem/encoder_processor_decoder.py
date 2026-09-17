# Dependencies
import torch
import torch.nn as nn
import math
import torch.nn.functional as F
from einops import rearrange, repeat, reduce

#from warp import pos

def build_mlp(
    input_size: int,
    hidden_layer_sizes: int,
    output_size: int = None,
    activation: nn.Module = nn.ReLU,
    output_activation: nn.Module = nn.Identity,
    layer_norm: bool = True, # Option to add LayerNorm at the end
    layer_norm_dim: int = None, # Dimension for LayerNorm
) -> nn.Sequential:
    
    # Determine layer sizes, including input and output
    layer_sizes = [input_size] + hidden_layer_sizes
    final_output_size = output_size if output_size is not None else layer_sizes[-1]
    if output_size is not None:
        layer_sizes.append(output_size)

    num_layers = len(layer_sizes) - 1

    # Prepare activation functions for each layer
    activations = [activation] * (num_layers - 1) + [output_activation]

    # Build the sequential MLP
    mlp = nn.Sequential()
    #if layer_norm:
    #    mlp.add_module("layer_norm_input", nn.LayerNorm(input_size))
    for i in range(num_layers):
        mlp.add_module(f"linear_{i}", nn.Linear(layer_sizes[i], layer_sizes[i+1]))
        # Add activation function only if it's not Identity (avoids adding Identity())
        if activations[i] is not nn.Identity:
            mlp.add_module(f"activation_{i}", activations[i]())

    # Add optional Layer Normalization at the end
    if layer_norm:
        if layer_norm_dim is None:
             layer_norm_dim = final_output_size
        mlp.add_module("layer_norm", nn.LayerNorm(final_output_size))

    return mlp

class Encoder(nn.Module):
    def __init__(self, n_in_features, n_out_features, nmlp_layers, mlp_hidden_dim):
        super().__init__()
        
        self.node_fn = build_mlp(
            input_size=n_in_features,# + pos_encoded_dim, 
            hidden_layer_sizes=[mlp_hidden_dim] * nmlp_layers,
            output_size=n_out_features,
            layer_norm=True
        )

        self.glob_fn = build_mlp(
            input_size=2,# + pos_encoded_dim, 
            hidden_layer_sizes=[mlp_hidden_dim] * 3,
            output_size=mlp_hidden_dim*2,
            layer_norm=True
        )
        '''
        self.glob_fn = nn.Sequential(
            nn.Linear(5, mlp_hidden_dim),
            nn.ReLU(),
            nn.Linear(mlp_hidden_dim, mlp_hidden_dim)
        )
        '''

        #self.to_gamma = nn.Linear(mlp_hidden_dim, mlp_hidden_dim)
        #self.to_beta  = nn.Linear(mlp_hidden_dim, mlp_hidden_dim)

        #self.embedding = nn.Embedding(2, mlp_hidden_dim)

    def forward(self, x, mat_features):
        #glob_feat = self.glob_fn(mat_features)
        #gamma = self.to_gamma(glob_feat)
        #beta = self.to_beta(glob_feat)
        gamma, beta = self.glob_fn(mat_features).chunk(2, dim=-1)
        #x_combined = torch.cat([x, mat_features], dim=-1)
        x = self.node_fn(x)
        return gamma * x + beta# + self.embedding(bound.long())
    

class Processor(nn.Module):
    """
    Processes a set of latent node representations using Transformer Decoder layers.

    This module applies self-attention mechanisms across the entire set of node embeddings
    received from the Encoder. It uses `nn.TransformerDecoderLayer` configured for
    self-attention (where the input sequence acts as both query, key, and value,
    i.e., `memory=tgt`). This allows each node's representation to be updated based
    on weighted information from all other nodes in the set. The process is repeated
    over multiple layers. Includes residual connections and layer normalization.
    """
    def __init__(
        self,
        n_in_features: int,           # Input dimension (latent dim from Encoder)
        mlp_hidden_dim: int,     # Dimension of the feedforward network within Transformer layers
        nhead: int = 4,          # Number of attention heads
        nlayers: int = 2,        # Number of Transformer Decoder layers
        dropout: float = 0.1,    # Dropout rate within Transformer layers
    ):
        """Initializes the Processor module.

        Args:
            nnode_in: Dimensionality of the input latent node features (from Encoder).
            mlp_hidden_dim: Dimension of the feedforward network model in Transformer layers.
            nhead: Number of parallel attention heads in Transformer layers. Must divide nnode_in.
            nlayers: Number of stacked Transformer Decoder layers.
            dropout: Dropout probability used in Transformer layers.
        """
        super().__init__()

        # Feature dimension for the Transformer model must match the input
        d_model = n_in_features

        # Check if nhead divides d_model
        if d_model % nhead != 0:
            raise ValueError(f"nhead ({nhead}) must divide d_model/nnode_in ({d_model})")
        '''
        # Configure the Transformer Decoder Layer for self-attention
        transformer_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=mlp_hidden_dim,
            dropout=dropout,
            activation=nn.ReLU(), # Standard activation for Transformer FFN
            batch_first=True      # Expect input shape [batch, sequence_len, features]
        )
        # Stack multiple TransformerDecoderLayers
        self.transformer = nn.TransformerDecoder(
            decoder_layer=transformer_layer,
            num_layers=nlayers
        )
        '''
        transformer_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=mlp_hidden_dim,
            dropout=dropout,
            activation=nn.ReLU(), # Standard activation for Transformer FFN
            batch_first=True,      # Expect input shape [batch, sequence_len, features]
            norm_first=True
        )
        # Stack multiple TransformerEncoderLayers
        self.transformer = nn.TransformerEncoder(
            encoder_layer=transformer_layer,
            num_layers=nlayers
        )

        self.inv_freq = nn.Parameter(
            torch.stack([
                10.0 / (10000 ** (torch.arange(32).float() / 32))
                for _ in range(3)
            ])
        )

        # Layer normalization applied after the transformer stack + residual
        #self.norm = nn.LayerNorm(d_model)
    
    def rope(self, x: torch.Tensor, pos: torch.Tensor) -> torch.Tensor:
        """
        Apply N-dimensional Rotary Positional Embedding (RoPE).
        
        Args:
            x: [batch, seq_len, dim] tensor (e.g. queries or keys)
            pos: [seq_len, n_dim] tensor of positions (e.g. x,y,z coordinates)
        """
        b, seq_len, dim = x.shape
        n_dim = pos.shape[-1]
        half_dim = dim // (2 * n_dim)
        
        if half_dim * 2 * n_dim != dim:
            raise ValueError(f"Feature dim ({dim}) must be divisible by 2 * n_dim ({2 * n_dim})")

        #freq_seq = torch.arange(half_dim, device=x.device, dtype=torch.float32)
        #inv_freq = 1.0 / (10000 ** (freq_seq / half_dim))  # frequency bands

        out = torch.zeros_like(x)

        for d in range(n_dim):
            start = d * 2 * half_dim
            end = (d + 1) * 2 * half_dim

            # Compute phase per node for dimension d
            freqs = torch.einsum('i,j->ij', pos[:, d], self.inv_freq[d])
            sin, cos = torch.sin(freqs), torch.cos(freqs)

            x_slice = x[:, :, start:end]
            x1, x2 = x_slice[..., :half_dim], x_slice[..., half_dim:]
            x_rot = torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)
            out[:, :, start:end] = x_rot

        return out

    def forward(self, x: torch.Tensor, batch: torch.Tensor, pos: torch.Tensor) -> torch.Tensor:
        """
        Applies Transformer-based self-attention processing to latent node features.

        Input `x` is expected to be shaped `[num_nodes, nnode_in]`.
        Applies self-attention, layer normalization, and a residual connection.

        Args:
            x: Latent node feature tensor with shape `[num_nodes, nnode_in]`.

        Returns:
            torch.Tensor: Processed latent node features with the same shape as input,
                      `[num_nodes, nnode_in]`.
        """

        # Store input for the final residual connection
        residual_input = x

        # Add batch dimension
        x_batched = residual_input.unsqueeze(0) # Shape: (1, n_nodes, input_dim)

        x_batched = self.rope(x_batched, pos)

        attn_mask = (batch.unsqueeze(1) != batch.unsqueeze(0))

        # Apply Transformer
        '''
        processed_x_batched = self.transformer(tgt=x_batched,
                                       memory=x_batched, 
                                       tgt_mask=attn_mask,
                                       memory_mask=None) # Output shape: (1, n_nodes, input_dim)
        '''
        processed_x_batched = self.transformer(src=x_batched, mask=attn_mask)
        # Remove batch dimension
        processed_x = processed_x_batched.squeeze(0) # Shape: (n_nodes, input_dim)

        # Apply residual connection and layer normalization (Post-LN style)
        #processed_x = residual_input + processed_x
        #processed_x = self.norm(processed_x)

        return processed_x # Output Shape: [num_nodes, nnode_in]

class Decoder(nn.Module):
    """
    Decodes processed latent node representations into physical quantities.

    Takes the final latent embeddings from the Processor and passes them through
    separate MLPs to predict velocity and pressure for each node.
    """
    def __init__(
        self,
        n_in_features: int,
        nmlp_layers: int,
        mlp_hidden_dim: int,
        output_dim_vel: int = 2,
        output_dim_press: int = 1
    ):
        """Initializes the Decoder module.

        Args:
            nnode_in: Dimensionality of the input latent features (from Processor).
            nmlp_layers: Number of hidden layers for the output MLPs.
            mlp_hidden_dim: Size of the hidden layers for the output MLPs.
            output_dim_vel: The dimension of the velocity output. Defaults to 2.
            output_dim_press: The dimension of the pressure output. Defaults to 1.
        """
        
        super().__init__()
        
        self.vel_fn = build_mlp(
            input_size=n_in_features,
            hidden_layer_sizes=[mlp_hidden_dim] * nmlp_layers,
            output_size=output_dim_vel*2 + 1,
            output_activation=nn.Identity,
            layer_norm=False
        )
        
        self.glob_fn = build_mlp(
            input_size=2,# + pos_encoded_dim, 
            hidden_layer_sizes=[mlp_hidden_dim] * 3,
            output_size=mlp_hidden_dim*2,
            layer_norm=True
        )
        '''
        self.pos_fn = build_mlp(
            input_size=output_dim_vel,
            hidden_layer_sizes=[mlp_hidden_dim] * nmlp_layers,
            output_size=output_dim_vel,
            output_activation=nn.Identity,
            layer_norm=False
        )
        '''
        
        '''
        self.press_fn = build_mlp(
            input_size=n_in_features,
            hidden_layer_sizes=[mlp_hidden_dim] * nmlp_layers,
            output_size=output_dim_press,
            output_activation=nn.Identity,
            layer_norm=False
        )
        
        self.velpress_fn = build_mlp(
            input_size=n_in_features,
            hidden_layer_sizes=[mlp_hidden_dim] * nmlp_layers,
            output_size=output_dim_press + output_dim_vel,
            output_activation=nn.Identity,
            layer_norm=False
        )
        '''


    def forward(self, x: torch.Tensor, glob_feat: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Applies the decoding process to the processed node features.

        Args:
            x: Processed latent node feature tensor. Expected shape is
               `[num_nodes, nnode_in]`.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: A tuple containing:
                - velocity (Tensor): Predicted velocity. Shape `[num_nodes, output_dim_vel]`.
                - pressure (Tensor): Predicted pressure. Shape `[num_nodes, output_dim_press]`.
        """
        
        #velocity = self.vel_fn(x)
        #position = self.pos_fn(velocity)
        #pressure = self.press_fn(x)
        #velpress = self.velpress_fn(x)
        #velocity = velpress[:, :3]
        #pressure = velpress[:, 3:]
        beta, gamma = self.glob_fn(glob_feat).chunk(2, dim=-1)
        x = beta * x + gamma
        output = self.vel_fn(x)
        velocity = output[:, :3]
        position = output[:, 3:-1]
        #velocity = output[:, :-1]
        pressure = output[:, -1]
        #position = self.pos_fn(velocity)

        return velocity, position, pressure