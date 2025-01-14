import torch
import torch.nn as nn
import torch.nn.functional as F


class VAE_resblock(nn.Module):
    """
    VAE_resblock is a residual block used in Variational Autoencoders (VAEs).
    It consists of two convolutional layers with Group Normalization and SiLU activation.
    If the input and output channels are different, a shortcut connection is used to match dimensions.
    """

    def __init__(self, in_channels, out_channels):
        """
        Initializes the VAE_resblock.

        Parameters:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        """
        super(VAE_resblock, self).__init__()
        # First Group Normalization layer
        self.groupnorm1 = nn.GroupNorm(32, in_channels)
        # First Convolutional layer
        self.conv1 = nn.Conv2d(
            in_channels, out_channels, kernel_size=3, stride=1, padding=1
        )
        # Second Group Normalization layer
        self.groupnorm2 = nn.GroupNorm(32, out_channels)

        # Shortcut connection to match dimensions if in_channels != out_channels
        if in_channels == out_channels:
            self.shortcut = nn.Identity()
        else:
            self.shortcut = nn.Conv2d(
                in_channels, out_channels, kernel_size=1, stride=1, padding=0
            )

    def forward(self, x):
        """
        Forward pass of the VAE_resblock.

        Parameters:
        x (torch.Tensor): Input tensor.

        Returns:
        torch.Tensor: Output tensor after passing through the residual block.
        """
        residual = x  # Save the input tensor for the shortcut connection
        out = self.groupnorm1(x)
        out = F.silu(out)
        out = self.conv1(out)
        out = self.groupnorm2(out)
        out = F.silu(out)
        out = self.conv1(out)
        out += self.shortcut(residual)  # Add the shortcut connection
        return out  # Return the output tensor


class VAE_attention(nn.Module):
    """
    VAE_attention is an attention mechanism used in Variational Autoencoders (VAEs).
    It applies Group Normalization followed by a multi-head attention mechanism.
    """

    def __init__(self, channels, num_heads):
        """
        Initializes the VAE_attention.

        Parameters:
        channels (int): Number of input and output channels.
        num_heads (int): Number of attention heads.
        """
        super(VAE_attention, self).__init__()
        self.groupnorm = nn.GroupNorm(32, channels)
        self.in_proj = nn.Linear(channels, 3 * channels, bias=True)
        self.out_proj = nn.Linear(channels, channels, bias=False)
        self.attention = nn.MultiheadAttention(channels, num_heads, )

    def forward(self, x):
        """
        Forward pass of the VAE_attention.

        Parameters:
        x (torch.Tensor): Input tensor with shape (Batch_Size, Features, Height, Width).

        Returns:
        torch.Tensor: Output tensor after applying attention mechanism.
        """
    

        x = self.groupnorm(x)  # Apply Group Normalization

        n, c, h, w = x.shape  # Get the shape of the input tensor
        key , query, value = self.in_proj(x).chunk(3, dim=-1)  

        key = key.view(n, h * w, c).transpose(1, 2)  # Reshape key tensor
        query = query.view(n, h * w, c).transpose(1, 2)
        value = value.view(n, h * w, c).transpose(1, 2)
        out, _ = self.attention(query, key, value)  
        out = out.transpose(1, 2).view(n, c, h, w)  
        out = self.out_proj(out)  
        return out 


        