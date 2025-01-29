import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import MultiheadAttention


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
        self.groupnorm1 = nn.GroupNorm(32, in_channels)
        self.conv1 = nn.Conv2d(
            in_channels, out_channels, kernel_size=3, stride=1, padding=1
        )
        self.groupnorm2 = nn.GroupNorm(32, out_channels)
        if in_channels == out_channels:
            self.shortcut = nn.Identity()
        else:
            self.shortcut = nn.Conv2d(
                in_channels, out_channels, kernel_size=1, padding=0
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
        out += self.shortcut(residual)
        return out


class VAE_encoder(nn.Module):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.conv1 = nn.Conv2d(3, 128, kernel_size=3, padding=1)
        self.resblock1 = VAE_resblock(128, 128)
        self.resblock2 = VAE_resblock(128, 128)

        self.conv2 = nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=0)
        self.resblock3 = VAE_resblock(256, 256)
        self.resblock4 = VAE_resblock(256, 256)

        self.conv3 = nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=0)
        self.resblock5 = VAE_resblock(512, 512)
        self.resblock6 = VAE_resblock(512, 512)

        self.conv4 = nn.Conv2d(512, 512, kernel_size=3, stride=2, padding=0)
        self.resblock7 = VAE_resblock(512, 512)
        self.resblock8 = VAE_resblock(512, 512)

        self.attantion = MultiheadAttention(512, 8, dropout=0.1)

        self.resblock9 = VAE_resblock(512, 512)
        self.groupnorm = nn.GroupNorm(32, 512)
        self.silu = nn.SiLU()

        self.modules_list = [
            self.conv1,
            self.resblock1,
            self.resblock2,
            self.conv2,
            self.resblock3,
            self.resblock4,
            self.conv3,
            self.resblock5,
            self.resblock6,
            self.conv4,
            self.resblock7,
            self.resblock8,
            self.attantion,
            self.resblock9,
            self.groupnorm,
            self.silu,
        ]

    def forward(self, x, noise):
        for i in self.modules_list:
            if getattr(i, "stride", None) == (2, 2):
                x = F.pad(x, (0, 1, 0, 1))

            # Multi-head attention layer
            if i == self.attantion:
                # Ensure x has 3 dimensions before permuting
                if x.dim() == 4:
                    n, c, h, w = x.shape
                    x = x.view(n, c, h * w).permute(2, 0, 1)
                else:
                    x = x.permute(2, 0, 1)
                x, _ = i(x, x, x)
                x = x.permute(1, 2, 0)
                if x.dim() == 3:
                    x = x.view(n, c, h, w)
            else:
                x = i(x)

        mean, log_variance = torch.chunk(x, 2, dim=1)
        log_variance = torch.clamp(log_variance, -30, 20)
        variance = log_variance.exp()
        stdev = variance.sqrt()
        x = mean + stdev * noise
        x *= 0.18215
        return x


class VAE_decoder(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.conv1 = nn.Conv2d(256, 4, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(4, 512, kernel_size=3, padding=1)

        self.resblock1 = VAE_resblock(512, 512)
        self.Mh_attention = MultiheadAttention(512, 8, dropout=0.1)
        self.resblock2 = VAE_resblock(512, 512)
        self.resblock3 = VAE_resblock(512, 512)
        self.resblock4 = VAE_resblock(512, 512)

        self.upfactor2 = nn.Upsample(scale_factor=2, mode="bilinear")
        self.conv3 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.resblock5 = VAE_resblock(512, 512)
        self.resblock6 = VAE_resblock(512, 512)
        self.resblock7 = VAE_resblock(512, 512)

        self.upfactor3 = nn.Upsample(scale_factor=2, mode="bilinear")
        self.conv4 = nn.Conv2d(512, 256, kernel_size=3, padding=1)

        self.resblock8 = VAE_resblock(256, 256)
        self.resblock9 = VAE_resblock(256, 256)
        self.resblock10 = VAE_resblock(256, 256)

        self.upfactor4 = nn.Upsample(scale_factor=2, mode="bilinear")
        self.conv5 = nn.Conv2d(256, 128, kernel_size=3, padding=1)

        self.resblock11 = VAE_resblock(128, 128)
        self.resblock12 = VAE_resblock(128, 128)
        self.resblock13 = VAE_resblock(128, 128)

        self.groupnorm = nn.GroupNorm(32, 128)
        self.silu = nn.SiLU()
        self.conv6 = nn.Conv2d(128, 3, kernel_size=3, padding=1)

        self.modules_list = [
            self.conv1,
            self.conv2,
            self.resblock1,
            self.Mh_attention,
            self.resblock2,
            self.resblock3,
            self.resblock4,
            self.upfactor2,
            self.conv3,
            self.resblock5,
            self.resblock6,
            self.resblock7,
            self.upfactor3,
            self.conv4,
            self.resblock8,
            self.resblock9,
            self.resblock10,
            self.upfactor4,
            self.conv5,
            self.resblock11,
            self.resblock12,
            self.resblock13,
            self.groupnorm,
            self.silu,
            self.conv6,
        ]

    def forward(self, x):
        x /= 0.18215

        for i in self.modules_list:
            if i == self.Mh_attention:
                # Ensure x has 3 dimensions before permuting
                if x.dim() == 4:
                    n, c, h, w = x.shape
                    x = x.view(n, c, h * w).permute(2, 0, 1)
                else:
                    x = x.permute(2, 0, 1)
                x, _ = i(x, x, x)
                x = x.permute(1, 2, 0)
                if x.dim() == 3:
                    x = x.view(n, c, h, w)
            else:
                x = i(x)

        return x


class VAE(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.encoder = VAE_encoder()
        self.decoder = VAE_decoder()

    def forward(self, x, noise):
        x = self.encoder(x, noise)
        x = self.decoder(x)
        return x


class Discriminator(nn.Module):

    def __init__(
        self,
        im_channels=3,
        conv_channels=[64, 128, 256],
        kernels=[4, 4, 4, 4],
        strides=[2, 2, 2, 1],
        paddings=[1, 1, 1, 1],
    ):
        super().__init__()
        self.im_channels = im_channels
        activation = nn.LeakyReLU(0.2)
        layers_dim = [self.im_channels] + conv_channels + [1]
        self.layers = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(
                        layers_dim[i],
                        layers_dim[i + 1],
                        kernel_size=kernels[i],
                        stride=strides[i],
                        padding=paddings[i],
                        bias=False if i != 0 else True,
                    ),
                    (
                        nn.BatchNorm2d(layers_dim[i + 1])
                        if i != len(layers_dim) - 2 and i != 0
                        else nn.Identity()
                    ),
                    activation if i != len(layers_dim) - 2 else nn.Identity(),
                )
                for i in range(len(layers_dim) - 1)
            ]
        )

    def forward(self, x):
        out = x
        for layer in self.layers:
            out = layer(out)
        return out











if __name__ == "__main__":
    from torchinfo import summary
    model = VAE_encoder().to('cuda')
    input_shape = 256
    x = torch.randn(1, 3, input_shape, input_shape).to('cuda')
    noise = torch.randn(1, 256, input_shape//8, input_shape//8).to('cuda')
    out = model(x, noise)
    print(out.shape)
    model = VAE_decoder().to('cuda')
    out = model(out)
    print("------decoder------")
    print(out.shape)
    print('done')

    #number of perameters
    # print(sum(p.numel() for p in model.parameters() if p.requires_grad))
