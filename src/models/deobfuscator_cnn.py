"""
CNN model architecture for deobfuscating text images.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Dict, Optional, Union
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class DoubleConv(nn.Module):
    """Double convolution block used in U-Net."""
    
    def __init__(self, in_channels: int, out_channels: int, use_batchnorm: bool = True):
        """
        Initialize double convolution block.
        
        Args:
            in_channels: Number of input channels
            out_channels: Number of output channels
            use_batchnorm: Whether to use batch normalization
        """
        super(DoubleConv, self).__init__()
        
        if use_batchnorm:
            self.conv = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            )
        else:
            self.conv = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=True),
                nn.ReLU(inplace=True)
            )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the double convolution block."""
        return self.conv(x)


class Down(nn.Module):
    """Downsampling block used in U-Net."""
    
    def __init__(self, in_channels: int, out_channels: int, use_batchnorm: bool = True):
        """
        Initialize downsampling block.
        
        Args:
            in_channels: Number of input channels
            out_channels: Number of output channels
            use_batchnorm: Whether to use batch normalization
        """
        super(Down, self).__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels, use_batchnorm)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the downsampling block."""
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upsampling block used in U-Net."""
    
    def __init__(self, in_channels: int, out_channels: int, use_batchnorm: bool = True,
                 bilinear: bool = True):
        """
        Initialize upsampling block.
        
        Args:
            in_channels: Number of input channels
            out_channels: Number of output channels
            use_batchnorm: Whether to use batch normalization
            bilinear: Whether to use bilinear upsampling (otherwise use transposed convolution)
        """
        super(Up, self).__init__()
        
        # Use bilinear interpolation or transposed convolution for upsampling
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, use_batchnorm)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels, use_batchnorm)
    
    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the upsampling block.
        
        Args:
            x1: Features from the encoder path
            x2: Features from the decoder path
            
        Returns:
            Output tensor
        """
        x1 = self.up(x1)
        
        # Pad x1 if sizes don't match
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
        
        # Concatenate along channel dimension
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    """Output convolution block used in U-Net."""
    
    def __init__(self, in_channels: int, out_channels: int):
        """
        Initialize output convolution block.
        
        Args:
            in_channels: Number of input channels
            out_channels: Number of output channels
        """
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the output convolution block."""
        return self.conv(x)


class UNet(nn.Module):
    """U-Net model for image-to-image translation."""
    
    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        features: List[int] = [64, 128, 256, 512],
        use_batchnorm: bool = True,
        bilinear: bool = True,
        dropout_rate: float = 0.0
    ):
        """
        Initialize U-Net model.
        
        Args:
            in_channels: Number of input channels
            out_channels: Number of output channels
            features: List of feature dimensions for each level
            use_batchnorm: Whether to use batch normalization
            bilinear: Whether to use bilinear upsampling
            dropout_rate: Dropout rate (0.0 means no dropout)
        """
        super(UNet, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.features = features
        self.bilinear = bilinear
        self.dropout_rate = dropout_rate
        
        # Input block
        self.inc = DoubleConv(in_channels, features[0], use_batchnorm)
        
        # Encoder (downsampling) blocks
        self.downs = nn.ModuleList()
        for i in range(len(features) - 1):
            self.downs.append(Down(features[i], features[i + 1], use_batchnorm))
        
        # Decoder (upsampling) blocks
        self.ups = nn.ModuleList()
        for i in range(len(features) - 1, 0, -1):
            self.ups.append(
                Up(features[i], features[i - 1], use_batchnorm, bilinear)
            )
        
        # Output block
        self.outc = OutConv(features[0], out_channels)
        
        # Dropout layer
        self.dropout = nn.Dropout2d(dropout_rate) if dropout_rate > 0 else None
        
        # Final activation (sigmoid for binary images)
        self.final_activation = nn.Sigmoid()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the U-Net model.
        
        Args:
            x: Input tensor of shape [batch_size, in_channels, height, width]
            
        Returns:
            Output tensor of shape [batch_size, out_channels, height, width]
        """
        # Encoder path
        skip_connections = []
        x1 = self.inc(x)
        skip_connections.append(x1)
        
        for down in self.downs:
            x = down(x)
            skip_connections.append(x)
        
        # Apply dropout at the bottleneck if specified
        if self.dropout is not None:
            x = self.dropout(x)
        
        # Decoder path with skip connections
        skip_connections = skip_connections[:-1]  # Remove the bottleneck from skip connections
        
        for up, skip in zip(self.ups, reversed(skip_connections)):
            x = up(x, skip)
        
        # Final convolution and activation
        x = self.outc(x)
        x = self.final_activation(x)
        
        return x


class SimpleAutoencoder(nn.Module):
    """Simple autoencoder model for deobfuscation."""
    
    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        hidden_dims: List[int] = [32, 64, 128],
        latent_dim: int = 128,
        input_size: Tuple[int, int] = (64, 64),
        use_batchnorm: bool = True,
        dropout_rate: float = 0.2
    ):
        """
        Initialize simple autoencoder model.
        
        Args:
            in_channels: Number of input channels
            out_channels: Number of output channels
            hidden_dims: List of hidden dimensions for encoder/decoder
            latent_dim: Dimension of the latent space
            input_size: Size of input images as (height, width)
            use_batchnorm: Whether to use batch normalization
            dropout_rate: Dropout rate
        """
        super(SimpleAutoencoder, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.latent_dim = latent_dim
        self.input_size = input_size
        
        # Calculate size of feature maps at bottleneck
        bottleneck_h = input_size[0] // (2 ** len(hidden_dims))
        bottleneck_w = input_size[1] // (2 ** len(hidden_dims))
        bottleneck_size = bottleneck_h * bottleneck_w * hidden_dims[-1]
        
        # Encoder
        modules = []
        in_dim = in_channels
        
        for h_dim in hidden_dims:
            modules.append(
                nn.Sequential(
                    nn.Conv2d(in_dim, h_dim, kernel_size=3, stride=2, padding=1),
                    nn.BatchNorm2d(h_dim) if use_batchnorm else nn.Identity(),
                    nn.ReLU(),
                    nn.Dropout2d(dropout_rate) if dropout_rate > 0 else nn.Identity()
                )
            )
            in_dim = h_dim
        
        self.encoder = nn.Sequential(*modules)
        
        # Bottleneck
        self.fc_mu = nn.Linear(bottleneck_size, latent_dim)
        self.fc_decoder = nn.Linear(latent_dim, bottleneck_size)
        
        # Decoder
        modules = []
        hidden_dims.reverse()
        
        for i in range(len(hidden_dims) - 1):
            modules.append(
                nn.Sequential(
                    nn.ConvTranspose2d(
                        hidden_dims[i], hidden_dims[i + 1],
                        kernel_size=3, stride=2, padding=1, output_padding=1
                    ),
                    nn.BatchNorm2d(hidden_dims[i + 1]) if use_batchnorm else nn.Identity(),
                    nn.ReLU(),
                    nn.Dropout2d(dropout_rate) if dropout_rate > 0 else nn.Identity()
                )
            )
        
        # Final layer
        modules.append(
            nn.Sequential(
                nn.ConvTranspose2d(
                    hidden_dims[-1], out_channels,
                    kernel_size=3, stride=2, padding=1, output_padding=1
                ),
                nn.Sigmoid()  # For binary images
            )
        )
        
        self.decoder = nn.Sequential(*modules)
    
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode input to latent space."""
        x = self.encoder(x)
        x = torch.flatten(x, start_dim=1)
        z = self.fc_mu(x)
        return z
    
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode from latent space to output."""
        x = self.fc_decoder(z)
        x = x.view(-1, self.hidden_dims[-1], 
                  self.input_size[0] // (2 ** len(self.hidden_dims)),
                  self.input_size[1] // (2 ** len(self.hidden_dims)))
        x = self.decoder(x)
        return x
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the autoencoder."""
        z = self.encode(x)
        x_recon = self.decode(z)
        return x_recon


class ResidualBlock(nn.Module):
    """Residual block for ResNet-based models."""
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        use_batchnorm: bool = True,
        dropout_rate: float = 0.0
    ):
        """
        Initialize residual block.
        
        Args:
            in_channels: Number of input channels
            out_channels: Number of output channels
            stride: Stride for convolution
            use_batchnorm: Whether to use batch normalization
            dropout_rate: Dropout rate
        """
        super(ResidualBlock, self).__init__()
        
        # First convolution
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, 
                              stride=stride, padding=1, bias=not use_batchnorm)
        self.bn1 = nn.BatchNorm2d(out_channels) if use_batchnorm else nn.Identity()
        self.relu = nn.ReLU(inplace=True)
        
        # Second convolution
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                              stride=1, padding=1, bias=not use_batchnorm)
        self.bn2 = nn.BatchNorm2d(out_channels) if use_batchnorm else nn.Identity()
        
        # Shortcut connection
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1,
                         stride=stride, bias=not use_batchnorm),
                nn.BatchNorm2d(out_channels) if use_batchnorm else nn.Identity()
            )
        
        # Dropout
        self.dropout = nn.Dropout2d(dropout_rate) if dropout_rate > 0 else None
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the residual block."""
        identity = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        if self.dropout is not None:
            out = self.dropout(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        out += self.shortcut(identity)
        out = self.relu(out)
        
        if self.dropout is not None:
            out = self.dropout(out)
        
        return out


class DeobfuscatorResNet(nn.Module):
    """ResNet-based model for deobfuscation."""
    
    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        base_width: int = 64,
        num_blocks: List[int] = [2, 2, 2, 2],
        use_batchnorm: bool = True,
        dropout_rate: float = 0.2
    ):
        """
        Initialize ResNet-based deobfuscator model.
        
        Args:
            in_channels: Number of input channels
            out_channels: Number of output channels
            base_width: Base width for the model
            num_blocks: Number of residual blocks per stage
            use_batchnorm: Whether to use batch normalization
            dropout_rate: Dropout rate
        """
        super(DeobfuscatorResNet, self).__init__()
        
        # Initial convolution
        self.conv1 = nn.Conv2d(in_channels, base_width, kernel_size=7, 
                              stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(base_width) if use_batchnorm else nn.Identity()
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        # Residual blocks
        self.layer1 = self._make_layer(base_width, base_width, num_blocks[0], 
                                      stride=1, use_batchnorm=use_batchnorm, 
                                      dropout_rate=dropout_rate)
        self.layer2 = self._make_layer(base_width, base_width*2, num_blocks[1], 
                                      stride=2, use_batchnorm=use_batchnorm,
                                      dropout_rate=dropout_rate)
        self.layer3 = self._make_layer(base_width*2, base_width*4, num_blocks[2], 
                                      stride=2, use_batchnorm=use_batchnorm,
                                      dropout_rate=dropout_rate)
        self.layer4 = self._make_layer(base_width*4, base_width*8, num_blocks[3], 
                                      stride=2, use_batchnorm=use_batchnorm,
                                      dropout_rate=dropout_rate)
        
        # Upsampling path
        self.up1 = nn.Sequential(
            nn.ConvTranspose2d(base_width*8, base_width*4, kernel_size=3, 
                              stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(base_width*4) if use_batchnorm else nn.Identity(),
            nn.ReLU(inplace=True)
        )
        self.up2 = nn.Sequential(
            nn.ConvTranspose2d(base_width*4, base_width*2, kernel_size=3, 
                              stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(base_width*2) if use_batchnorm else nn.Identity(),
            nn.ReLU(inplace=True)
        )
        self.up3 = nn.Sequential(
            nn.ConvTranspose2d(base_width*2, base_width, kernel_size=3, 
                              stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(base_width) if use_batchnorm else nn.Identity(),
            nn.ReLU(inplace=True)
        )
        self.up4 = nn.Sequential(
            nn.ConvTranspose2d(base_width, base_width//2, kernel_size=3, 
                              stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(base_width//2) if use_batchnorm else nn.Identity(),
            nn.ReLU(inplace=True)
        )
        self.up5 = nn.Sequential(
            nn.ConvTranspose2d(base_width//2, out_channels, kernel_size=3, 
                              stride=2, padding=1, output_padding=1),
            nn.Sigmoid()  # For binary images
        )
    
    def _make_layer(
        self,
        in_channels: int,
        out_channels: int,
        num_blocks: int,
        stride: int,
        use_batchnorm: bool,
        dropout_rate: float
    ) -> nn.Sequential:
        """
        Create a sequence of residual blocks.
        
        Args:
            in_channels: Number of input channels
            out_channels: Number of output channels
            num_blocks: Number of residual blocks
            stride: Stride for the first block
            use_batchnorm: Whether to use batch normalization
            dropout_rate: Dropout rate
            
        Returns:
            Sequential module containing residual blocks
        """
        layers = []
        layers.append(ResidualBlock(in_channels, out_channels, stride=stride, 
                                   use_batchnorm=use_batchnorm, dropout_rate=dropout_rate))
        
        for _ in range(1, num_blocks):
            layers.append(ResidualBlock(out_channels, out_channels, stride=1,
                                       use_batchnorm=use_batchnorm, dropout_rate=dropout_rate))
        
        return nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the ResNet-based deobfuscator."""
        # Encoder path
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        # Decoder path
        x = self.up1(x)
        x = self.up2(x)
        x = self.up3(x)
        x = self.up4(x)
        x = self.up5(x)
        
        return x


def get_model(
    model_name: str,
    config: Dict
) -> nn.Module:
    """
    Get model based on configuration.
    
    Args:
        model_name: Name of the model architecture
        config: Model configuration
        
    Returns:
        PyTorch model instance
    """
    if model_name == "unet":
        model = UNet(
            in_channels=config.get("in_channels", 1),
            out_channels=config.get("out_channels", 1),
            features=config.get("filters", [64, 128, 256, 512]),
            use_batchnorm=config.get("use_batch_norm", True),
            dropout_rate=config.get("dropout_rate", 0.0),
            bilinear=config.get("bilinear", True)
        )
    elif model_name == "autoencoder":
        model = SimpleAutoencoder(
            in_channels=config.get("in_channels", 1),
            out_channels=config.get("out_channels", 1),
            hidden_dims=config.get("hidden_dims", [32, 64, 128]),
            latent_dim=config.get("latent_dim", 128),
            input_size=config.get("input_shape", (64, 64))[:2],
            use_batchnorm=config.get("use_batch_norm", True),
            dropout_rate=config.get("dropout_rate", 0.2)
        )
    elif model_name == "resnet":
        model = DeobfuscatorResNet(
            in_channels=config.get("in_channels", 1),
            out_channels=config.get("out_channels", 1),
            base_width=config.get("base_width", 64),
            num_blocks=config.get("num_blocks", [2, 2, 2, 2]),
            use_batchnorm=config.get("use_batch_norm", True),
            dropout_rate=config.get("dropout_rate", 0.2)
        )
    else:
        raise ValueError(f"Unsupported model architecture: {model_name}")
    
    return model
