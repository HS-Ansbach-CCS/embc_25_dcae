"""
Deep Convolutional Autoencoder for EEG Signal Processing

This module implements a deep convolutional autoencoder (DCAE) architecture
specifically designed for processing EEG signals. The model uses residual 
connections and operates on 1D time-series data.

Author: [Your Name]
Date: July 2025
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional


class DeepConvolutionalAutoencoder(nn.Module):
    """
    Deep Convolutional Autoencoder for EEG signal reconstruction and feature learning.

    This architecture processes EEG signals through a series of convolutional layers
    in the encoder, compresses the data into a latent representation, and reconstructs
    the original signal through transposed convolutions in the decoder.

    Args:
        input_channels (int): Number of EEG channels (default: 23)
        time_steps (int): Number of time steps in the input signal (default: 512)
        latent_dim (int): Dimensionality of the latent space (default: 1000)
        leaky_relu_slope (float): Negative slope for LeakyReLU activation (default: 0.1)
        dropout_rate (float): Dropout rate for regularization (default: 0.2)

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: Reconstructed signal and latent representation
    """

    def __init__(
        self,
        input_channels: int = 23,
        time_steps: int = 512,
        latent_dim: int = 500,
        leaky_relu_slope: float = 0.1,
        dropout_rate: float = 0.2
    ):
        super(DeepConvolutionalAutoencoder, self).__init__()

        # Store hyperparameters
        self.input_channels = input_channels
        self.time_steps = time_steps
        self.latent_dim = latent_dim
        self.leaky_relu_slope = leaky_relu_slope
        self.dropout_rate = dropout_rate

        # Calculate the flattened size after convolution and pooling operations
        self._calculate_conv_output_size()

        # Encoder architecture
        self._build_encoder()

        # Latent space layers
        self.fc_encoder = nn.Linear(self.conv_output_size, latent_dim)
        self.fc_decoder = nn.Linear(latent_dim, self.conv_output_size)

        # Decoder architecture
        self._build_decoder()

        # Initialize weights
        self._initialize_weights()

    def _calculate_conv_output_size(self) -> None:
        """Calculate the output size after convolution and pooling operations."""
        # After conv1: (input_channels, time_steps) -> (32, time_steps)
        # After conv2: (32, time_steps) -> (64, time_steps - 7)
        # After pool2: (64, time_steps - 7) -> (64, (time_steps - 7) // 3)
        # After conv3: (64, (time_steps - 7) // 3) -> (96, (time_steps - 7) // 3 - 15)
        # After pool3: (96, (time_steps - 7) // 3 - 15) -> (96, ((time_steps - 7) // 3 - 15) // 4)

        size_after_conv2 = self.time_steps - 7
        size_after_pool2 = size_after_conv2 // 3
        size_after_conv3 = size_after_pool2 - 15
        size_after_pool3 = size_after_conv3 // 4

        self.conv_output_size = 96 * 1 * size_after_pool3
        self.decoder_reshape_size = (96, 1, size_after_pool3)

    def _build_encoder(self) -> None:
        """Build the encoder layers."""
        # Encoder Layer 1: Spatial convolution across channels
        self.encoder_conv1 = nn.Conv2d(
            in_channels=1, out_channels=32,
            kernel_size=(self.input_channels, 1),
            padding=(0, 0)
        )
        self.encoder_bn1 = nn.BatchNorm2d(32)
        self.encoder_dropout1 = nn.Dropout(self.dropout_rate)

        # Encoder Layer 2: Temporal convolution with pooling
        self.encoder_conv2 = nn.Conv2d(
            in_channels=32, out_channels=64,
            kernel_size=(1, 8),
            padding=(0, 0)
        )
        self.encoder_bn2 = nn.BatchNorm2d(64)
        self.encoder_pool2 = nn.MaxPool2d(kernel_size=(1, 3), stride=(1, 3))
        self.encoder_dropout2 = nn.Dropout(0.1)

        # Encoder Layer 3: Deeper temporal features with pooling
        self.encoder_conv3 = nn.Conv2d(
            in_channels=64, out_channels=96,
            kernel_size=(1, 16),
            padding=(0, 0)
        )
        self.encoder_bn3 = nn.BatchNorm2d(96)
        self.encoder_pool3 = nn.MaxPool2d(kernel_size=(1, 4), stride=(1, 4))
        self.encoder_dropout3 = nn.Dropout(0.1)

    def _build_decoder(self) -> None:
        """Build the decoder layers."""
        # Batch normalization for latent representation
        self.decoder_bn_latent = nn.BatchNorm1d(self.conv_output_size)

        # Decoder Layer 3: Upsample and transpose convolution
        self.decoder_upsample3 = nn.Upsample(size=(1, 153), mode='nearest')
        self.decoder_conv3 = nn.ConvTranspose2d(
            in_channels=96, out_channels=64,
            kernel_size=(1, 16),
            padding=(0, 0)
        )
        self.decoder_bn3 = nn.BatchNorm2d(64)
        self.decoder_dropout3 = nn.Dropout(0.3)

        # Decoder Layer 2: Upsample and transpose convolution
        self.decoder_upsample2 = nn.Upsample(size=(1, 505), mode='nearest')
        self.decoder_conv2 = nn.ConvTranspose2d(
            in_channels=64, out_channels=32,
            kernel_size=(1, 8),
            padding=(0, 0)
        )
        self.decoder_bn2 = nn.BatchNorm2d(32)
        self.decoder_dropout2 = nn.Dropout(0.1)

        # Decoder Layer 1: Final reconstruction layer
        self.decoder_conv1 = nn.ConvTranspose2d(
            in_channels=32, out_channels=1,
            kernel_size=(self.input_channels, 1),
            padding=(0, 0)
        )
        self.decoder_bn1 = nn.BatchNorm2d(1)

    def _initialize_weights(self) -> None:
        """Initialize network weights using Xavier/Glorot initialization."""
        for module in self.modules():
            if isinstance(module, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear)):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode input EEG signal into latent representation.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, channels, time_steps)
                            or (channels, time_steps) or (batch_size, time_steps)

        Returns:
            torch.Tensor: Latent representation of shape (batch_size, latent_dim)

        Raises:
            ValueError: If input tensor has invalid dimensions
        """
        # Validate and reshape input to 4D tensor for Conv2D
        x = self._validate_and_reshape_input(x)

        # Encoder Layer 1: Spatial convolution across EEG channels
        x = self.encoder_conv1(x)
        x = self.encoder_bn1(x)
        x = F.leaky_relu(x, negative_slope=self.leaky_relu_slope)
        x = self.encoder_dropout1(x)

        # Encoder Layer 2: Temporal convolution with pooling
        x = self.encoder_conv2(x)
        x = self.encoder_bn2(x)
        x = F.leaky_relu(x, negative_slope=self.leaky_relu_slope)
        x = self.encoder_pool2(x)
        x = self.encoder_dropout2(x)

        # Encoder Layer 3: Deep temporal features with pooling
        x = self.encoder_conv3(x)
        x = self.encoder_bn3(x)
        x = F.leaky_relu(x, negative_slope=self.leaky_relu_slope)
        x = self.encoder_pool3(x)
        x = self.encoder_dropout3(x)

        # Flatten and project to latent space
        x = x.view(x.size(0), -1)
        x = self.fc_encoder(x)

        return x

    def _validate_and_reshape_input(self, x: torch.Tensor) -> torch.Tensor:
        """
        Validate input dimensions and reshape to 4D tensor for Conv2D operations.

        Args:
            x (torch.Tensor): Input tensor

        Returns:
            torch.Tensor: Reshaped tensor of shape (batch_size, 1, channels, time_steps)

        Raises:
            ValueError: If input tensor has invalid dimensions
        """
        if len(x.shape) == 2:
            # Shape: (channels, time_steps) -> (1, 1, channels, time_steps)
            x = x.unsqueeze(0).unsqueeze(0)
        elif len(x.shape) == 3:
            # Shape: (batch_size, channels, time_steps) -> (batch_size, 1, channels, time_steps)
            x = x.unsqueeze(1)
        elif len(x.shape) == 4:
            # Already correct shape
            pass
        else:
            raise ValueError(
                f"Invalid input shape: {x.shape}. Expected 2D, 3D, or 4D tensor.")

        # Validate dimensions
        if x.size(2) != self.input_channels:
            raise ValueError(
                f"Expected {self.input_channels} channels, got {x.size(2)}. "
                f"Input shape: {x.shape}"
            )

        return x

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """
        Decode latent representation back to EEG signal.

        Args:
            z (torch.Tensor): Latent representation of shape (batch_size, latent_dim)

        Returns:
            torch.Tensor: Reconstructed EEG signal of shape (batch_size, 1, channels, time_steps)
        """
        # Project latent representation back to feature map space
        x = self.fc_decoder(z)
        x = self.decoder_bn_latent(x)
        x = F.leaky_relu(x, negative_slope=self.leaky_relu_slope)

        # Reshape back to 4D tensor for transpose convolutions
        x = x.view(x.size(0), *self.decoder_reshape_size)

        # Decoder Layer 3: Upsample and transpose convolution
        x = self.decoder_upsample3(x)
        x = self.decoder_conv3(x)
        x = self.decoder_bn3(x)
        x = F.leaky_relu(x, negative_slope=self.leaky_relu_slope)
        x = self.decoder_dropout3(x)

        # Decoder Layer 2: Upsample and transpose convolution
        x = self.decoder_upsample2(x)
        x = self.decoder_conv2(x)
        x = self.decoder_bn2(x)
        x = F.leaky_relu(x, negative_slope=self.leaky_relu_slope)
        x = self.decoder_dropout2(x)

        # Decoder Layer 1: Final reconstruction
        x = self.decoder_conv1(x)
        x = self.decoder_bn1(x)

        # Apply tanh activation to constrain output to [-1, 1]
        x = torch.tanh(x)

        return x

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through the autoencoder.

        Args:
            x (torch.Tensor): Input EEG signal

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Reconstructed signal and latent representation
        """
        latent_representation = self.encode(x)
        reconstructed_signal = self.decode(latent_representation)

        return reconstructed_signal, latent_representation

    def get_latent_representation(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract latent representation without reconstruction.

        Args:
            x (torch.Tensor): Input EEG signal

        Returns:
            torch.Tensor: Latent representation
        """
        return self.encode(x)

    def reconstruct(self, x: torch.Tensor) -> torch.Tensor:
        """
        Reconstruct input signal through encode-decode cycle.

        Args:
            x (torch.Tensor): Input EEG signal

        Returns:
            torch.Tensor: Reconstructed signal
        """
        latent = self.encode(x)
        return self.decode(latent)

    def get_model_info(self) -> dict:
        """
        Get model configuration and parameter information.

        Returns:
            dict: Model information including architecture details and parameter count
        """
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel()
                               for p in self.parameters() if p.requires_grad)

        return {
            'model_name': 'DeepConvolutionalAutoencoder',
            'input_channels': self.input_channels,
            'time_steps': self.time_steps,
            'latent_dim': self.latent_dim,
            'leaky_relu_slope': self.leaky_relu_slope,
            'dropout_rate': self.dropout_rate,
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'conv_output_size': self.conv_output_size,
            'architecture': {
                'encoder_layers': 3,
                'decoder_layers': 3,
                'activation': 'LeakyReLU',
                'normalization': 'BatchNorm2d',
                'regularization': 'Dropout'
            }
        }
