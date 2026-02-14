# Adapted from X-Codec-2.0: https://github.com/zhenye234/xcodec2
# ISTFT-based waveform synthesis head for direct waveform generation

import torch
import torch.nn as nn
import torch.nn.functional as F


class SnakeBeta(nn.Module):
    """
    Snake activation with separate learnable parameters for frequency (alpha) and magnitude (beta).

    SnakeBeta(x) = x + (1/β) * sin²(αx)

    This periodic activation function is well-suited for audio synthesis as it can learn
    to generate periodic patterns at different frequencies and magnitudes.

    Adapted from X-Codec-2.0, originally from https://arxiv.org/abs/2006.08195

    Args:
        channels: Number of input channels.
        alpha_logscale: If True, alpha and beta are stored in log scale for more stable training.
    """

    def __init__(self, channels: int, alpha_logscale: bool = True):
        super().__init__()
        self.channels = channels
        self.alpha_logscale = alpha_logscale

        # Initialize alpha (frequency) and beta (magnitude)
        if alpha_logscale:
            # Log scale: initialized to zeros (exp(0) = 1)
            self.alpha = nn.Parameter(torch.zeros(channels))
            self.beta = nn.Parameter(torch.zeros(channels))
        else:
            # Linear scale: initialized to ones
            self.alpha = nn.Parameter(torch.ones(channels))
            self.beta = nn.Parameter(torch.ones(channels))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor of shape (B, C, T) where C is channels.

        Returns:
            Activated tensor of the same shape.
        """
        # Reshape for broadcasting: (C,) -> (1, C, 1)
        alpha = self.alpha.unsqueeze(0).unsqueeze(-1)
        beta = self.beta.unsqueeze(0).unsqueeze(-1)

        if self.alpha_logscale:
            alpha = torch.exp(alpha)
            beta = torch.exp(beta)

        # SnakeBeta: x + (1/β) * sin²(αx)
        return x + (1.0 / (beta + 1e-9)) * torch.sin(alpha * x).pow(2)


class ISTFT(nn.Module):
    """
    Custom implementation of ISTFT with "same" padding support.

    Standard torch.istft doesn't allow custom padding (other than center=True) with
    windowing due to NOLA (Nonzero Overlap Add) check failures at the edges.
    This implementation supports "same" padding for neural vocoding where we want
    consistent input/output lengths.

    Args:
        n_fft: Size of Fourier transform.
        hop_length: The distance between neighboring sliding window frames.
        win_length: The size of window frame and STFT filter.
        padding: Type of padding. Options are "center" or "same".
    """

    def __init__(self, n_fft: int, hop_length: int, win_length: int, padding: str = "same"):
        super().__init__()
        if padding not in ["center", "same"]:
            raise ValueError("Padding must be 'center' or 'same'.")
        self.padding = padding
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        window = torch.hann_window(win_length)
        self.register_buffer("window", window)

    def forward(self, spec: torch.Tensor) -> torch.Tensor:
        """
        Compute the Inverse Short Time Fourier Transform (ISTFT) of a complex spectrogram.

        Args:
            spec: Input complex spectrogram of shape (B, N, T), where B is the batch size,
                  N is the number of frequency bins, and T is the number of time frames.

        Returns:
            Reconstructed time-domain signal of shape (B, L), where L is the output signal length.
        """
        if self.padding == "center":
            # Fallback to pytorch native implementation
            return torch.istft(spec, self.n_fft, self.hop_length, self.win_length, self.window, center=True)
        elif self.padding == "same":
            pad = (self.win_length - self.hop_length) // 2
        else:
            raise ValueError("Padding must be 'center' or 'same'.")

        assert spec.dim() == 3, "Expected a 3D tensor as input"
        B, N, T = spec.shape

        # Inverse FFT
        ifft = torch.fft.irfft(spec, self.n_fft, dim=1, norm="backward")
        ifft = ifft * self.window[None, :, None]

        # Overlap and Add
        output_size = (T - 1) * self.hop_length + self.win_length
        y = F.fold(
            ifft,
            output_size=(1, output_size),
            kernel_size=(1, self.win_length),
            stride=(1, self.hop_length),
        )[:, 0, 0, pad:-pad]

        # Window envelope
        window_sq = self.window.square().expand(1, T, -1).transpose(1, 2)
        window_envelope = F.fold(
            window_sq,
            output_size=(1, output_size),
            kernel_size=(1, self.win_length),
            stride=(1, self.hop_length),
        ).squeeze()[pad:-pad]

        # Normalize
        assert (window_envelope > 1e-11).all()
        y = y / window_envelope

        return y


class ISTFTHead(nn.Module):
    """
    ISTFT Head module for predicting STFT complex coefficients and synthesizing waveform.

    This module takes a hidden representation and predicts magnitude and phase components
    of the STFT, then uses ISTFT to synthesize the waveform.

    Args:
        dim: Hidden dimension of the input features.
        n_fft: Size of Fourier transform.
        hop_length: The distance between neighboring sliding window frames.
        padding: Type of padding for ISTFT. Options are "center" or "same".
    """

    def __init__(self, dim: int, n_fft: int, hop_length: int, padding: str = "same"):
        super().__init__()
        # Output dimension: (n_fft // 2 + 1) * 2 = n_fft + 2
        # First half for magnitude, second half for phase
        out_dim = n_fft + 2
        self.out = nn.Linear(dim, out_dim)
        self.istft = ISTFT(n_fft=n_fft, hop_length=hop_length, win_length=n_fft, padding=padding)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the ISTFTHead module.

        Args:
            x: Input tensor of shape (B, L, H), where B is the batch size,
               L is the sequence length, and H denotes the model dimension.

        Returns:
            Reconstructed time-domain audio signal of shape (B, T), where T is the output signal length.
        """
        x = self.out(x)
        x = x.transpose(1, 2)  # (B, out_dim, L)
        mag, phase = x.chunk(2, dim=1)

        # Convert log-magnitude to magnitude
        mag = torch.exp(mag)
        mag = torch.clamp(mag, max=1e2)  # Safeguard to prevent excessively large magnitudes

        # Construct complex spectrogram from magnitude and phase
        # Using cos and sin directly is more stable than exp(1j * phase)
        real = mag * torch.cos(phase)
        imag = mag * torch.sin(phase)
        spec = torch.complex(real, imag)

        # ISTFT to waveform
        audio = self.istft(spec)
        return audio


class ResNetBlock(nn.Module):
    """
    ResNet block with GroupNorm for 1D convolutions.

    This block is used before and after the Transformer in the wave decoder
    to capture local patterns (e.g., consonants, transients).

    Args:
        channels: Number of input and output channels.
        kernel_size: Kernel size for convolutions.
        dilation: Dilation factor for convolutions.
        num_groups: Number of groups for GroupNorm.
        dropout: Dropout probability.
    """

    def __init__(
        self,
        channels: int,
        kernel_size: int = 3,
        dilation: int = 1,
        num_groups: int = 32,
        dropout: float = 0.1,
    ):
        super().__init__()
        padding = (kernel_size - 1) * dilation // 2

        self.norm1 = nn.GroupNorm(num_groups=num_groups, num_channels=channels, eps=1e-6)
        self.conv1 = nn.Conv1d(channels, channels, kernel_size=kernel_size, padding=padding, dilation=dilation)
        self.norm2 = nn.GroupNorm(num_groups=num_groups, num_channels=channels, eps=1e-6)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size=kernel_size, padding=padding, dilation=dilation)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the ResNetBlock.

        Args:
            x: Input tensor of shape (B, C, L) where C is channels and L is sequence length.

        Returns:
            Output tensor of shape (B, C, L).
        """
        residual = x

        x = self.norm1(x)
        x = F.silu(x)
        x = self.conv1(x)

        x = self.norm2(x)
        x = F.silu(x)
        x = self.dropout(x)
        x = self.conv2(x)

        return x + residual


class ResNetStack(nn.Module):
    """
    Stack of ResNet blocks for local pattern processing.

    Args:
        channels: Number of channels.
        num_blocks: Number of ResNet blocks.
        kernel_size: Kernel size for convolutions.
        num_groups: Number of groups for GroupNorm.
        dropout: Dropout probability.
    """

    def __init__(
        self,
        channels: int,
        num_blocks: int = 2,
        kernel_size: int = 3,
        num_groups: int = 32,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.blocks = nn.ModuleList(
            [ResNetBlock(channels, kernel_size, num_groups=num_groups, dropout=dropout) for _ in range(num_blocks)]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through all ResNet blocks.

        Args:
            x: Input tensor of shape (B, C, L).

        Returns:
            Output tensor of shape (B, C, L).
        """
        for block in self.blocks:
            x = block(x)
        return x


class UpSamplerBlock(nn.Module):
    """
    Upsampler block using transposed convolutions, SnakeBeta activations, and ResNet blocks.

    This module upsamples feature embeddings in the time dimension using a series of
    ConvTranspose1d layers, each followed by SnakeBeta activation and a ResNet block.
    Adapted from X-Codec-2.0.

    The upsampling is used to increase the frame rate before ISTFT, enabling
    higher sample rate output (e.g., 44.1kHz) from lower frame rate features.

    Args:
        in_channels: Number of input channels.
        upsample_factors: List of upsampling factors for each stage.
            Total upsampling = product of all factors (e.g., [3, 3] -> 9x).
        kernel_sizes: List of kernel sizes for each ConvTranspose1d.
            If None, defaults to factor * 2 for each stage.
        num_groups: Number of groups for GroupNorm in ResNet blocks.
    """

    def __init__(
        self,
        in_channels: int,
        upsample_factors: list[int],
        kernel_sizes: list[int] | None = None,
        num_groups: int = 32,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.upsample_factors = list(upsample_factors)
        self.kernel_sizes = list(kernel_sizes or [f * 2 for f in self.upsample_factors])

        if len(self.kernel_sizes) != len(self.upsample_factors):
            raise ValueError("kernel_sizes and upsample_factors must have the same length")

        self.upsample_layers = nn.ModuleList()
        self.snake_activations = nn.ModuleList()
        self.resnet_blocks = nn.ModuleList()

        # Final projection and activation
        final_channels = self.in_channels // (2 ** len(self.upsample_factors))
        self.out_proj = nn.Linear(final_channels, self.in_channels, bias=True)
        self.out_snake = SnakeBeta(self.in_channels, alpha_logscale=True)

        for i, (k, u) in enumerate(zip(self.kernel_sizes, self.upsample_factors)):
            c_in = self.in_channels // (2**i)
            c_out = self.in_channels // (2 ** (i + 1))
            # ConvTranspose1d for upsampling
            self.upsample_layers.append(
                nn.utils.parametrizations.weight_norm(
                    nn.ConvTranspose1d(c_in, c_out, kernel_size=k, stride=u, padding=(k - u) // 2)
                )
            )
            # SnakeBeta activation after ConvTranspose
            self.snake_activations.append(SnakeBeta(c_out, alpha_logscale=True))
            # Adjust num_groups if c_out is smaller
            effective_groups = min(num_groups, c_out)
            self.resnet_blocks.append(
                ResNetBlock(channels=c_out, num_groups=effective_groups, dropout=0.0)
            )

    @property
    def total_upsample_factor(self) -> int:
        """Total upsampling factor (product of all stage factors)."""
        result = 1
        for f in self.upsample_factors:
            result *= f
        return result

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the UpSamplerBlock.

        Args:
            x: Input tensor of shape (B, C, L) where C is in_channels and L is sequence length.

        Returns:
            Output tensor of shape (B, L', C) where L' = L * total_upsample_factor.
            Note: Output is transposed to (B, L', C) for compatibility with ISTFTHead.
        """
        for up, snake, resblk in zip(self.upsample_layers, self.snake_activations, self.resnet_blocks):
            x = snake(up(x))  # ConvTranspose -> SnakeBeta
            x = resblk(x)     # ResNet block
        # Project back to original channel dimension, apply SnakeBeta, and transpose to (B, L', C)
        x = self.out_proj(x.transpose(1, 2))  # (B, L', C)
        x = self.out_snake(x.transpose(1, 2))  # (B, C, L') -> SnakeBeta -> (B, C, L')
        return x.transpose(1, 2)  # (B, L', C)
