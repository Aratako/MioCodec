import math
from contextlib import nullcontext
from dataclasses import dataclass

import jsonargparse
import torch
import torch.nn as nn
import torch.nn.functional as F

from .module.fsq import FiniteScalarQuantizer
from .module.global_encoder import GlobalEncoder
from .module.istft_head import ISTFTHead, ResNetStack, UpSamplerBlock
from .module.postnet import PostNet
from .module.ssl_extractor import SSLFeatureExtractor
from .module.transformer import Transformer
from .util import freeze_modules, get_logger

logger = get_logger()


def _get_autocast_context(device_type: str):
    """Get autocast context for the given device type.

    bfloat16 autocast is only enabled for CUDA devices because
    torch.complex does not support bfloat16 on CPU/MPS.
    """
    if device_type == "cuda":
        return torch.autocast(device_type=device_type, dtype=torch.bfloat16, enabled=True)
    return nullcontext()


@dataclass
class MioCodecModelConfig:
    # SSL Feature settings
    local_ssl_layers: tuple[int, ...] = (6, 9)  # Indices of SSL layers for local branch
    global_ssl_layers: tuple[int, ...] = (1, 2)  # Indices of SSL layers for global branch
    normalize_ssl_features: bool = True  # Whether to normalize local SSL features before encoding

    # Down/up-sampling settings
    downsample_factor: int = 2  # Temporal downsampling factor for local features
    mel_upsample_factor: int = 4  # Conv1DTranspose upsampling factor for mel features before interpolation
    use_conv_downsample: bool = True  # Whether to use Conv1D for downsampling instead average pooling
    local_interpolation_mode: str = "linear"  # Interpolation mode for local upsampling ("linear", "nearest")
    mel_interpolation_mode: str = "linear"  # Interpolation mode for mel upsampling ("linear", "nearest")

    # Mel spectrogram settings
    sample_rate: int = 24000
    n_fft: int = 1024
    hop_length: int = 256
    n_mels: int = 100
    padding: str = "center"
    mel_backend: str = "vocos"  # "vocos" (torchaudio) or "pupu" (librosa-compatible)
    mel_fmin: float = 0.0
    mel_fmax: float | None = None
    mel_win_length: int | None = None

    # Wave decoder settings (direct waveform synthesis without mel intermediate)
    use_wave_decoder: bool = False  # Whether to use direct wave decoder instead of mel decoder
    wave_upsample_factor: int = 4  # Conv1DTranspose upsampling factor for wave decoder
    wave_interpolation_mode: str = "linear"  # Interpolation mode for wave upsampling
    wave_decoder_dim: int = 512  # Hidden dimension for wave decoder
    wave_resnet_num_blocks: int = 2  # Number of ResNet blocks before/after transformer
    wave_resnet_kernel_size: int = 3  # Kernel size for ResNet blocks
    wave_resnet_num_groups: int = 32  # Number of groups for GroupNorm in ResNet
    wave_resnet_dropout: float = 0.1  # Dropout for ResNet blocks
    istft_padding: str = "same"  # Padding mode for ISTFT ("same" or "center")
    wave_upsampler_factors: tuple[int, ...] | None = None  # e.g., (3, 3) for 9x upsampling for 44.1kHz
    wave_upsampler_kernel_sizes: tuple[int, ...] | None = None  # Kernel sizes for each upsampler stage


@dataclass
class MioCodecFeatures:
    content_embedding: torch.Tensor | None = None  # (seq_len, dim)
    content_token_indices: torch.Tensor | None = None  # (seq_len,)
    global_embedding: torch.Tensor | None = None  # (dim,)


class MioCodecModel(nn.Module):
    """Model architecture and forward pass logic for MioCodec."""

    def __init__(
        self,
        config: MioCodecModelConfig,
        ssl_feature_extractor: SSLFeatureExtractor,
        local_encoder: Transformer,
        local_quantizer: FiniteScalarQuantizer,
        feature_decoder: Transformer | None,
        global_encoder: GlobalEncoder,
        mel_prenet: Transformer | None = None,
        mel_decoder: Transformer | None = None,
        mel_postnet: PostNet | None = None,
        wave_prenet: Transformer | None = None,
        wave_decoder: Transformer | None = None,
    ):
        super().__init__()
        self.config = config
        self._init_ssl_extractor(config, ssl_feature_extractor)
        self._init_local_branch(config, local_encoder, local_quantizer, feature_decoder)
        self._init_global_branch(global_encoder)

        # Initialize either mel decoder or wave decoder based on config
        if config.use_wave_decoder:
            self._init_wave_decoder(config, wave_prenet, wave_decoder)
            # Set mel decoder components to None
            self.mel_prenet = None
            self.mel_decoder = None
            self.mel_postnet = None
            self.mel_conv_upsample = None
        else:
            self._init_mel_decoder(config, mel_prenet, mel_decoder, mel_postnet)
            # Set wave decoder components to None
            self.wave_prenet = None
            self.wave_decoder = None
            self.wave_prior_net = None
            self.wave_post_net = None
            self.wave_conv_upsample = None
            self.wave_upsampler = None
            self.istft_head = None

    def _init_ssl_extractor(self, config: MioCodecModelConfig, ssl_feature_extractor: SSLFeatureExtractor):
        """Initialize and configure SSL feature extractor."""
        self.ssl_feature_extractor = ssl_feature_extractor
        freeze_modules([self.ssl_feature_extractor])
        logger.debug(
            f"SSL feature extractor initialized and frozen, feature dim: {self.ssl_feature_extractor.feature_dim}"
        )

        # Configure local SSL layers
        self.local_ssl_layers = list(config.local_ssl_layers)
        if len(self.local_ssl_layers) > 1:
            logger.debug(
                f"Using average of {len(self.local_ssl_layers)} SSL layers for local branch: {self.local_ssl_layers}"
            )
        else:
            logger.debug(f"Using single SSL layer {self.local_ssl_layers[0]} for local branch")

        if config.normalize_ssl_features:
            logger.debug("Normalizing local SSL features before encoding")

        # Configure global SSL layers
        self.global_ssl_layers = list(config.global_ssl_layers)
        if len(self.global_ssl_layers) > 1:
            logger.debug(
                f"Using average of {len(self.global_ssl_layers)} SSL layers for global branch: {self.global_ssl_layers}"
            )
        else:
            logger.debug(f"Using single SSL layer {self.global_ssl_layers[0]} for global branch")

    def _init_local_branch(
        self,
        config: MioCodecModelConfig,
        local_encoder: Transformer,
        local_quantizer: FiniteScalarQuantizer,
        feature_decoder: Transformer | None,
    ):
        """Initialize local branch components (encoder, downsampling, quantizer, decoder)."""
        self.local_encoder = local_encoder
        self.local_quantizer = local_quantizer
        self.feature_decoder = feature_decoder

        # Configure downsampling
        self.downsample_factor = config.downsample_factor
        if self.downsample_factor > 1:
            logger.debug(f"Using temporal downsampling with factor {self.downsample_factor}")
            if config.use_conv_downsample:
                # Create Conv1d layers for downsampling and upsampling local embeddings
                feature_dim = local_encoder.output_dim
                self.conv_downsample = nn.Conv1d(
                    feature_dim, feature_dim, kernel_size=config.downsample_factor, stride=config.downsample_factor
                )
                self.conv_upsample = nn.ConvTranspose1d(
                    feature_dim, feature_dim, kernel_size=config.downsample_factor, stride=config.downsample_factor
                )  # won't be used unless training feature reconstruction
                logger.debug(f"Using Conv1d downsampling/upsampling with kernel size {config.downsample_factor}")
            else:
                self.conv_downsample = None
                self.conv_upsample = None
                logger.debug("Using average pooling and linear interpolation for downsampling/upsampling")
        else:
            self.conv_downsample = None
            self.conv_upsample = None

    def _init_global_branch(self, global_encoder: GlobalEncoder):
        """Initialize global branch components."""
        self.global_encoder = global_encoder

    def _init_mel_decoder(
        self, config: MioCodecModelConfig, mel_prenet: Transformer, mel_decoder: Transformer, mel_postnet: PostNet
    ):
        """Initialize mel decoder components (prenet, upsampling, decoder, postnet)."""
        self.mel_prenet = mel_prenet
        self.mel_decoder = mel_decoder
        self.mel_postnet = mel_postnet

        # Configure mel upsampling
        self.mel_conv_upsample = None
        if config.mel_upsample_factor > 1:
            # Create Conv1DTranspose layer for mel upsampling
            input_dim = mel_prenet.output_dim
            self.mel_conv_upsample = nn.ConvTranspose1d(
                input_dim, input_dim, kernel_size=config.mel_upsample_factor, stride=config.mel_upsample_factor
            )
            logger.debug(f"Using Conv1DTranspose for mel upsampling with factor {config.mel_upsample_factor}")

    def _init_wave_decoder(
        self, config: MioCodecModelConfig, wave_prenet: Transformer, wave_decoder: Transformer
    ):
        """Initialize wave decoder components for direct waveform synthesis.

        Architecture:
            content_tokens -> wave_prenet -> conv_upsample -> interpolate
            -> wave_prior_net (ResNet) -> wave_decoder (Transformer with AdaLN-Zero)
            -> wave_post_net (ResNet) -> istft_head -> waveform
        """
        self.wave_prenet = wave_prenet
        self.wave_decoder = wave_decoder

        wave_dim = config.wave_decoder_dim

        # Configure wave upsampling
        self.wave_conv_upsample = None
        if config.wave_upsample_factor > 1:
            input_dim = wave_prenet.output_dim
            self.wave_conv_upsample = nn.ConvTranspose1d(
                input_dim, input_dim, kernel_size=config.wave_upsample_factor, stride=config.wave_upsample_factor
            )
            logger.debug(f"Using Conv1DTranspose for wave upsampling with factor {config.wave_upsample_factor}")

        # ResNet blocks before transformer (prior_net)
        self.wave_prior_net = ResNetStack(
            channels=wave_dim,
            num_blocks=config.wave_resnet_num_blocks,
            kernel_size=config.wave_resnet_kernel_size,
            num_groups=config.wave_resnet_num_groups,
            dropout=config.wave_resnet_dropout,
        )
        logger.debug(f"Wave prior_net: {config.wave_resnet_num_blocks} ResNet blocks")

        # ResNet blocks after transformer (post_net)
        self.wave_post_net = ResNetStack(
            channels=wave_dim,
            num_blocks=config.wave_resnet_num_blocks,
            kernel_size=config.wave_resnet_kernel_size,
            num_groups=config.wave_resnet_num_groups,
            dropout=config.wave_resnet_dropout,
        )
        logger.debug(f"Wave post_net: {config.wave_resnet_num_blocks} ResNet blocks")

        # UpSampler for higher sample rates (e.g., 44.1kHz)
        self.wave_upsampler = None
        if config.wave_upsampler_factors:
            upsample_factors = list(config.wave_upsampler_factors)
            kernel_sizes = list(config.wave_upsampler_kernel_sizes) if config.wave_upsampler_kernel_sizes else None
            self.wave_upsampler = UpSamplerBlock(
                in_channels=wave_dim,
                upsample_factors=upsample_factors,
                kernel_sizes=kernel_sizes,
                num_groups=config.wave_resnet_num_groups,
            )
            logger.debug(
                f"Wave upsampler initialized: factors={upsample_factors}, "
                f"total={self.wave_upsampler.total_upsample_factor}x"
            )

        # ISTFT head for waveform synthesis
        self.istft_head = ISTFTHead(
            dim=wave_dim,
            n_fft=config.n_fft,
            hop_length=config.hop_length,
            padding=config.istft_padding,
        )
        logger.debug(
            f"ISTFTHead initialized: dim={wave_dim}, n_fft={config.n_fft}, "
            f"hop_length={config.hop_length}, padding={config.istft_padding}"
        )

    def _calculate_waveform_padding(self, audio_length: int, ensure_recon_length: bool = False) -> int:
        """Calculate required padding for input waveform to ensure consistent SSL feature lengths."""
        extractor = self.ssl_feature_extractor
        sample_rate = self.config.sample_rate
        # SSL may resample the input to its own sample rate, so calculate the number of samples after resampling
        num_samples_after_resampling = audio_length / sample_rate * extractor.ssl_sample_rate
        # We expect the SSL feature extractor to be consistent with its hop size
        expected_ssl_output_length = math.ceil(num_samples_after_resampling / extractor.hop_size)
        # If ensure_recon_length is True, we want to make sure the output length is exactly divisible by downsample factor
        if ensure_recon_length and (remainder := expected_ssl_output_length % self.downsample_factor) != 0:
            expected_ssl_output_length += self.downsample_factor - remainder
        # But it may require more input samples to produce that output length, so calculate the required input length
        num_samples_required_after_resampling = extractor.get_minimum_input_length(expected_ssl_output_length)
        # That number of samples is at the SSL sample rate, so convert back to our original sample rate
        num_samples_required = num_samples_required_after_resampling / extractor.ssl_sample_rate * sample_rate
        # Calculate padding needed on each side
        padding = math.ceil((num_samples_required - audio_length) / 2)
        return padding

    def _calculate_original_audio_length(self, token_length: int) -> int:
        """Calculate the original audio length based on token length."""
        extractor = self.ssl_feature_extractor
        sample_rate = self.config.sample_rate
        # Calculate the feature length before downsampling
        feature_length = token_length * self.downsample_factor
        num_samples_required_after_resampling = extractor.get_minimum_input_length(feature_length)
        num_samples_required = num_samples_required_after_resampling / extractor.ssl_sample_rate * sample_rate
        return math.ceil(num_samples_required)

    def _calculate_target_mel_length(self, audio_length: int) -> int:
        """Calculate the target mel spectrogram length based on audio length."""
        if self.config.mel_backend == "pupu":
            # Pupu mel uses reflect padding and center=False STFT, yielding floor(audio_length / hop_length)
            return audio_length // self.config.hop_length
        if self.config.padding == "center":
            return audio_length // self.config.hop_length + 1
        elif self.config.padding == "same":
            return audio_length // self.config.hop_length
        else:
            return (audio_length - self.config.n_fft) // self.config.hop_length + 1

    def _calculate_target_stft_length(self, audio_length: int) -> int:
        """Calculate the target STFT frame length based on audio length for wave decoder.

        When wave_upsampler is configured, this returns the frame length BEFORE upsampling
        (i.e., the length that wave_post_net should output). The upsampler will then
        expand it to the actual STFT frame length.
        """
        if self.config.istft_padding == "same":
            istft_frames = audio_length // self.config.hop_length
        else:  # center
            istft_frames = audio_length // self.config.hop_length + 1

        # If upsampler is configured, return the length before upsampling
        if self.wave_upsampler is not None:
            # istft_frames = pre_upsample_length * total_upsample_factor
            return istft_frames // self.wave_upsampler.total_upsample_factor

        return istft_frames

    def _process_ssl_features(self, features: list[torch.Tensor], layers: list[int]) -> torch.Tensor:
        if len(layers) > 1:
            # Get features from multiple layers and average them
            selected_features = [features[i - 1] for i in layers]
            mixed_features = torch.stack(selected_features, dim=0).mean(dim=0)
        else:
            # Just take the single specified layer
            mixed_features = features[layers[0] - 1]
        return mixed_features

    def _normalize_ssl_features(self, features: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
        if not self.config.normalize_ssl_features:
            return features

        # Compute mean and std across time steps for each sample and feature dimension
        mean = torch.mean(features, dim=1, keepdim=True)  # (B, 1, C)
        std = torch.std(features, dim=1, keepdim=True)  # (B, 1, C)
        return (features - mean) / (std + eps)

    def forward_ssl_features(
        self, waveform: torch.Tensor, padding: int | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass to extract SSL features. (B, T, C)
        Args:
            waveform: Input waveform tensor of shape (B, channels, samples)
            padding: Optional padding to apply on both sides of the waveform. This is useful to ensure
                     that the SSL feature extractor produces consistent output lengths.
        Returns:
            local_ssl_features: Local SSL features for local branch. (B, T, C)
            global_ssl_features: Global SSL features for global branch. (B, T, C)
        """
        # Prepare input waveform
        if waveform.dim() == 3:
            waveform = waveform.squeeze(1)

        # 1. Extract SSL features
        if padding and padding > 0:
            waveform = F.pad(waveform, (padding, padding), mode="constant")

        with torch.no_grad():
            ssl_features = self.ssl_feature_extractor(waveform)

        local_ssl_features = self._process_ssl_features(ssl_features, self.local_ssl_layers)
        local_ssl_features = self._normalize_ssl_features(local_ssl_features)

        global_ssl_features = self._process_ssl_features(ssl_features, self.global_ssl_layers)

        return local_ssl_features, global_ssl_features

    def forward_content(
        self, local_ssl_features: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor] | None:
        """Forward pass to extract content embeddings from the local branch.
        Args:
            local_ssl_features: Local SSL features tensor of shape (B, T, C)
        Returns:
            local_quantized: Quantized local embeddings. (B, T/factor, C)
            indices: Content token indices. (B, T/factor)
            ssl_recon: Reconstructed SSL features (if feature decoder is present). (B, T, C)
            perplexity: Quantizer perplexity (if feature decoder is present). Scalar tensor.
        """
        local_encoded = self.local_encoder(local_ssl_features)

        # Downsample temporally if needed: (B, T, C) -> (B, T/factor, C)
        if self.downsample_factor > 1:
            if self.config.use_conv_downsample:
                local_encoded = self.conv_downsample(local_encoded.transpose(1, 2)).transpose(1, 2)
            else:
                local_encoded = F.avg_pool1d(
                    local_encoded.transpose(1, 2), kernel_size=self.downsample_factor, stride=self.downsample_factor
                ).transpose(1, 2)

        # If training feature reconstruction, decode local embeddings
        ssl_recon = None
        perplexity = torch.tensor(0.0)
        if self.feature_decoder is not None:
            local_quantized, local_quantize_info = self.local_quantizer(local_encoded)
            indices = local_quantize_info["indices"]
            perplexity = torch.mean(local_quantize_info["perplexity"])

            local_latent_for_ssl = local_quantized
            # Upsample if needed
            if self.downsample_factor > 1:
                if self.config.use_conv_downsample:
                    # Use conv transpose for upsampling: (B, T/factor, C) -> (B, C, T/factor) -> conv -> (B, C, T) -> (B, T, C)
                    local_latent_for_ssl = self.conv_upsample(local_latent_for_ssl.transpose(1, 2)).transpose(1, 2)
                else:
                    # (B, T/factor, C) -> (B, T, C)
                    local_latent_for_ssl = F.interpolate(
                        local_latent_for_ssl.transpose(1, 2),
                        size=local_ssl_features.shape[1],
                        mode=self.config.local_interpolation_mode,
                    ).transpose(1, 2)

            ssl_recon = self.feature_decoder(local_latent_for_ssl)
        else:
            # If not training feature reconstruction, just get quantized local embeddings
            local_quantized, indices = self.local_quantizer.encode(local_encoded)

        return local_quantized, indices, ssl_recon, perplexity

    def forward_global(self, global_ssl_features: torch.Tensor) -> torch.Tensor:
        """Forward pass to extract global embeddings from the global branch.
        Args:
            global_ssl_features: Global SSL features tensor of shape (B, T, C)
        Returns:
            global_encoded: Global embeddings. (B, C)
        """
        global_encoded = self.global_encoder(global_ssl_features)
        return global_encoded

    def forward_mel(
        self, content_embeddings: torch.Tensor, global_embeddings: torch.Tensor, mel_length: int
    ) -> torch.Tensor:
        """Forward pass to generate mel spectrogram from content and global embeddings.
        Args:
            content_embeddings: Content embeddings tensor of shape (B, T, C)
            global_embeddings: Global embeddings tensor of shape (B, C)
            mel_length: Target mel spectrogram length (T_mel)
        Returns:
            mel_recon: Reconstructed mel spectrogram tensor of shape (B, n_mels, T_mel)
        """
        local_latent = self.mel_prenet(content_embeddings)

        # Upsample local latent to match mel spectrogram length
        # First use Conv1DTranspose if configured
        if self.mel_conv_upsample is not None:
            # (B, T/factor, C) -> (B, C, T/factor) -> conv -> (B, C, T*upsample_factor) -> (B, T*upsample_factor, C)
            local_latent = self.mel_conv_upsample(local_latent.transpose(1, 2)).transpose(1, 2)
        local_latent = F.interpolate(
            local_latent.transpose(1, 2), size=mel_length, mode=self.config.mel_interpolation_mode
        ).transpose(1, 2)  # (B, T_current, C) -> (B, T_mel, C)

        # Generate mel spectrogram, conditioned on global embeddings
        mel_recon = self.mel_decoder(local_latent, condition=global_embeddings.unsqueeze(1))
        mel_recon = mel_recon.transpose(1, 2)  # (B, n_mels, T)

        mel_recon = self.mel_postnet(mel_recon)
        return mel_recon

    def forward_wave(
        self, content_embeddings: torch.Tensor, global_embeddings: torch.Tensor, stft_length: int
    ) -> torch.Tensor:
        """Forward pass to generate waveform directly from content and global embeddings.

        This method bypasses mel spectrogram prediction and directly generates waveform
        using ISTFT-based synthesis.

        Args:
            content_embeddings: Content embeddings tensor of shape (B, T, C)
            global_embeddings: Global embeddings tensor of shape (B, C)
            stft_length: Target STFT frame length (T_stft)
        Returns:
            waveform: Reconstructed waveform tensor of shape (B, samples)
        """
        # Prenet transformation
        local_latent = self.wave_prenet(content_embeddings)

        # Upsample to match STFT frame rate
        # First use Conv1DTranspose if configured
        if self.wave_conv_upsample is not None:
            # (B, T, C) -> (B, C, T) -> conv -> (B, C, T*factor) -> (B, T*factor, C)
            local_latent = self.wave_conv_upsample(local_latent.transpose(1, 2)).transpose(1, 2)

        # Interpolate to exact STFT length
        local_latent = F.interpolate(
            local_latent.transpose(1, 2), size=stft_length, mode=self.config.wave_interpolation_mode
        ).transpose(1, 2)  # (B, T_stft, C)

        # Prior ResNet blocks (local pattern processing)
        # ResNet expects (B, C, T), so transpose
        local_latent = self.wave_prior_net(local_latent.transpose(1, 2)).transpose(1, 2)

        # Transformer decoder with speaker conditioning via AdaLN-Zero
        local_latent = self.wave_decoder(local_latent, condition=global_embeddings.unsqueeze(1))

        # Post ResNet blocks
        local_latent = self.wave_post_net(local_latent.transpose(1, 2)).transpose(1, 2)

        # Apply upsampler if configured (for higher sample rates like 44.1kHz)
        if self.wave_upsampler is not None:
            # wave_upsampler expects (B, C, L) and outputs (B, L', C)
            local_latent = self.wave_upsampler(local_latent.transpose(1, 2))

        # ISTFT head: predict magnitude + phase and synthesize waveform
        waveform = self.istft_head(local_latent)  # (B, samples)

        return waveform

    # ======== Inference methods ========

    def weights_to_save(self, *, include_modules: list[str]) -> dict[str, torch.Tensor]:
        """Get model weights for saving. Excludes certain modules not needed for inference."""
        excluded_modules = [
            m
            for m in ["ssl_feature_extractor", "feature_decoder", "conv_upsample"]
            if m not in include_modules
        ]
        state_dict = {
            name: param
            for name, param in self.named_parameters()
            if not any(name.startswith(excl) for excl in excluded_modules)
        }
        return state_dict

    @classmethod
    def from_hparams(cls, config_path: str) -> "MioCodecModel":
        """Instantiate MioCodecModel from config file.
        Args:
            config_path (str): Path to model configuration file (.yaml).
        Returns:
            MioCodecModel: Instantiated MioCodecModel.
        """
        parser = jsonargparse.ArgumentParser(exit_on_error=False)
        parser.add_argument("--model", type=MioCodecModel)
        cfg = parser.parse_path(config_path)
        cfg = parser.instantiate_classes(cfg)
        return cfg.model

    @classmethod
    def from_pretrained(
        cls,
        repo_id: str | None = None,
        revision: str | None = None,
        config_path: str | None = None,
        weights_path: str | None = None,
    ) -> "MioCodecModel":
        """Load MioCodecModel either from HuggingFace Hub or local config and weights files.
        Args:
            repo_id (str, optional): HuggingFace Hub repository ID. If provided, loads config and weights from the hub.
            revision (str, optional): Revision (branch, tag, commit) for the HuggingFace Hub repo.
            config_path (str, optional): Path to model configuration file (.yaml). Required if repo_id is not provided.
            weights_path (str, optional): Path to model weights file (.safetensors). Required if repo_id is not provided.
        Returns:
            MioCodecModel: Loaded MioCodecModel instance.
        """
        if repo_id is not None:
            # Load from HuggingFace Hub
            from huggingface_hub import hf_hub_download

            config_path = hf_hub_download(repo_id, "config.yaml", revision=revision)
            weights_path = hf_hub_download(repo_id, "model.safetensors", revision=revision)
        else:
            # Check local paths
            if config_path is None or weights_path is None:
                raise ValueError(
                    "Please provide either HuggingFace Hub repo_id or both config_path and weights_path for model loading."
                )

        # Load model from config
        model = cls.from_hparams(config_path)

        # Load weights
        from safetensors.torch import load_file

        state_dict = load_file(weights_path, device="cpu")
        model.load_state_dict(state_dict, strict=False)
        logger.info(f"Loaded weights from safetensors file: {weights_path}")

        return model

    @torch.inference_mode()
    def encode(self, waveform: torch.Tensor, return_content: bool = True, return_global: bool = True) -> MioCodecFeatures:
        """Extract content and/or global features from audio using MioCodec.
        Args:
            waveform (torch.Tensor): Input audio waveform tensor (samples,). The sample rate should match model config.
            return_content (bool): Whether to extract content features.
            return_global (bool): Whether to extract global features.
        Returns:
            dict[str, torch.Tensor]: Extracted features.
        """
        audio_length = waveform.size(0)
        padding = self._calculate_waveform_padding(audio_length)
        local_ssl_features, global_ssl_features = self.forward_ssl_features(waveform.unsqueeze(0), padding=padding)

        result = MioCodecFeatures()
        device_type = local_ssl_features.device.type
        with _get_autocast_context(device_type):
            if return_content:
                content_embedding, token_indices, _, _ = self.forward_content(local_ssl_features)
                result.content_embedding = content_embedding.squeeze(0)  # (seq_len, dim)
                result.content_token_indices = token_indices.squeeze(0)  # (seq_len,)

            if return_global:
                global_embedding = self.forward_global(global_ssl_features)
                result.global_embedding = global_embedding.squeeze(0)  # (dim,)

        return result

    def decode_token_indices(self, indices: torch.Tensor) -> torch.Tensor:
        """Get content embeddings from content token indices. (..., seq_len) -> (..., seq_len, dim)"""
        content_embedding = self.local_quantizer.decode(indices)
        return content_embedding

    @torch.inference_mode()
    def decode(
        self,
        global_embedding: torch.Tensor,
        content_token_indices: torch.Tensor | None = None,
        content_embedding: torch.Tensor | None = None,
        target_audio_length: int | None = None,
    ) -> torch.Tensor:
        """Synthesize audio from content and global features using MioCodec.

        If use_wave_decoder is True, returns waveform directly.
        Otherwise, returns mel spectrogram (requires external vocoder for waveform synthesis).

        Args:
            global_embedding (torch.Tensor): Global embedding tensor (dim,).
            content_token_indices (torch.Tensor, optional): Optional content token indices tensor (seq_len).
            content_embedding (torch.Tensor, optional): Optional content embedding tensor (seq_len, dim).
                If both content_token_indices and content_embedding are provided, content_embedding takes precedence.
            target_audio_length (int, optional): Target length of the output audio in samples.
                If None, uses the original audio length estimated from the sequence length of content tokens.
        Returns:
            torch.Tensor: If use_wave_decoder=True, returns waveform tensor (samples,).
                          Otherwise, returns mel spectrogram tensor (n_mels, T).
        """
        # Obtain content embedding if not provided
        if content_embedding is None:
            if content_token_indices is None:
                raise ValueError("Either content_token_indices or content_embedding must be provided.")
            content_embedding = self.decode_token_indices(content_token_indices)

        if target_audio_length is None:
            # Estimate original audio length from content token sequence length
            seq_len = content_embedding.size(0)
            target_audio_length = self._calculate_original_audio_length(seq_len)

        device_type = content_embedding.device.type
        with _get_autocast_context(device_type):
            content_embedding = content_embedding.unsqueeze(0)  # (1, seq_len, dim)
            global_embedding = global_embedding.unsqueeze(0)  # (1, dim)

            if self.config.use_wave_decoder:
                # Direct waveform synthesis
                stft_length = self._calculate_target_stft_length(target_audio_length)
                waveform = self.forward_wave(content_embedding, global_embedding, stft_length=stft_length)
                return waveform.squeeze(0)  # (samples,)
            else:
                # Mel spectrogram synthesis (requires external vocoder)
                mel_length = self._calculate_target_mel_length(target_audio_length)
                mel_spectrogram = self.forward_mel(content_embedding, global_embedding, mel_length=mel_length)
                return mel_spectrogram.squeeze(0)  # (n_mels, T)

    @torch.inference_mode()
    def decode_batch(
        self,
        global_embeddings: torch.Tensor,
        content_token_indices: torch.Tensor | None = None,
        content_embeddings: torch.Tensor | None = None,
        content_lengths: torch.Tensor | list[int] | None = None,
        target_audio_lengths: torch.Tensor | list[int] | None = None,
        padding_token_idx: int = 0,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Synthesize audio from batched content and global features using MioCodec.

        If use_wave_decoder is True, returns waveforms directly.
        Otherwise, returns mel spectrograms (requires external vocoder for waveform synthesis).

        Supports variable-length sequences via padding. Each sample in the batch can have
        different content lengths and target audio lengths.

        Args:
            global_embeddings (torch.Tensor): Global embedding tensor (B, dim).
            content_token_indices (torch.Tensor, optional): Content token indices tensor (B, max_seq_len).
                Padded with padding_token_idx for sequences shorter than max_seq_len.
            content_embeddings (torch.Tensor, optional): Content embedding tensor (B, max_seq_len, dim).
                If both content_token_indices and content_embeddings are provided, content_embeddings takes precedence.
            content_lengths (torch.Tensor | list[int], optional): Actual content length for each sample (B,).
                If None, assumes all samples have the same length (max_seq_len).
            target_audio_lengths (torch.Tensor | list[int], optional): Target audio length for each sample (B,).
                If None, uses the audio length estimated from content_lengths.
            padding_token_idx (int): Token index used for padding in content_token_indices.

        Returns:
            tuple[torch.Tensor, torch.Tensor]:
                If use_wave_decoder=True:
                    - waveforms: Generated waveform tensor (B, max_samples), padded to max length
                    - audio_lengths: Actual audio length for each sample (B,)
                Otherwise:
                    - mel_spectrograms: Generated mel spectrogram tensor (B, n_mels, max_mel_len), padded to max length
                    - mel_lengths: Actual mel length for each sample (B,)
        """
        # Obtain content embeddings if not provided
        if content_embeddings is None:
            if content_token_indices is None:
                raise ValueError("Either content_token_indices or content_embeddings must be provided.")
            content_embeddings = self.decode_token_indices(content_token_indices)  # (B, max_seq_len, dim)

        batch_size = content_embeddings.size(0)
        max_seq_len = content_embeddings.size(1)
        device = content_embeddings.device

        # Handle content_lengths
        if content_lengths is None:
            content_lengths = torch.full((batch_size,), max_seq_len, dtype=torch.long, device=device)
        elif isinstance(content_lengths, list):
            content_lengths = torch.tensor(content_lengths, dtype=torch.long, device=device)
        else:
            content_lengths = content_lengths.to(device)

        # Handle target_audio_lengths
        if target_audio_lengths is None:
            # Estimate from content lengths
            target_audio_lengths = torch.tensor(
                [self._calculate_original_audio_length(length.item()) for length in content_lengths],
                dtype=torch.long,
                device=device,
            )
        elif isinstance(target_audio_lengths, list):
            target_audio_lengths = torch.tensor(target_audio_lengths, dtype=torch.long, device=device)
        else:
            target_audio_lengths = target_audio_lengths.to(device)

        if self.config.use_wave_decoder:
            # Wave decoder path: generate waveforms directly
            return self._decode_batch_wave(
                content_embeddings, global_embeddings, content_lengths, target_audio_lengths
            )
        else:
            # Mel decoder path: generate mel spectrograms
            return self._decode_batch_mel(
                content_embeddings, global_embeddings, content_lengths, target_audio_lengths, device
            )

    def _decode_batch_mel(
        self,
        content_embeddings: torch.Tensor,
        global_embeddings: torch.Tensor,
        content_lengths: torch.Tensor,
        target_audio_lengths: torch.Tensor,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Internal method for batch mel spectrogram synthesis."""
        batch_size = content_embeddings.size(0)
        max_seq_len = content_embeddings.size(1)

        # Calculate mel lengths for each sample
        mel_lengths = torch.tensor(
            [self._calculate_target_mel_length(length.item()) for length in target_audio_lengths],
            dtype=torch.long,
            device=device,
        )
        max_mel_length = mel_lengths.max().item()

        # Create padding mask for content embeddings (True = valid, False = padding)
        content_padding_mask = torch.arange(max_seq_len, device=device).unsqueeze(0) < content_lengths.unsqueeze(1)

        device_type = content_embeddings.device.type
        with _get_autocast_context(device_type):
            # Process through mel_prenet with padding mask
            local_latent = self.mel_prenet(content_embeddings, key_padding_mask=content_padding_mask)

            # Upsample with Conv1DTranspose if configured
            if self.mel_conv_upsample is not None:
                local_latent = self.mel_conv_upsample(local_latent.transpose(1, 2)).transpose(1, 2)

            # Calculate the latent length after conv upsample (for padding mask update)
            if self.mel_conv_upsample is not None:
                upsampled_content_lengths = content_lengths * self.config.mel_upsample_factor
            else:
                upsampled_content_lengths = content_lengths

            # Interpolate each sample to its target mel length, then pad to max_mel_length
            # We need to process each sample separately for variable-length interpolation
            mel_latents = []
            for i in range(batch_size):
                # Use only valid portion for interpolation
                valid_len = upsampled_content_lengths[i].item()
                sample_latent = local_latent[i : i + 1, :valid_len, :]  # (1, valid_len, C)
                mel_len = mel_lengths[i].item()

                # Interpolate to target mel length
                sample_latent = F.interpolate(
                    sample_latent.transpose(1, 2),  # (1, C, valid_len)
                    size=mel_len,
                    mode=self.config.mel_interpolation_mode,
                ).transpose(1, 2)  # (1, mel_len, C)

                # Pad to max_mel_length
                if mel_len < max_mel_length:
                    padding = torch.zeros(
                        1, max_mel_length - mel_len, sample_latent.size(2),
                        device=device, dtype=sample_latent.dtype
                    )
                    sample_latent = torch.cat([sample_latent, padding], dim=1)

                mel_latents.append(sample_latent)

            local_latent = torch.cat(mel_latents, dim=0)  # (B, max_mel_length, C)

            # Create padding mask for mel decoder (True = valid, False = padding)
            mel_padding_mask = torch.arange(max_mel_length, device=device).unsqueeze(0) < mel_lengths.unsqueeze(1)

            # Generate mel spectrogram, conditioned on global embeddings
            mel_recon = self.mel_decoder(
                local_latent,
                condition=global_embeddings.unsqueeze(1),
                key_padding_mask=mel_padding_mask
            )
            mel_recon = mel_recon.transpose(1, 2)  # (B, n_mels, max_mel_length)

            mel_recon = self.mel_postnet(mel_recon)

        return mel_recon, mel_lengths

    def _decode_batch_wave(
        self,
        content_embeddings: torch.Tensor,
        global_embeddings: torch.Tensor,
        content_lengths: torch.Tensor,
        target_audio_lengths: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Internal method for batch waveform synthesis.

        Processes each sample individually to ensure identical results to single-sample decode.
        """
        batch_size = content_embeddings.size(0)
        max_audio_length = target_audio_lengths.max().item()

        device_type = content_embeddings.device.type
        with _get_autocast_context(device_type):
            waveforms = []
            for i in range(batch_size):
                # Extract valid portion of content embedding for this sample
                valid_len = content_lengths[i].item()
                sample_content = content_embeddings[i : i + 1, :valid_len, :]  # (1, valid_len, C)
                sample_global = global_embeddings[i : i + 1, :]  # (1, dim)
                target_audio_len = target_audio_lengths[i].item()

                # Calculate target STFT length (same as single decode)
                stft_length = self._calculate_target_stft_length(target_audio_len)

                # Generate waveform using forward_wave (identical to single decode path)
                waveform = self.forward_wave(sample_content, sample_global, stft_length=stft_length)  # (1, samples)
                waveform = waveform.squeeze(0)  # (samples,)

                # Pad or trim to target audio length
                current_len = waveform.size(0)
                if current_len > target_audio_len:
                    waveform = waveform[:target_audio_len]
                elif current_len < target_audio_len:
                    pad_size = target_audio_len - current_len
                    waveform = F.pad(waveform, (0, pad_size), mode="constant", value=0.0)

                # Pad to max_audio_length for batching
                if target_audio_len < max_audio_length:
                    pad_size = max_audio_length - target_audio_len
                    waveform = F.pad(waveform, (0, pad_size), mode="constant", value=0.0)

                waveforms.append(waveform.unsqueeze(0))

            waveforms = torch.cat(waveforms, dim=0)  # (B, max_audio_length)

        return waveforms, target_audio_lengths

    @torch.inference_mode()
    def voice_conversion(self, source_waveform: torch.Tensor, reference_waveform: torch.Tensor) -> torch.Tensor:
        """Convert voice using MioCodec, keeping content from source and global characteristics from reference.

        If use_wave_decoder is True, returns waveform directly.
        Otherwise, returns mel spectrogram (requires external vocoder for waveform synthesis).

        Args:
            source_waveform (torch.Tensor): Source audio waveform tensor (samples,).
            reference_waveform (torch.Tensor): Reference audio waveform tensor (samples_ref,).
        Returns:
            torch.Tensor: If use_wave_decoder=True, returns waveform tensor (samples,).
                          Otherwise, returns mel spectrogram tensor (n_mels, T).
        """
        # Extract source content features and reference global features
        source_features = self.encode(source_waveform, return_content=True, return_global=False)
        reference_features = self.encode(reference_waveform, return_content=False, return_global=True)

        # Synthesize using source content and reference global features
        output = self.decode(
            content_embedding=source_features.content_embedding,
            global_embedding=reference_features.global_embedding,
            target_audio_length=source_waveform.size(0),
        )
        return output
