from __future__ import annotations

import torch
import torch.nn as nn

from .model import MioCodecFeatures, MioCodecModel
from .util import get_logger, load_pupu_vocoder_state_dict, vocode

logger = get_logger()

_VOCODER_PREFIX = "vocoder."


def _split_combined_state_dict(
    state_dict: dict[str, torch.Tensor], prefix: str = _VOCODER_PREFIX
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    codec_state: dict[str, torch.Tensor] = {}
    vocoder_state: dict[str, torch.Tensor] = {}
    for key, value in state_dict.items():
        if key.startswith(prefix):
            vocoder_state[key[len(prefix) :]] = value
        else:
            codec_state[key] = value
    if not vocoder_state:
        raise ValueError(f"No vocoder weights found with prefix '{prefix}'.")
    return codec_state, vocoder_state


class MioCodec(nn.Module):
    """Inference wrapper that bundles MioCodec + vocoder into a single model API."""

    def __init__(self, model: MioCodecModel, vocoder: nn.Module):
        super().__init__()
        self.model = model
        self.vocoder = vocoder
        self.config = model.config

    @classmethod
    def from_pretrained(
        cls,
        repo_id: str | None = None,
        revision: str | None = None,
        config_path: str | None = None,
        weights_path: str | None = None,
        vocoder_config_path: str | None = None,
        hf_vocoder_config_path: str | None = None,
        afgen_root: str | None = None,
    ) -> "MioCodec":
        """Load a single-file MioCodec + vocoder bundle.

        The combined weights file must include vocoder weights prefixed with 'vocoder.'.
        """
        if repo_id is not None:
            from huggingface_hub import hf_hub_download

            config_path = hf_hub_download(repo_id, "config.yaml", revision=revision)
            weights_path = hf_hub_download(repo_id, "model.safetensors", revision=revision)
            if vocoder_config_path is None:
                if hf_vocoder_config_path is None:
                    hf_vocoder_config_path = "vocoder_config.json"
                try:
                    vocoder_config_path = hf_hub_download(
                        repo_id, hf_vocoder_config_path, revision=revision
                    )
                except Exception:
                    logger.warning(
                        "Vocoder config not found on HF (expected '%s'); falling back to local defaults.",
                        hf_vocoder_config_path,
                    )
        else:
            if config_path is None or weights_path is None:
                raise ValueError(
                    "Please provide either HuggingFace Hub repo_id or both config_path and weights_path."
                )

        model = MioCodecModel.from_hparams(config_path)

        from safetensors.torch import load_file

        state_dict = load_file(weights_path, device="cpu")
        codec_state, vocoder_state = _split_combined_state_dict(state_dict)
        model.load_state_dict(codec_state, strict=False)
        logger.info(f"Loaded codec weights from: {weights_path}")

        vocoder = load_pupu_vocoder_state_dict(
            vocoder_state, config_path=vocoder_config_path, afgen_root=afgen_root
        )
        logger.info("Loaded vocoder weights from checkpoint.")

        return cls(model, vocoder)

    @torch.inference_mode()
    def encode(
        self, waveform: torch.Tensor, return_content: bool = True, return_global: bool = True
    ) -> MioCodecFeatures:
        return self.model.encode(waveform, return_content=return_content, return_global=return_global)

    @torch.inference_mode()
    def decode(
        self,
        global_embedding: torch.Tensor | None = None,
        content_token_indices: torch.Tensor | None = None,
        content_embedding: torch.Tensor | None = None,
        target_audio_length: int | None = None,
        features: MioCodecFeatures | None = None,
    ) -> torch.Tensor:
        if features is not None:
            if any(
                value is not None
                for value in (global_embedding, content_token_indices, content_embedding)
            ):
                raise ValueError("Pass either features or explicit embeddings, not both.")
            global_embedding = features.global_embedding
            content_embedding = features.content_embedding
            content_token_indices = features.content_token_indices

        if global_embedding is None:
            raise ValueError("global_embedding is required (or pass features with global_embedding).")

        mel = self.model.decode(
            global_embedding=global_embedding,
            content_token_indices=content_token_indices,
            content_embedding=content_embedding,
            target_audio_length=target_audio_length,
        )
        audio = vocode(self.vocoder, mel.unsqueeze(0))
        return audio.squeeze(0)

    @torch.inference_mode()
    def synthesize_from_tokens(
        self,
        content_token_indices: torch.Tensor | list[int],
        reference_waveform: torch.Tensor,
        target_audio_length: int | None = None,
    ) -> torch.Tensor:
        """Synthesize audio from content token indices and a reference waveform.

        Args:
            content_token_indices (torch.Tensor | list[int]): Content token indices (seq_len,).
            reference_waveform (torch.Tensor): Reference waveform tensor (samples,).
            target_audio_length (int, optional): Target length of the output audio in samples.
        Returns:
            torch.Tensor: Synthesized audio waveform (samples,).
        """
        device = next(self.model.parameters()).device
        if isinstance(content_token_indices, list):
            content_token_indices = torch.tensor(content_token_indices, dtype=torch.long, device=device)
        elif isinstance(content_token_indices, torch.Tensor):
            if content_token_indices.dtype != torch.long:
                content_token_indices = content_token_indices.long()
            if content_token_indices.device != device:
                content_token_indices = content_token_indices.to(device)
        else:
            raise TypeError("content_token_indices must be a torch.Tensor or list[int].")

        if reference_waveform.device != device:
            reference_waveform = reference_waveform.to(device)

        ref_features = self.model.encode(reference_waveform, return_content=False, return_global=True)
        mel = self.model.decode(
            global_embedding=ref_features.global_embedding,
            content_token_indices=content_token_indices,
            target_audio_length=target_audio_length,
        )
        audio = vocode(self.vocoder, mel.unsqueeze(0))
        return audio.squeeze(0)

    @torch.inference_mode()
    def synthesize_from_tokens_batch(
        self,
        content_token_indices_list: list[torch.Tensor] | list[list[int]],
        reference_waveform: torch.Tensor,
        target_audio_lengths: list[int] | None = None,
        padding_token_idx: int = 0,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Synthesize multiple audio waveforms from multiple token sequences and a single reference waveform.

        Args:
            content_token_indices_list (list[torch.Tensor] | list[list[int]]): List of content token indices.
                Each element is a 1D tensor or list of shape (seq_len_i,).
            reference_waveform (torch.Tensor): Reference waveform tensor (samples,).
                The global embedding extracted from this waveform is used for all samples.
            target_audio_lengths (list[int], optional): Target audio length for each sample.
                If None, uses the audio length estimated from token sequence lengths.
            padding_token_idx (int): Token index used for padding shorter sequences.

        Returns:
            tuple[torch.Tensor, torch.Tensor]:
                - audio_waveforms: Generated audio waveform tensor (B, max_audio_len), padded to max length
                - audio_lengths: Actual audio length for each sample (B,)
        """
        device = next(self.model.parameters()).device
        batch_size = len(content_token_indices_list)

        # Convert all token sequences to tensors and find max length
        token_tensors = []
        content_lengths = []
        for tokens in content_token_indices_list:
            if isinstance(tokens, list):
                t = torch.tensor(tokens, dtype=torch.long, device=device)
            elif isinstance(tokens, torch.Tensor):
                t = tokens.long().to(device)
            else:
                raise TypeError("Each element of content_token_indices_list must be a torch.Tensor or list[int].")
            token_tensors.append(t)
            content_lengths.append(t.shape[0])

        max_seq_len = max(content_lengths)

        # Create padded token tensor (B, max_seq_len)
        batch_tokens = torch.full(
            (batch_size, max_seq_len), padding_token_idx, dtype=torch.long, device=device
        )
        for i, t in enumerate(token_tensors):
            batch_tokens[i, : t.shape[0]] = t

        # Extract global embedding from reference waveform
        if reference_waveform.device != device:
            reference_waveform = reference_waveform.to(device)
        ref_features = self.model.encode(reference_waveform, return_content=False, return_global=True)

        # Expand global embedding to batch size (B, dim)
        global_embeddings = ref_features.global_embedding.unsqueeze(0).expand(batch_size, -1)

        # Use decode_batch
        audio_waveforms, audio_lengths = self.decode_batch(
            global_embeddings=global_embeddings,
            content_token_indices=batch_tokens,
            content_lengths=content_lengths,
            target_audio_lengths=target_audio_lengths,
            padding_token_idx=padding_token_idx,
        )

        return audio_waveforms, audio_lengths

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
        """Synthesize audio waveforms from batched content and global features using MioCodec.

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
                - audio_waveforms: Generated audio waveform tensor (B, max_audio_len), padded to max length
                - audio_lengths: Actual audio length for each sample (B,)
        """
        # Generate mel spectrograms using the model's decode_batch
        mel_spectrograms, mel_lengths = self.model.decode_batch(
            global_embeddings=global_embeddings,
            content_token_indices=content_token_indices,
            content_embeddings=content_embeddings,
            content_lengths=content_lengths,
            target_audio_lengths=target_audio_lengths,
            padding_token_idx=padding_token_idx,
        )

        # Convert mel spectrograms to waveforms using vocoder
        audio_waveforms = vocode(self.vocoder, mel_spectrograms)  # (B, max_audio_len)

        # Calculate actual audio lengths from mel lengths
        # Audio length = mel_length * hop_length (approximately, depends on vocoder)
        hop_length = self.config.hop_length
        audio_lengths = mel_lengths * hop_length

        return audio_waveforms, audio_lengths

    @torch.inference_mode()
    def voice_conversion(
        self, source_waveform: torch.Tensor, reference_waveform: torch.Tensor
    ) -> torch.Tensor:
        mel = self.model.voice_conversion(source_waveform, reference_waveform)
        audio = vocode(self.vocoder, mel.unsqueeze(0))
        return audio.squeeze(0)
