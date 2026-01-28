import logging
import os
from pathlib import Path
from contextlib import contextmanager

import torch
import torch.nn as nn

# Configure logger
logger = logging.getLogger("miocodec")
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
handler.setLevel(logging.INFO)
handler.setFormatter(logging.Formatter("[%(asctime)s] %(levelname)s %(name)s: %(message)s"))
logger.addHandler(handler)


def get_logger() -> logging.Logger:
    return logger


def freeze_modules(modules: list[nn.Module] | None):
    for module in modules:
        if module is not None:
            for param in module.parameters():
                param.requires_grad = False


def _env_truthy(name: str) -> bool:
    value = os.environ.get(name, "").strip().lower()
    return value in {"1", "true", "yes", "on"}


@contextmanager
def _suppress_stderr(enabled: bool):
    if not enabled:
        yield
        return
    devnull = os.open(os.devnull, os.O_WRONLY)
    old_stderr = os.dup(2)
    try:
        os.dup2(devnull, 2)
        yield
    finally:
        os.dup2(old_stderr, 2)
        os.close(old_stderr)
        os.close(devnull)


def _load_audio_internal(
    path: str, frame_offset: int | None = None, num_frames: int | None = None
) -> tuple[torch.Tensor, int]:
    # TorchAudio >= 2.9.0 removed decoding and encoding capabilities to TorchCodec.
    # See: https://github.com/pytorch/audio/issues/3902
    # waveform, sample_rate = torchaudio.load(path, frame_offset=frame_offset or 0, num_frames=num_frames or -1)

    import soundfile as sf

    suppress_warnings = _env_truthy("MIOCODEC_SUPPRESS_AUDIO_WARNINGS") or _env_truthy(
        "KANADE_TOKENIZER_SUPPRESS_AUDIO_WARNINGS"
    )
    with _suppress_stderr(suppress_warnings):
        with sf.SoundFile(path) as f:
            if frame_offset is not None:
                f.seek(frame_offset)
            frames = f.read(frames=num_frames or -1, dtype="float32", always_2d=True)
            waveform = torch.from_numpy(frames.T)
            sample_rate = f.samplerate
    return waveform, sample_rate


def load_audio(audio_path: str, sample_rate: int = 24000) -> torch.Tensor:
    import torchaudio

    """Load and preprocess audio file."""
    waveform, sr = _load_audio_internal(audio_path)

    # Convert to mono if stereo
    if waveform.shape[0] > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)

    # Resample if necessary
    if sr != sample_rate:
        resampler = torchaudio.transforms.Resample(sr, sample_rate)
        waveform = resampler(waveform)

    # Normalize waveform
    max_val = torch.max(torch.abs(waveform)) + 1e-8
    waveform = waveform / max_val  # Normalize to [-1, 1]

    return waveform.squeeze(0)  # Remove channel dimension


class _PupuVocoderWrapper(nn.Module):
    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def decode(self, mel: torch.Tensor) -> torch.Tensor:
        # PupuVocoder returns (B, 1, T). Match Vocos-style output (B, T).
        audio = self.model(mel)
        if audio.dim() == 3:
            audio = audio.squeeze(1)
        return audio


def _iter_unique_paths(paths: list[Path]) -> list[Path]:
    seen: set[Path] = set()
    unique: list[Path] = []
    for path in paths:
        if path not in seen:
            seen.add(path)
            unique.append(path)
    return unique


def _resolve_pupu_roots(afgen_root: str | None) -> list[Path]:
    roots: list[Path] = []
    if afgen_root is not None:
        root = Path(afgen_root)
        if not root.exists():
            raise FileNotFoundError(f"afgen_root not found: {afgen_root}")
        roots.append(root)

    internal_root = Path(__file__).resolve().parent / "pupu" / "assets"
    if internal_root.exists():
        roots.append(internal_root)

    try:
        file_path = Path(__file__).resolve()
        for parent in file_path.parents:
            candidate = parent / "AliasingFreeNeuralAudioSynthesis"
            if candidate.exists():
                roots.append(candidate)
                break
    except IndexError:
        pass
    cwd_candidate = Path.cwd() / "AliasingFreeNeuralAudioSynthesis"
    if cwd_candidate.exists():
        roots.append(cwd_candidate)

    return _iter_unique_paths(roots)


def _select_pupu_checkpoint(checkpoint_path: str | None) -> Path | None:
    if checkpoint_path is None:
        return None

    path = Path(checkpoint_path)
    if path.is_file():
        return path

    if path.is_dir():
        model_path = path / "model.safetensors"
        if model_path.exists():
            return model_path

        subdirs = [p for p in path.iterdir() if p.is_dir()]
        if subdirs:
            # Names are zero-padded; lexicographic sort works for epoch/step.
            subdirs.sort(key=lambda p: p.name, reverse=True)
            candidate = subdirs[0] / "model.safetensors"
            if candidate.exists():
                return candidate

    raise FileNotFoundError(f"Unsupported checkpoint path: {checkpoint_path}")


def _load_pupu_state_dict(checkpoint_path: Path) -> dict[str, torch.Tensor]:
    if checkpoint_path.suffix == ".safetensors":
        from safetensors.torch import load_file

        state_dict = load_file(str(checkpoint_path))
    else:
        pkg = torch.load(checkpoint_path, map_location="cpu")
        state_dict = pkg["generator"] if isinstance(pkg, dict) and "generator" in pkg else pkg

    if any(key.startswith("module.") for key in state_dict.keys()):
        state_dict = {key.replace("module.", "", 1): value for key, value in state_dict.items()}
    return state_dict


def _strip_state_dict_prefix(state_dict: dict[str, torch.Tensor], prefix: str) -> dict[str, torch.Tensor]:
    if not prefix:
        return state_dict
    if not any(key.startswith(prefix) for key in state_dict):
        return state_dict
    return {key[len(prefix) :] if key.startswith(prefix) else key: value for key, value in state_dict.items()}


def _hf_download_first(repo_id: str, candidates: list[str], revision: str | None) -> Path:
    from huggingface_hub import hf_hub_download

    last_error: Exception | None = None
    for filename in candidates:
        try:
            return Path(hf_hub_download(repo_id=repo_id, filename=filename, revision=revision))
        except Exception as exc:  # pragma: no cover - hub errors are opaque
            last_error = exc
    raise FileNotFoundError(
        f"Could not find any of {candidates} in HuggingFace repo {repo_id}."
    ) from last_error


def _hf_select_pupu_checkpoint(
    repo_id: str, checkpoint_path: str, revision: str | None
) -> Path:
    path = checkpoint_path.rstrip("/")
    if path.endswith((".safetensors", ".ckpt", ".pt", ".pth", ".bin")):
        return _hf_download_first(repo_id, [path], revision)

    candidates = [
        f"{path}/model.safetensors",
        f"{path}/model.pt",
        f"{path}/model.pth",
        f"{path}/model.bin",
    ]
    return _hf_download_first(repo_id, candidates, revision)


def load_pupu_vocoder(
    config_path: str | None = None,
    checkpoint_path: str | None = None,
    hf_repo: str | None = None,
    hf_revision: str | None = None,
    hf_config_path: str | None = None,
    hf_checkpoint_path: str | None = None,
    afgen_root: str | None = None,
    device: str | torch.device | None = None,
) -> nn.Module:
    from miocodec.pupu.config import DEFAULT_PUPU_CONFIG, JsonHParams, load_config
    from miocodec.pupu.models.vocoders.gan.generator.pupuvocoder import PupuVocoder

    search_roots = _resolve_pupu_roots(afgen_root)
    default_root = search_roots[0] if search_roots else None

    if hf_repo is not None:
        if config_path is None:
            if hf_config_path is not None:
                config_path = str(_hf_download_first(hf_repo, [hf_config_path], hf_revision))
            else:
                config_candidates = [
                    "egs/pupuvocoder/exp_config_pupuvocoder.json",
                    "egs/pupuvocoder/exp_config_pupuvocoder_large.json",
                    "egs/exp_config_pupuvocoder.json",
                    "egs/exp_config_pupuvocoder_large.json",
                    "exp_config_pupuvocoder.json",
                    "exp_config_pupuvocoder_large.json",
                    "config.json",
                ]
                config_path = str(_hf_download_first(hf_repo, config_candidates, hf_revision))
        if hf_checkpoint_path is not None:
            # Explicit HF checkpoint takes precedence over any local checkpoint_path.
            checkpoint_path = str(
                _hf_select_pupu_checkpoint(hf_repo, hf_checkpoint_path, hf_revision)
            )
        elif checkpoint_path is None:
            raise ValueError(
                "hf_repo is set but no checkpoint provided. "
                "Pass hf_checkpoint_path (e.g., 'pupuvocoder/checkpoint/epoch-XXXX...')."
            )
    if config_path is None:
        if default_root is not None:
            config_path = str(default_root / "egs/pupuvocoder/exp_config_pupuvocoder.json")
        else:
            cfg = JsonHParams(**DEFAULT_PUPU_CONFIG)
            model = PupuVocoder(cfg).eval()
            selected_ckpt = _select_pupu_checkpoint(checkpoint_path)
            if selected_ckpt is not None:
                state_dict = _load_pupu_state_dict(selected_ckpt)
                model.load_state_dict(state_dict, strict=False)
            else:
                logger.warning("Pupu-Vocoder loaded without checkpoint; outputs will be random.")
            if device is not None:
                model = model.to(device)
            return _PupuVocoderWrapper(model)

    cfg = load_config(config_path, search_roots=search_roots)
    model = PupuVocoder(cfg).eval()

    selected_ckpt = _select_pupu_checkpoint(checkpoint_path)
    if selected_ckpt is not None:
        state_dict = _load_pupu_state_dict(selected_ckpt)
        model.load_state_dict(state_dict, strict=False)
    else:
        logger.warning("Pupu-Vocoder loaded without checkpoint; outputs will be random.")

    if device is not None:
        model = model.to(device)

    return _PupuVocoderWrapper(model)


def load_pupu_vocoder_state_dict(
    state_dict: dict[str, torch.Tensor],
    config_path: str | None = None,
    afgen_root: str | None = None,
    device: str | torch.device | None = None,
) -> nn.Module:
    """Instantiate a Pupu-Vocoder and load weights from an in-memory state dict."""
    if not state_dict:
        raise ValueError("Empty state_dict passed to load_pupu_vocoder_state_dict.")

    from miocodec.pupu.config import DEFAULT_PUPU_CONFIG, JsonHParams, load_config
    from miocodec.pupu.models.vocoders.gan.generator.pupuvocoder import PupuVocoder

    search_roots = _resolve_pupu_roots(afgen_root)
    default_root = search_roots[0] if search_roots else None

    if config_path is None and default_root is not None:
        config_path = str(default_root / "egs/pupuvocoder/exp_config_pupuvocoder.json")

    if config_path is None:
        cfg = JsonHParams(**DEFAULT_PUPU_CONFIG)
    else:
        cfg = load_config(config_path, search_roots=search_roots)

    model = PupuVocoder(cfg).eval()
    normalized_state = _strip_state_dict_prefix(state_dict, "model.")
    model.load_state_dict(normalized_state, strict=False)

    wrapper = _PupuVocoderWrapper(model)
    if device is not None:
        wrapper = wrapper.to(device)
    return wrapper


def load_vocoder(backend: str = "pupu", **kwargs):
    if backend != "pupu":
        raise ValueError(
            "Only the 'pupu' vocoder backend is supported in this inference-only build."
        )
    return load_pupu_vocoder(**kwargs)


def vocode(vocoder, mel_spectrogram: torch.Tensor) -> torch.Tensor:
    """Convert mel spectrogram to waveform using the selected vocoder.
    Args:
        vocoder (nn.Module): Vocoder with a decode() method.
        mel_spectrogram (torch.Tensor): Input mel spectrogram tensor (..., n_mels, frame).
    Returns:
        torch.Tensor: Generated audio waveform tensor (..., samples).
    """
    mel_spectrogram = mel_spectrogram.to(torch.float32)  # Ensure mel spectrogram is in float32
    with torch.inference_mode():
        generated_waveform = vocoder.decode(mel_spectrogram)
    return generated_waveform
