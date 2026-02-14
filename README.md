# MioCodec: High-Fidelity Neural Audio Codec for Efficient Spoken Language Modeling

[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Model-MioCodec--25Hz--44.1kHz--v2-green)](https://huggingface.co/Aratako/MioCodec-25Hz-44.1kHz-v2)
[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Model-MioCodec--25Hz--24kHz-yellow)](https://huggingface.co/Aratako/MioCodec-25Hz-24kHz)
[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Model-MioCodec--25Hz--44.1kHz-blue)](https://huggingface.co/Aratako/MioCodec-25Hz-44.1kHz)

MioCodec is a high-fidelity neural audio codec for efficient spoken language modeling. Multiple variants are available:

- **MioCodec-25Hz-44.1kHz-v2** (Recommended): High-quality 44.1 kHz model with integrated wave decoder
- **MioCodec-25Hz-24kHz**: Lightweight 24 kHz model with integrated wave decoder
- **MioCodec-25Hz-44.1kHz**: Legacy 44.1 kHz model using external vocoder (MioVocoder)

## Model Comparison

| Model | Token Rate | Vocab Size | Bit Rate | Sample Rate | SSL Encoder | Vocoder | Parameters | Highlights |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **MioCodec-25Hz-44.1kHz-v2** | 25 Hz | 12,800 | 341 bps | 44.1 kHz | WavLM-base+ | - (iSTFTHead) | 133M | Fast inference, good quality |
| **MioCodec-25Hz-24kHz** | 25 Hz | 12,800 | 341 bps | 24 kHz | WavLM-base+ | - (iSTFTHead) | 132M | Lightweight, fast inference |
| **MioCodec-25Hz-44.1kHz** | 25 Hz | 12,800 | 341 bps | 44.1 kHz | WavLM-base+ | MioVocoder | 118M (w/o vocoder) | High-quality, high sample rate |
| kanade-12.5hz | 12.5 Hz | 12,800 | 171 bps | 24 kHz | WavLM-base+ | Vocos 24kHz | 120M (w/o vocoder) | Original 12.5Hz model |
| kanade-25hz | 25 Hz | 12,800 | 341 bps | 24 kHz | WavLM-base+ | Vocos 24kHz | 118M (w/o vocoder) | Original 25Hz model |
| kanade-25hz-clean | 25 Hz | 12,800 | 341 bps | 24 kHz | WavLM-base+ | HiFT 24kHz | 142M (w/o vocoder) | Original 25Hz clean model |

## Repository Structure

- `src/miocodec/`: Inference code (model + modules + vocoder helpers; includes `pupu/` for Pupu-Vocoder inference + default config assets)
- `config/model/25hz_miocodec.yaml`: Model configuration

## Setup

Create and activate a virtual environment before installing.

1. Install directly from Git:

```bash
uv add git+https://github.com/Aratako/MioCodec
# or
pip install git+https://github.com/Aratako/MioCodec
```

2. Alternatively, clone the repository and install in editable mode:

```bash
git clone https://github.com/Aratako/MioCodec
cd MioCodec
```

```bash
uv sync
# or
pip install -e .
```

### FlashAttention (recommended)

To fully reproduce our setup, we recommend installing FlashAttention. If it is unavailable, the model falls back to PyTorch SDPA, but behavior and quality are not guaranteed.

Note: even if FlashAttention is installed, CPU inference automatically falls back to PyTorch SDPA.

If you use uv, you can install it like:

```bash
uv pip install flash-attn --no-build-isolation
```

Ensure `ninja` is installed on your system, or the build will be very slow.

## Usage

### Inference

#### MioCodec-25Hz-44.1kHz-v2 / MioCodec-25Hz-24kHz (Recommended)

These models use an integrated wave decoder (iSTFT-based) for direct waveform synthesis without requiring an external vocoder. This makes them lightweight and fast.

```python
from miocodec import MioCodecModel, load_audio
import soundfile as sf

# Load model from Hugging Face
# Use "Aratako/MioCodec-25Hz-44.1kHz-v2" for 44.1kHz or "Aratako/MioCodec-25Hz-24kHz" for 24kHz
model = MioCodecModel.from_pretrained("Aratako/MioCodec-25Hz-44.1kHz-v2")
model = model.eval().cuda()

# Load audio
waveform = load_audio("path/to/audio.wav", sample_rate=model.config.sample_rate).cuda()

# Encode
features = model.encode(waveform)

# Decode to waveform (directly, no vocoder needed)
resynth = model.decode(
    content_token_indices=features.content_token_indices,
    global_embedding=features.global_embedding,
)

# Save
sf.write("resynth.wav", resynth.cpu().numpy(), model.config.sample_rate)
```

#### MioCodec-25Hz-44.1kHz (Legacy)

The legacy 44.1kHz version uses an external vocoder (MioVocoder) for waveform synthesis.

```python
from miocodec import MioCodec, load_audio
import soundfile as sf

# Load model from Hugging Face
model = MioCodec.from_pretrained("Aratako/MioCodec-25Hz-44.1kHz")
model = model.eval().cuda()

# Load audio
waveform = load_audio("path/to/audio.wav", sample_rate=model.config.sample_rate).cuda()

# Encode
features = model.encode(waveform)

# Decode to waveform
resynth = model.decode(features=features)

# Save
sf.write("resynth.wav", resynth.cpu().numpy(), samplerate=model.config.sample_rate)
```

If you want to encode/decode with CPU, just move the model and tensors to CPU with `.cpu()`.

Note: when running on CPU, FlashAttention (even if installed) is not used and the model falls back to PyTorch SDPA.

### Voice Conversion

This model encodes speech into content tokens (primarily what is being said) and a global embedding (primarily speaker traits, acoustic conditions, microphone characteristics, and the like). By combining content tokens from one audio with the global embedding from another, you can perform voice conversion (the `voice_conversion` helper wraps this).

```python
# Voice conversion (content from source, speaker from reference)
source = load_audio("path/to/source.wav", sample_rate=model.config.sample_rate).cuda()
reference = load_audio("path/to/reference.wav", sample_rate=model.config.sample_rate).cuda()

# Perform voice conversion
vc_wave = model.voice_conversion(source, reference)
sf.write("vc.wav", vc_wave.cpu().numpy(), samplerate=model.config.sample_rate)
```

## Acknowledgements

Thanks to the following projects and repositories:

- Codec architecture and implementation are based on [kanade-tokenizer](https://github.com/frothywater/kanade-tokenizer).
- Original vocoder weights and codebase are based on [AliasingFreeNeuralAudioSynthesis](https://github.com/sizigi/AliasingFreeNeuralAudioSynthesis).
- Decoder design for the 24kHz and 44.1kHz-v2 versions is inspired by [XCodec2](https://github.com/zhenye234/X-Codec-2.0).

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Citation

```bibtex
@misc{miocodec,
  author = {Chihiro Arata},
  title = {MioCodec: A High-Fidelity 44.1kHz Neural Audio Codec},
  year = {2026},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/Aratako/MioCodec}}
}
```
