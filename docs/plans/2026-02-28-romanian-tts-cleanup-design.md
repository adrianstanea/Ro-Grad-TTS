# Romanian Grad-TTS: Cleanup, Inference API & Packaging Design

Date: 2026-02-28

## Goals

1. Clean up the commit history on `adaptation_romanian` into atomic logical units
2. Create a `GradTTSRomanian` inference API with HuggingFace Hub integration
3. Package the Grad-TTS code with `uv`/`pyproject.toml` for reproducibility

## Decisions

- **HuggingFace repo**: `adrianstanea/Ro-Grad-TTS`
- **Device support**: Auto-detect CPU/CUDA
- **Checkpoints**: Removed from git, served from HuggingFace Hub only
- **Python version**: Start with 3.6.9, increment only if needed
- **Commit cleanup**: Squash 25 commits into ~6 logical groups

---

## 1. HuggingFace Hub Repository Structure

```
adrianstanea/Ro-Grad-TTS/
├── config.json                      # Model hyperparams (replaces params.py for inference)
├── hifigan_config.json              # HiFi-GAN vocoder config
├── models/
│   ├── swara/
│   │   └── grad-tts-base-1000.pt    # Baseline model (1000 epochs)
│   ├── bas/
│   │   ├── grad-tts-bas-10_100.pt
│   │   └── grad-tts-bas-950_100.pt
│   ├── sgs/
│   │   ├── grad-tts-sgs-10_100.pt
│   │   └── grad-tts-sgs-950_100.pt
│   └── vocoder/
│       └── hifigan_univ_v1
```

`config.json` encodes all model hyperparameters currently in `params.py`:

```json
{
  "n_vocab": 178,
  "n_spks": 1,
  "spk_emb_dim": 64,
  "n_enc_channels": 192,
  "filter_channels": 768,
  "filter_channels_dp": 256,
  "n_heads": 2,
  "n_enc_layers": 6,
  "enc_kernel": 3,
  "enc_dropout": 0.1,
  "window_size": 4,
  "n_feats": 80,
  "dec_dim": 64,
  "beta_min": 0.05,
  "beta_max": 20.0,
  "pe_scale": 1000,
  "sample_rate": 22050,
  "add_blank": true
}
```

## 2. Inference API

### Usage

```python
from gradtts_ro import GradTTSRomanian

# From HuggingFace Hub
pipe = GradTTSRomanian.from_pretrained(
    "adrianstanea/Ro-Grad-TTS",
    variant="bas-10_100"   # optional, defaults to base model
)

# From local files
pipe = GradTTSRomanian.from_local(
    model_path="./checkpts/grad-tts-ro.pt",
    vocoder_path="./checkpts/hifigan_univ_v1",
    vocoder_config_path="./checkpts/hifigan-config.json"
)

# Synthesize (returns numpy int16 array at 22050 Hz)
audio = pipe("Salut, aceasta este o propozitie.")
pipe.save_wav(audio, "output.wav")

# Batch
audios = pipe(["Text one.", "Text two."])
```

### Package Structure

```
Grad-TTS/gradtts_ro/
├── __init__.py          # Exports GradTTSRomanian
├── pipeline.py          # Main class: from_pretrained, from_local, __call__, save_wav
├── text_processing.py   # Espeak phonemization (from tools/text_processing/)
├── symbols.py           # Phoneme symbol set
├── cleaners.py          # Text normalization
└── vocoder/             # Vendored HiFi-GAN
    ├── __init__.py
    ├── models.py
    └── env.py
```

### Key Implementation Details

- `from_pretrained()`: Uses `huggingface_hub.hf_hub_download()` to fetch config.json, model checkpoint, vocoder checkpoint. The `variant` param maps to HF directory structure.
- `from_local()`: Loads directly from local paths, requires a config.json or uses default hyperparams.
- Device auto-detection: `torch.device("cuda" if torch.cuda.is_available() else "cpu")`
- Eliminates `sys.path` hacks for hifi-gan by vendoring it properly.
- The `model/` directory (GradTTS, TextEncoder, Diffusion, monotonic_align) is imported as-is.

## 3. Packaging with uv

### pyproject.toml (in Grad-TTS/)

```toml
[project]
name = "gradtts-ro"
version = "0.1.0"
description = "Romanian Grad-TTS: Diffusion-based Text-to-Speech"
requires-python = ">=3.6.9"
dependencies = [
    "torch>=1.9.0",
    "torchaudio>=0.9.0",
    "numpy>=1.19.5",
    "scipy>=1.5.4",
    "phonemizer>=3.2.1",
    "huggingface-hub>=0.14.0",
]

[project.optional-dependencies]
train = [
    "matplotlib>=3.3.3",
    "tensorboard>=2.10.1",
    "tqdm",
    "Cython>=0.29.23",
    "librosa>=0.8.0",
]

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"
```

- `uv lock` generates a lockfile for reproducible installs
- Core deps are inference-only; training deps are optional
- `requirements.txt` kept for backward compat

### Directory Layout After Packaging

```
Grad-TTS/
├── pyproject.toml
├── uv.lock
├── gradtts_ro/              # New inference package
│   └── ...
├── model/                   # Existing model code
├── hifi-gan/                # Original (kept for training scripts)
├── train.py                 # Training (unchanged)
├── data.py                  # Dataset (unchanged)
├── params.py                # Training params only
├── tools/                   # Original text processing (kept)
├── requirements.txt         # Legacy
└── ...
```

## 4. Git History Cleanup

Squash 25 commits into 6 atomic commits on a new clean branch:

| # | Commit | Content |
|---|--------|---------|
| 1 | `feat: add Romanian text processing with Espeak phonemization` | `tools/text_processing/`, drop `text/`, update `data.py`, add phonemizer dep |
| 2 | `feat: adapt model architecture for Romanian multi-speaker TTS` | `model/tts.py` (spk_emb_dim, n_spks), mel cropping fix |
| 3 | `feat: configure training pipeline for SWARA 1.0 dataset` | `params.py`, `train.py`, `finetune.sh`, SWARA filelists, `.gitignore` |
| 4 | `feat: add Romanian inference pipeline` | `inference_RO.py`, `inference_RO.sh`, `inference.py` updates |
| 5 | `feat: add installable package with HuggingFace integration` | `pyproject.toml`, `gradtts_ro/`, vendored vocoder, `uv.lock` |
| 6 | `docs: document Romanian TTS adaptation and usage` | README updates |

Binary checkpoints are excluded from git history entirely.
