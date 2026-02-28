# Grad-TTS for Romanian (Ro-Grad-TTS)

Adaptation of [Grad-TTS](https://arxiv.org/abs/2105.06337) (ICML 2021) for Romanian text-to-speech synthesis, trained on the [SWARA 1.0](https://huggingface.co/datasets/swara/swara1.0) dataset.

## Prerequisites

- Python >= 3.9
- [espeak-ng](https://github.com/espeak-ng/espeak-ng) (system package, required for Romanian phonemization)
- A C compiler (gcc/clang) for building the Cython extension

Install espeak-ng:

```bash
# Ubuntu/Debian
sudo apt-get install espeak-ng

# macOS
brew install espeak-ng
```

## Installation

### Using uv (recommended)

```bash
cd Grad-TTS

# Install the package and all dependencies
uv sync

# Build the monotonic alignment C extension (must run from this directory)
cd model/monotonic_align
uv run python setup.py build_ext --inplace
cd ../..
```

### Using pip

```bash
cd Grad-TTS

pip install -e .

# Install Cython (needed to build the C extension)
pip install Cython numpy

# Build the monotonic alignment C extension (must run from this directory)
cd model/monotonic_align
python3 setup.py build_ext --inplace
cd ../..
```

> **Note:** The `setup.py` for monotonic_align must be run from inside `model/monotonic_align/`, not from the project root. It compiles `core.pyx` into a shared library (`.so`) used at runtime.

## Quick Start

### From HuggingFace Hub

```python
from gradtts_ro import GradTTSRomanian

pipe = GradTTSRomanian.from_pretrained("adrianstanea/Ro-Grad-TTS")
audio = pipe("Buna ziua! Aceasta este o demonstratie.")
pipe.save_wav(audio, "output.wav")
```

### From local checkpoints

```python
from gradtts_ro import GradTTSRomanian

pipe = GradTTSRomanian.from_local(
    model_path="path/to/grad-tts-base-1000.pt",
    vocoder_path="path/to/hifigan_univ_v1",
    vocoder_config_path="path/to/hifigan_config.json",
)
audio = pipe("Buna ziua! Aceasta este o demonstratie.")
pipe.save_wav(audio, "output.wav")
```

### Using the example script

```bash
# From HuggingFace Hub
uv run python examples/synthesize.py

# From local checkpoints
uv run python examples/synthesize.py \
    --model-path path/to/grad-tts-base-1000.pt \
    --vocoder-path path/to/hifigan_univ_v1 \
    --vocoder-config path/to/hifigan_config.json

# Custom text
uv run python examples/synthesize.py --text "Salut, cum te cheama?"
```

## API Reference

### `GradTTSRomanian.from_pretrained()`

```python
pipe = GradTTSRomanian.from_pretrained(
    repo_id="adrianstanea/Ro-Grad-TTS",  # HuggingFace repo
    variant="base",                       # Model variant (see below)
    device=None,                          # "cpu", "cuda", or auto-detect
)
```

### `GradTTSRomanian.from_local()`

```python
pipe = GradTTSRomanian.from_local(
    model_path="path/to/grad-tts.pt",
    vocoder_path="path/to/hifigan_univ_v1",
    vocoder_config_path="path/to/hifigan_config.json",
    config_path=None,   # Optional: model config JSON, uses defaults if None
    device=None,        # "cpu", "cuda", or auto-detect
)
```

### Synthesis

```python
# Single text
audio = pipe("Romanian text here.", n_timesteps=50, temperature=1.5, length_scale=0.91)

# Batch
audios = pipe(["Text one.", "Text two."])

# Save to file
pipe.save_wav(audio, "output.wav")
```

| Parameter | Default | Description |
|-----------|---------|-------------|
| `n_timesteps` | 50 | Diffusion reverse steps (more = higher quality, slower) |
| `temperature` | 1.5 | Sampling variance (higher = more diverse) |
| `length_scale` | 0.91 | Speech pace (> 1.0 = slower) |
| `speaker_id` | None | Speaker ID for multi-speaker models |

## Available Model Variants

| Variant | Speaker | Samples | Fine-tune Epochs |
|---------|---------|---------|-----------------|
| `base` | SWARA baseline | all | 1000 |
| `bas-10_15` | BAS | 10 | 15 |
| `bas-10_50` | BAS | 10 | 50 |
| `bas-10_100` | BAS | 10 | 100 |
| `bas-950_15` | BAS | 950 | 15 |
| `bas-950_50` | BAS | 950 | 50 |
| `bas-950_100` | BAS | 950 | 100 |
| `sgs-10_15` | SGS | 10 | 15 |
| `sgs-10_50` | SGS | 10 | 50 |
| `sgs-10_100` | SGS | 10 | 100 |
| `sgs-950_15` | SGS | 950 | 15 |
| `sgs-950_50` | SGS | 950 | 50 |
| `sgs-950_100` | SGS | 950 | 100 |

```python
# Load a specific variant
pipe = GradTTSRomanian.from_pretrained("adrianstanea/Ro-Grad-TTS", variant="bas-950_100")
```

## Model Weights

Checkpoints are hosted on HuggingFace Hub at [`adrianstanea/Ro-Grad-TTS`](https://huggingface.co/adrianstanea/Ro-Grad-TTS):

```
adrianstanea/Ro-Grad-TTS/
├── config.json
├── hifigan_config.json
├── models/
│   ├── swara/grad-tts-base-1000.pt
│   ├── bas/grad-tts-bas-{10,950}_{15,50,100}.pt
│   ├── sgs/grad-tts-sgs-{10,950}_{15,50,100}.pt
│   └── vocoder/hifigan_univ_v1
```

## Project Structure

```
Grad-TTS/
├── gradtts_ro/              # Inference package (pip installable)
│   ├── pipeline.py          # GradTTSRomanian class
│   ├── text_processing.py   # Espeak phonemization
│   ├── symbols.py           # IPA symbol set
│   ├── cleaners.py          # Text normalization
│   └── vocoder/             # Vendored HiFi-GAN
├── model/                   # Grad-TTS model architecture
│   ├── tts.py               # GradTTS module
│   ├── text_encoder.py      # Text encoder
│   ├── diffusion.py         # Diffusion decoder
│   └── monotonic_align/     # MAS C extension (requires build step)
├── examples/
│   └── synthesize.py        # Example synthesis script
├── tools/text_processing/   # Text processing (used by training scripts)
├── train.py                 # Training script
├── finetune.sh              # Speaker fine-tuning script
├── pyproject.toml           # Package definition
└── uv.lock                  # Reproducible dependency lock
```

## Training

Training requires the optional `[train]` dependencies and the SWARA 1.0 dataset.

```bash
uv sync --extra train
```

### Base Model

Edit `params.py` to point to your SWARA 1.0 filelists, then:

```bash
uv run python train.py
```

### Fine-tuning for Specific Speakers

```bash
uv run python train.py \
    --checkpoint_path path/to/base_model.pt \
    --train_filelist_path resources/filelists/SWARA_1_0/finetune_meta_bas_10.csv \
    --n_epochs 100 \
    --batch_size 5 \
    --log_dir log_dir/bas_10
```

Or use the batch fine-tuning script (edit paths inside first):

```bash
./finetune.sh
```

## Key Changes from Original Grad-TTS

- **Phonemization**: Espeak (Romanian) replaces CMUDict (English)
- **Model**: `spk_emb_dim` and `n_spks` passed to TextEncoder; mel cropping fix
- **Training**: Argparse support, checkpoint resume, SWARA 1.0 integration
- **Inference**: `gradtts_ro` package with HuggingFace Hub integration, CPU/CUDA auto-detection

## Citation

```bibtex
@article{popov2021grad,
  title={Grad-TTS: A Diffusion Probabilistic Model for Text-to-Speech},
  author={Popov, Vadim and Vovk, Ivan and Gogoryan, Vladimir and Sadekova, Tasnima and Kudinov, Mikhail},
  journal={International Conference on Machine Learning},
  year={2021}
}
```

## License

MIT License. See [LICENSE](LICENSE) for details.
