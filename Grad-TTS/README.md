# Grad-TTS for Romanian (Ro-Grad-TTS)

Adaptation of [Grad-TTS](https://arxiv.org/abs/2105.06337) (ICML 2021) for Romanian text-to-speech synthesis, trained on the [SWARA 1.0](https://huggingface.co/datasets/swara/swara1.0) dataset.

## Quick Start

### Installation

Requires Python >= 3.9 and [espeak-ng](https://github.com/espeak-ng/espeak-ng) installed on your system.

```bash
# Install espeak-ng (required for Romanian phonemization)
# Ubuntu/Debian:
sudo apt-get install espeak-ng
# macOS:
brew install espeak-ng

# Install the package
cd Grad-TTS
pip install -e .

# Build the monotonic alignment C extension
cd model/monotonic_align
python setup.py build_ext --inplace
cd ../..
```

Or using [uv](https://docs.astral.sh/uv/):

```bash
cd Grad-TTS
uv pip install -e .
```

### Synthesize Speech

**From HuggingFace Hub** (once models are published to `adrianstanea/Ro-Grad-TTS`):

```python
from gradtts_ro import GradTTSRomanian

pipe = GradTTSRomanian.from_pretrained("adrianstanea/Ro-Grad-TTS")
audio = pipe("Buna ziua! Aceasta este o demonstratie.")
pipe.save_wav(audio, "output.wav")
```

**From local checkpoints:**

```python
from gradtts_ro import GradTTSRomanian

pipe = GradTTSRomanian.from_local(
    model_path="checkpts/grad-tts-ro.pt",
    vocoder_path="checkpts/hifigan.pt",
    vocoder_config_path="checkpts/hifigan-config.json",
)
audio = pipe("Buna ziua! Aceasta este o demonstratie.")
pipe.save_wav(audio, "output.wav")
```

**Using the example script:**

```bash
# From HuggingFace Hub
python examples/synthesize.py

# From local checkpoints
python examples/synthesize.py \
    --model-path checkpts/grad-tts-ro.pt \
    --vocoder-path checkpts/hifigan.pt \
    --vocoder-config checkpts/hifigan-config.json

# Custom text
python examples/synthesize.py --text "Salut, cum te cheama?"
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

**Available variants:**

| Variant | Description |
|---------|-------------|
| `base` | Baseline SWARA model (1000 epochs) |
| `bas-10_100` | BAS speaker, 10 samples, 100 fine-tune epochs |
| `bas-950_100` | BAS speaker, 950 samples, 100 fine-tune epochs |
| `sgs-10_100` | SGS speaker, 10 samples, 100 fine-tune epochs |
| `sgs-950_100` | SGS speaker, 950 samples, 100 fine-tune epochs |

### `GradTTSRomanian.from_local()`

```python
pipe = GradTTSRomanian.from_local(
    model_path="path/to/grad-tts.pt",
    vocoder_path="path/to/hifigan",
    vocoder_config_path="path/to/hifigan-config.json",
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

**Parameters:**

| Parameter | Default | Description |
|-----------|---------|-------------|
| `n_timesteps` | 50 | Diffusion reverse steps (more = higher quality, slower) |
| `temperature` | 1.5 | Sampling variance (higher = more diverse) |
| `length_scale` | 0.91 | Speech pace (> 1.0 = slower) |
| `speaker_id` | None | Speaker ID for multi-speaker models |

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
│   └── monotonic_align/     # MAS C extension
├── examples/
│   └── synthesize.py        # Example synthesis script
├── tools/text_processing/   # Text processing (used by training scripts)
├── train.py                 # Training script
├── finetune.sh              # Speaker fine-tuning script
├── inference_RO.py          # Legacy inference script
├── config.json              # Model hyperparameters template
├── pyproject.toml           # Package definition
└── uv.lock                  # Reproducible dependency lock
```

## Training

Training requires the optional `[train]` dependencies and the SWARA 1.0 dataset.

```bash
pip install -e ".[train]"
```

### Base Model

Edit `params.py` to point to your SWARA 1.0 filelists, then:

```bash
python train.py
```

### Fine-tuning for Specific Speakers

```bash
python train.py \
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

## Model Weights

Model checkpoints are hosted on HuggingFace Hub at [`adrianstanea/Ro-Grad-TTS`](https://huggingface.co/adrianstanea/Ro-Grad-TTS):

```
adrianstanea/Ro-Grad-TTS/
├── config.json
├── hifigan_config.json
├── models/
│   ├── swara/grad-tts-base-1000.pt
│   ├── bas/grad-tts-bas-{10,950}_100.pt
│   ├── sgs/grad-tts-sgs-{10,950}_100.pt
│   └── vocoder/hifigan_univ_v1
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
