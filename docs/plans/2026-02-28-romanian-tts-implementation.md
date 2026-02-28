# Romanian Grad-TTS Cleanup, Inference API & Packaging — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Clean up the `adaptation_romanian` branch into atomic commits, create a `GradTTSRomanian` inference API with HuggingFace Hub integration, and package the project with `uv`/`pyproject.toml`.

**Architecture:** A new `gradtts_ro/` Python package inside `Grad-TTS/` provides the inference API. It vendors HiFi-GAN code, imports the existing `model/` code, and uses `huggingface_hub` for weight downloading. The commit history is rewritten via interactive rebase into 6 logical commits. A `pyproject.toml` with `uv.lock` handles reproducible installs.

**Tech Stack:** Python (>=3.6.9, incrementing if needed), PyTorch, phonemizer/espeak, huggingface_hub, uv, hatchling

**Design doc:** `docs/plans/2026-02-28-romanian-tts-cleanup-design.md`

---

## Phase 1: Create the `gradtts_ro` Inference Package

This is the new code that needs to be written. We build it incrementally, test each piece, then commit.

### Task 1: Vendor HiFi-GAN into `gradtts_ro/vocoder/`

**Files:**
- Create: `Grad-TTS/gradtts_ro/__init__.py`
- Create: `Grad-TTS/gradtts_ro/vocoder/__init__.py`
- Create: `Grad-TTS/gradtts_ro/vocoder/models.py`
- Create: `Grad-TTS/gradtts_ro/vocoder/env.py`
- Create: `Grad-TTS/gradtts_ro/vocoder/xutils.py`

**Step 1: Create the package directory structure**

```bash
mkdir -p Grad-TTS/gradtts_ro/vocoder
```

**Step 2: Create `gradtts_ro/__init__.py`**

```python
from gradtts_ro.pipeline import GradTTSRomanian

__all__ = ["GradTTSRomanian"]
```

Note: This will fail to import until pipeline.py exists, which is fine — we'll create it in Task 3.

**Step 3: Vendor the HiFi-GAN files**

Copy `hifi-gan/env.py` → `gradtts_ro/vocoder/env.py` (unchanged).

Copy `hifi-gan/xutils.py` → `gradtts_ro/vocoder/xutils.py` (unchanged).

Copy `hifi-gan/models.py` → `gradtts_ro/vocoder/models.py` and fix the import:

```python
# Change line 8 from:
from xutils import init_weights, get_padding
# To:
from gradtts_ro.vocoder.xutils import init_weights, get_padding
```

**Step 4: Create `gradtts_ro/vocoder/__init__.py`**

```python
from gradtts_ro.vocoder.env import AttrDict
from gradtts_ro.vocoder.models import Generator as HiFiGAN

__all__ = ["AttrDict", "HiFiGAN"]
```

**Step 5: Verify vendored vocoder imports work**

```bash
cd Grad-TTS && python -c "from gradtts_ro.vocoder import HiFiGAN, AttrDict; print('OK')"
```

Expected: `OK`

**Step 6: Commit**

```bash
git add gradtts_ro/vocoder/ gradtts_ro/__init__.py
git commit -m "feat(gradtts_ro): vendor HiFi-GAN vocoder as subpackage"
```

---

### Task 2: Create text processing module in `gradtts_ro/`

**Files:**
- Create: `Grad-TTS/gradtts_ro/symbols.py`
- Create: `Grad-TTS/gradtts_ro/cleaners.py`
- Create: `Grad-TTS/gradtts_ro/text_processing.py`

**Step 1: Create `gradtts_ro/symbols.py`**

Copy from `tools/text_processing/symbols.py` verbatim:

```python
"""
From: https://github.com/keithito/tacotron

Defines the set of symbols used in text input to the model.
"""

_pad        = '_'
_punctuation = ';:,.!?¡¿—…"«»"" '
_letters = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz'
_letters_ipa = "ɑɐɒæɓʙβɔɕçɗɖðʤəɘɚɛɜɝɞɟʄɡɠɢʛɦɧħɥʜɨɪʝɭɬɫɮʟɱɯɰŋɳɲɴøɵɸθœɶʘɹɺɾɻʀʁɽʂʃʈʧʉʊʋⱱʌɣɤʍχʎʏʑʐʒʔʡʕʢǀǁǂǃˈˌːˑʼʴʰʱʲʷˠˤ˞↓↑→↗↘'̩'ᵻ̃"

# Export all symbols:
symbols = [_pad] + list(_punctuation) + list(_letters) + list(_letters_ipa)

# Special symbol ids
SPACE_ID = symbols.index(" ")

# Mappings from symbol to numeric ID and vice versa:
_symbol_to_id = {s: i for i, s in enumerate(symbols)}
_id_to_symbol = {i: s for i, s in enumerate(symbols)}
```

**Step 2: Create `gradtts_ro/cleaners.py`**

Copy from `tools/text_processing/cleaners.py` verbatim:

```python
""" from https://github.com/gmltmd789/UnitSpeech.git """

import re
from unidecode import unidecode

_whitespace_re = re.compile(r'\s+')

def to_lowercase(text):
    return text.lower()

def collapse_whitespace(text):
    return re.sub(_whitespace_re, ' ', text)

def to_ascii(text):
    return unidecode(text)
```

**Step 3: Create `gradtts_ro/text_processing.py`**

```python
from phonemizer.backend import EspeakBackend

from gradtts_ro.symbols import symbols, _symbol_to_id
from gradtts_ro.cleaners import to_lowercase, collapse_whitespace


def create_phonemizer():
    """Create an Espeak backend for Romanian phonemization."""
    return EspeakBackend(
        language='ro',
        preserve_punctuation=True,
        with_stress=True,
        language_switch="remove-flags",
    )


def text_to_phoneme(text, phonemizer):
    """Convert text to phoneme string using the given phonemizer backend."""
    text = to_lowercase(text)
    phonemes = phonemizer.phonemize([text], strip=True)[0]
    phonemes = collapse_whitespace(phonemes)
    return phonemes


def cleaned_text_to_sequence(cleaned_text):
    """Converts a string of phonemes to a sequence of IDs corresponding to the symbols."""
    return [_symbol_to_id[symbol] for symbol in cleaned_text]


def intersperse(lst, item):
    """Adds blank symbol between each element."""
    result = [item] * (len(lst) * 2 + 1)
    result[1::2] = lst
    return result
```

**Step 4: Verify text processing works**

```bash
cd Grad-TTS && python -c "
from gradtts_ro.text_processing import create_phonemizer, text_to_phoneme, cleaned_text_to_sequence, intersperse
from gradtts_ro.symbols import symbols
ph = create_phonemizer()
phonemes = text_to_phoneme('Salut', ph)
seq = cleaned_text_to_sequence(phonemes)
seq = intersperse(seq, len(symbols))
print(f'Phonemes: {phonemes}, Sequence length: {len(seq)}')
"
```

Expected: Prints phonemes and sequence length without errors.

**Step 5: Commit**

```bash
git add gradtts_ro/symbols.py gradtts_ro/cleaners.py gradtts_ro/text_processing.py
git commit -m "feat(gradtts_ro): add text processing module with Espeak phonemization"
```

---

### Task 3: Create the `GradTTSRomanian` pipeline class

**Files:**
- Create: `Grad-TTS/gradtts_ro/pipeline.py`
- Modify: `Grad-TTS/gradtts_ro/__init__.py`

**Step 1: Create `gradtts_ro/pipeline.py`**

```python
import json
import os
from pathlib import Path
from typing import List, Optional, Union

import numpy as np
import torch
from scipy.io.wavfile import write as write_wav

from gradtts_ro.text_processing import (
    create_phonemizer,
    text_to_phoneme,
    cleaned_text_to_sequence,
    intersperse,
)
from gradtts_ro.symbols import symbols
from gradtts_ro.vocoder import HiFiGAN, AttrDict

# Import model from the existing Grad-TTS code
import sys
_grad_tts_root = str(Path(__file__).resolve().parent.parent)
if _grad_tts_root not in sys.path:
    sys.path.insert(0, _grad_tts_root)
from model import GradTTS


# Default model hyperparameters (matching the trained Romanian model)
DEFAULT_CONFIG = {
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
    "add_blank": True,
}

# Variant name -> (subfolder, filename) in the HuggingFace repo
VARIANT_MAP = {
    "base": ("models/swara", "grad-tts-base-1000.pt"),
    "bas-10_100": ("models/bas", "grad-tts-bas-10_100.pt"),
    "bas-950_100": ("models/bas", "grad-tts-bas-950_100.pt"),
    "sgs-10_100": ("models/sgs", "grad-tts-sgs-10_100.pt"),
    "sgs-950_100": ("models/sgs", "grad-tts-sgs-950_100.pt"),
}

HF_VOCODER_SUBFOLDER = "models/vocoder"
HF_VOCODER_FILENAME = "hifigan_univ_v1"


class GradTTSRomanian:
    """Romanian Grad-TTS text-to-speech pipeline.

    Usage:
        # From HuggingFace Hub
        pipe = GradTTSRomanian.from_pretrained("adrianstanea/Ro-Grad-TTS")
        audio = pipe("Salut!")
        pipe.save_wav(audio, "output.wav")

        # From local files
        pipe = GradTTSRomanian.from_local(
            model_path="checkpts/grad-tts-ro.pt",
            vocoder_path="checkpts/hifigan_univ_v1",
            vocoder_config_path="checkpts/hifigan-config.json",
        )
    """

    def __init__(self, generator, vocoder, phonemizer, config, device):
        self.generator = generator
        self.vocoder = vocoder
        self.phonemizer = phonemizer
        self.config = config
        self.device = device

    @classmethod
    def from_pretrained(
        cls,
        repo_id="adrianstanea/Ro-Grad-TTS",
        variant="base",
        device=None,
        revision=None,
        cache_dir=None,
    ):
        """Load model from a HuggingFace Hub repository.

        Args:
            repo_id: HuggingFace repository ID.
            variant: Model variant to load. One of:
                "base", "bas-10_100", "bas-950_100", "sgs-10_100", "sgs-950_100".
            device: Device to load model on. Auto-detects if None.
            revision: Git revision (branch/tag/commit) on the HF repo.
            cache_dir: Where to cache downloaded files.
        """
        from huggingface_hub import hf_hub_download

        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            device = torch.device(device)

        # Download config
        config_path = hf_hub_download(
            repo_id, "config.json",
            revision=revision, cache_dir=cache_dir,
        )
        with open(config_path) as f:
            config = json.load(f)

        # Download model checkpoint
        if variant not in VARIANT_MAP:
            raise ValueError(
                f"Unknown variant '{variant}'. Choose from: {list(VARIANT_MAP.keys())}"
            )
        subfolder, filename = VARIANT_MAP[variant]
        model_path = hf_hub_download(
            repo_id, filename, subfolder=subfolder,
            revision=revision, cache_dir=cache_dir,
        )

        # Download vocoder
        vocoder_config_path = hf_hub_download(
            repo_id, "hifigan_config.json",
            revision=revision, cache_dir=cache_dir,
        )
        vocoder_path = hf_hub_download(
            repo_id, HF_VOCODER_FILENAME, subfolder=HF_VOCODER_SUBFOLDER,
            revision=revision, cache_dir=cache_dir,
        )

        return cls._build(config, model_path, vocoder_path, vocoder_config_path, device)

    @classmethod
    def from_local(
        cls,
        model_path,
        vocoder_path,
        vocoder_config_path,
        config_path=None,
        device=None,
    ):
        """Load model from local file paths.

        Args:
            model_path: Path to Grad-TTS checkpoint (.pt file).
            vocoder_path: Path to HiFi-GAN vocoder checkpoint.
            vocoder_config_path: Path to HiFi-GAN config JSON.
            config_path: Path to model config JSON. Uses defaults if None.
            device: Device to load model on. Auto-detects if None.
        """
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            device = torch.device(device)

        if config_path is not None:
            with open(config_path) as f:
                config = json.load(f)
        else:
            config = dict(DEFAULT_CONFIG)

        return cls._build(config, model_path, vocoder_path, vocoder_config_path, device)

    @classmethod
    def _build(cls, config, model_path, vocoder_path, vocoder_config_path, device):
        """Internal builder: constructs models from config and paths."""
        # Build Grad-TTS generator
        n_vocab = len(symbols) + 1 if config.get("add_blank", True) else len(symbols)
        generator = GradTTS(
            n_vocab,
            config["n_spks"],
            config.get("spk_emb_dim"),
            config["n_enc_channels"],
            config["filter_channels"],
            config["filter_channels_dp"],
            config["n_heads"],
            config["n_enc_layers"],
            config["enc_kernel"],
            config["enc_dropout"],
            config["window_size"],
            config["n_feats"],
            config["dec_dim"],
            config["beta_min"],
            config["beta_max"],
            config["pe_scale"],
        )
        generator.load_state_dict(
            torch.load(model_path, map_location=device)
        )
        generator = generator.to(device).eval()

        # Build HiFi-GAN vocoder
        with open(vocoder_config_path) as f:
            h = AttrDict(json.load(f))
        vocoder = HiFiGAN(h)
        vocoder_state = torch.load(vocoder_path, map_location=device)
        if "generator" in vocoder_state:
            vocoder_state = vocoder_state["generator"]
        vocoder.load_state_dict(vocoder_state)
        vocoder = vocoder.to(device).eval()
        vocoder.remove_weight_norm()

        # Create phonemizer
        phonemizer = create_phonemizer()

        return cls(generator, vocoder, phonemizer, config, device)

    def __call__(
        self,
        text,
        n_timesteps=50,
        temperature=1.5,
        length_scale=0.91,
        speaker_id=None,
    ):
        """Synthesize speech from text.

        Args:
            text: A string or list of strings to synthesize.
            n_timesteps: Number of diffusion reverse steps.
            temperature: Controls variance of terminal distribution.
            length_scale: Controls speech pace (>1 = slower).
            speaker_id: Speaker ID for multi-speaker models.

        Returns:
            A numpy int16 array (single text) or list of arrays (batch).
        """
        if isinstance(text, str):
            return self._synthesize_one(text, n_timesteps, temperature, length_scale, speaker_id)
        return [
            self._synthesize_one(t, n_timesteps, temperature, length_scale, speaker_id)
            for t in text
        ]

    def _synthesize_one(self, text, n_timesteps, temperature, length_scale, speaker_id):
        """Synthesize a single utterance."""
        # Text to phoneme sequence
        phonemes = text_to_phoneme(text, self.phonemizer)
        seq = cleaned_text_to_sequence(phonemes)
        if self.config.get("add_blank", True):
            seq = intersperse(seq, len(symbols))

        x = torch.LongTensor(seq).to(self.device).unsqueeze(0)
        x_lengths = torch.LongTensor([x.shape[-1]]).to(self.device)

        spk = None
        if speaker_id is not None:
            spk = torch.LongTensor([speaker_id]).to(self.device)

        with torch.no_grad():
            _, y_dec, _ = self.generator(
                x, x_lengths,
                n_timesteps=n_timesteps,
                temperature=temperature,
                stoc=False,
                spk=spk,
                length_scale=length_scale,
            )
            audio = self.vocoder(y_dec).cpu().squeeze().clamp(-1, 1).numpy()

        return (audio * 32768).astype(np.int16)

    @property
    def sample_rate(self):
        return self.config.get("sample_rate", 22050)

    def save_wav(self, audio, path):
        """Save audio array to a WAV file.

        Args:
            audio: numpy int16 array from __call__.
            path: Output file path.
        """
        write_wav(path, self.sample_rate, audio)
```

**Step 2: Update `gradtts_ro/__init__.py`** (already written in Task 1, should work now)

Verify the import works:

```bash
cd Grad-TTS && python -c "from gradtts_ro import GradTTSRomanian; print('OK')"
```

Expected: `OK`

**Step 3: Test `from_local()` with existing checkpoints**

This test requires the checkpoint files. If they exist locally:

```bash
cd Grad-TTS && python -c "
from gradtts_ro import GradTTSRomanian
pipe = GradTTSRomanian.from_local(
    model_path='checkpts/grad-tts-ro.pt',
    vocoder_path='checkpts/hifigan.pt',
    vocoder_config_path='checkpts/hifigan-config.json',
    device='cpu',
)
audio = pipe('Salut, aceasta este un test.')
print(f'Audio shape: {audio.shape}, dtype: {audio.dtype}, sample_rate: {pipe.sample_rate}')
pipe.save_wav(audio, 'out/test_pipeline.wav')
print('Saved test_pipeline.wav')
"
```

Expected: Prints audio shape, dtype=int16, sample_rate=22050, saves WAV file.

If checkpoints don't exist locally, verify the class instantiation logic is correct by checking the constructor path only:

```bash
cd Grad-TTS && python -c "
from gradtts_ro.pipeline import GradTTSRomanian, VARIANT_MAP, DEFAULT_CONFIG
print('Variants:', list(VARIANT_MAP.keys()))
print('Config keys:', list(DEFAULT_CONFIG.keys()))
print('Pipeline class loaded OK')
"
```

**Step 4: Commit**

```bash
git add gradtts_ro/pipeline.py
git commit -m "feat(gradtts_ro): add GradTTSRomanian pipeline with from_pretrained and from_local"
```

---

## Phase 2: Packaging with pyproject.toml and uv

### Task 4: Create `pyproject.toml` and generate `uv.lock`

**Files:**
- Create: `Grad-TTS/pyproject.toml`

**Step 1: Create `Grad-TTS/pyproject.toml`**

```toml
[project]
name = "gradtts-ro"
version = "0.1.0"
description = "Romanian Grad-TTS: Diffusion-based Text-to-Speech"
readme = "README.md"
license = "MIT"
requires-python = ">=3.6.9"
dependencies = [
    "torch>=1.9.0",
    "torchaudio>=0.9.0",
    "numpy>=1.19.5",
    "scipy>=1.5.4",
    "phonemizer>=3.2.1",
    "huggingface-hub>=0.14.0",
    "Unidecode>=1.1.2",
]

[project.optional-dependencies]
train = [
    "matplotlib>=3.3.3",
    "tensorboard>=2.10.1",
    "tqdm",
    "Cython>=0.29.23",
    "librosa>=0.8.0",
    "inflect>=5.0.2",
]

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["gradtts_ro"]
```

**Step 2: Verify `requires-python` works**

Try to resolve dependencies with uv (if installed):

```bash
cd Grad-TTS && uv lock
```

If Python 3.6.9 causes resolution failures, increment `requires-python` to `">=3.7"` and retry. Continue incrementing (3.8, 3.9) until resolution succeeds. Document the final version.

**Step 3: Verify editable install works**

```bash
cd Grad-TTS && uv pip install -e .
```

Or with pip:

```bash
cd Grad-TTS && pip install -e .
```

Then verify:

```bash
python -c "import gradtts_ro; print('Package installed OK')"
```

**Step 4: Commit**

```bash
git add pyproject.toml
# If uv.lock was generated:
git add uv.lock
git commit -m "build: add pyproject.toml for installable gradtts-ro package"
```

---

### Task 5: Create `config.json` for the HuggingFace repo

**Files:**
- Create: `Grad-TTS/config.json` (this is a template/reference that will be uploaded to HF)

**Step 1: Create `Grad-TTS/config.json`**

```json
{
    "model_type": "grad-tts",
    "language": "ro",
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

**Step 2: Commit**

```bash
git add config.json
git commit -m "feat: add config.json template for HuggingFace Hub"
```

---

## Phase 3: Git History Cleanup

### Task 6: Rewrite commit history into atomic logical units

**IMPORTANT:** This task rewrites git history. Back up the current branch first.

**Step 1: Verify backup branch exists**

```bash
git branch backup-adaptation-romanian-pre-cleanup adaptation_romanian
```

**Step 2: Plan the interactive rebase**

The target is to squash 25+ commits (from `main` to HEAD) into 6 atomic commits. Since interactive rebase with -i is not supported non-interactively in this environment, we use a **soft-reset + re-commit approach**.

Create a new clean branch from main:

```bash
git checkout main
git checkout -b adaptation_romanian_clean
```

**Step 3: Apply Commit 1 — Romanian text processing**

Cherry-pick the relevant changes or apply diffs for:
- Add `tools/text_processing/` directory (symbols.py, cleaners.py, __init__.py, LICENSE)
- Remove `text/` directory (CMUDict-based processing)
- Update `data.py` to use Espeak phonemization
- Update `inference.py` imports
- Update `train_multi_speaker.py` imports
- Add `phonemizer>=3.2.1` to requirements.txt
- Remove `resources/cmu_dictionary`

```bash
# Get the files from adaptation_romanian
git checkout adaptation_romanian -- Grad-TTS/tools/text_processing/
git checkout adaptation_romanian -- Grad-TTS/data.py
git checkout adaptation_romanian -- Grad-TTS/inference.py
git checkout adaptation_romanian -- Grad-TTS/train_multi_speaker.py
git checkout adaptation_romanian -- Grad-TTS/requirements.txt
# Remove CMUDict text processing
git rm -r Grad-TTS/text/
git rm Grad-TTS/resources/cmu_dictionary

git add -A
git commit -m "feat: replace CMUDict with Espeak for Romanian phonemization

- Add tools/text_processing/ with Espeak backend for Romanian language
- Update data.py to use phonemizer instead of CMUDict
- Update inference.py and train_multi_speaker.py imports
- Remove text/ directory (CMUDict-based processing)
- Remove resources/cmu_dictionary
- Add phonemizer, torchaudio, tensorboard to requirements.txt"
```

**Step 4: Apply Commit 2 — Model architecture changes**

```bash
git checkout adaptation_romanian -- Grad-TTS/model/tts.py

git add Grad-TTS/model/tts.py
git commit -m "fix: adapt model architecture for Romanian multi-speaker TTS

- Pass spk_emb_dim and n_spks to TextEncoder constructor
- Fix mel cropping logic: add out_size < y_max_length guard
- Fix sequence_mask call to include out_size parameter"
```

**Step 5: Apply Commit 3 — Training pipeline**

```bash
git checkout adaptation_romanian -- Grad-TTS/params.py
git checkout adaptation_romanian -- Grad-TTS/train.py
git checkout adaptation_romanian -- Grad-TTS/finetune.sh
git checkout adaptation_romanian -- Grad-TTS/resources/filelists/SWARA_1_0/
git checkout adaptation_romanian -- .gitignore
git checkout adaptation_romanian -- Grad-TTS/.gitignore

git add -A
git commit -m "feat: configure training pipeline for SWARA 1.0 Romanian dataset

- Update params.py for SWARA 1.0 filelists and training config
- Add argparse support and checkpoint resume to train.py
- Add finetune.sh for speaker-specific fine-tuning (BAS, SGS)
- Add SWARA 1.0 filelists (train, val, finetune metadata)
- Add .gitignore entries for logs and build artifacts"
```

**Step 6: Apply Commit 4 — Romanian inference pipeline**

```bash
git checkout adaptation_romanian -- Grad-TTS/inference_RO.py
git checkout adaptation_romanian -- Grad-TTS/inference_RO.sh

git add Grad-TTS/inference_RO.py Grad-TTS/inference_RO.sh
git commit -m "feat: add Romanian inference pipeline

- Add inference_RO.py with Romanian text-to-speech synthesis
- Add inference_RO.sh for batch inference across checkpoints"
```

**Step 7: Apply Commit 5 — Inference package + packaging**

```bash
# Copy the gradtts_ro package (created in Phase 1)
git checkout adaptation_romanian -- Grad-TTS/gradtts_ro/
git checkout adaptation_romanian -- Grad-TTS/pyproject.toml
git checkout adaptation_romanian -- Grad-TTS/config.json
# Include uv.lock if it exists
git checkout adaptation_romanian -- Grad-TTS/uv.lock 2>/dev/null || true

git add -A
git commit -m "feat: add installable gradtts_ro package with HuggingFace integration

- Add gradtts_ro/ package with GradTTSRomanian pipeline class
- Support from_pretrained() for HuggingFace Hub model loading
- Support from_local() for local checkpoint loading
- Auto-detect CPU/CUDA device
- Vendor HiFi-GAN vocoder as gradtts_ro.vocoder subpackage
- Add pyproject.toml for uv/pip installable package
- Add config.json template for HuggingFace Hub"
```

**Step 8: Apply Commit 6 — Documentation**

```bash
git checkout adaptation_romanian -- README.md
git checkout adaptation_romanian -- Grad-TTS/README.md
git checkout adaptation_romanian -- docs/

git add -A
git commit -m "docs: document Romanian TTS adaptation and usage"
```

**Step 9: Verify the clean branch**

```bash
git log --oneline
# Should show exactly 6 clean commits on top of main

# Verify the diff matches the original (minus checkpoints)
git diff adaptation_romanian -- ':!Grad-TTS/checkpts/' ':!docs/plans/'
# Should show minimal/no differences (only the new gradtts_ro package and pyproject.toml additions)
```

**Step 10: Replace the original branch**

```bash
git branch -m adaptation_romanian adaptation_romanian_old
git branch -m adaptation_romanian_clean adaptation_romanian
```

---

## Phase 4: Final Verification

### Task 7: End-to-end verification

**Step 1: Verify all imports work from a clean state**

```bash
cd Grad-TTS
python -c "
from gradtts_ro import GradTTSRomanian
from gradtts_ro.text_processing import create_phonemizer, text_to_phoneme
from gradtts_ro.symbols import symbols
from gradtts_ro.vocoder import HiFiGAN, AttrDict
print('All imports OK')
print(f'Symbol count: {len(symbols)}')
"
```

**Step 2: Verify the model/ code still imports correctly**

```bash
cd Grad-TTS
python -c "
from model import GradTTS
from model.utils import fix_len_compatibility
print('Model imports OK')
"
```

Note: This requires the monotonic_align C extension to be built. If not built:
```bash
cd model/monotonic_align && python setup.py build_ext --inplace && cd ../..
```

**Step 3: Verify git log is clean**

```bash
git log --oneline main..adaptation_romanian
```

Expected: 6 clean commits.

**Step 4: Verify no checkpoints in git**

```bash
git ls-files Grad-TTS/checkpts/
```

Expected: Empty (no checkpoint files tracked).

---

## Summary of Execution Order

1. **Task 1**: Vendor HiFi-GAN → `gradtts_ro/vocoder/`
2. **Task 2**: Create text processing module → `gradtts_ro/text_processing.py`
3. **Task 3**: Create `GradTTSRomanian` pipeline class → `gradtts_ro/pipeline.py`
4. **Task 4**: Create `pyproject.toml` + `uv.lock`
5. **Task 5**: Create `config.json` template
6. **Task 6**: Rewrite git history (6 atomic commits)
7. **Task 7**: End-to-end verification

Tasks 1-5 are sequential (each builds on the previous). Task 6 depends on Tasks 1-5 being done. Task 7 is the final check.
