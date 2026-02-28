# Speech-Backbones (Romanian Adaptation Fork)

Fork of [Speech-Backbones](https://github.com/huawei-noah/Speech-Backbones) by Huawei Noah's Ark Lab, adapted for **Romanian text-to-speech** using the SWARA 1.0 dataset.

## Ro-Grad-TTS

The main contribution is in [`Grad-TTS/`](Grad-TTS/) — a Romanian adaptation of the Grad-TTS diffusion-based TTS model, packaged as an installable Python library with HuggingFace Hub integration.

### Quick Start

```bash
# Install
cd Grad-TTS
sudo apt-get install espeak-ng   # required for phonemization
pip install -e .
cd model/monotonic_align && python setup.py build_ext --inplace && cd ../..

# Synthesize (from HuggingFace Hub, once published)
python -c "
from gradtts_ro import GradTTSRomanian
pipe = GradTTSRomanian.from_pretrained('adrianstanea/Ro-Grad-TTS')
audio = pipe('Buna ziua!')
pipe.save_wav(audio, 'output.wav')
"
```

See [`Grad-TTS/README.md`](Grad-TTS/README.md) for full documentation, API reference, and training instructions.

### Model Weights

Pre-trained checkpoints are hosted at [`adrianstanea/Ro-Grad-TTS`](https://huggingface.co/adrianstanea/Ro-Grad-TTS) on HuggingFace Hub.

## Other Components (Original)

- **[SPIRAL](SPIRAL/)**: Self-supervised speech pre-training (ICLR 2022)
- **[DiffVC](DiffVC/)**: Diffusion-based voice conversion (ICLR 2022)

## License

See individual component directories for license terms.
