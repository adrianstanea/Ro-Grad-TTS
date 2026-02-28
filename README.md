# Speech-Backbones (Romanian Adaptation Fork)

This is a fork of the original [Speech-Backbones repository](https://github.com/huawei-noah/Speech-Backbones) by Huawei Noah's Ark Lab, adapted for Romanian text-to-speech synthesis using the SWARA 1.0 dataset.

## Original Repository

The original repository contains open-sourced speech technology implementations including:
- **Grad-TTS**: Diffusion-based text-to-speech model (ICML 2021)
- **SPIRAL**: Self-supervised speech pre-training (ICLR 2022)
- **DiffVC**: Diffusion-based voice conversion (ICLR 2022, Oral)

## Romanian Adaptation Work

This fork focuses on adapting Grad-TTS for Romanian language synthesis.

### Key Contributions

1. **Romanian Language Support**
   - Integration with SWARA 1.0 Romanian speech dataset
   - Espeak-based phonemization replacing CMUDict
   - Romanian-specific text processing pipeline

2. **Multi-Speaker Fine-tuning**
   - Support for BAS and SGS speaker identities
   - Speaker embedding dimension configuration
   - Fine-tuning scripts for few-shot adaptation

3. **Enhanced Inference Pipeline**
   - Romanian-specific inference script (`inference_RO.py`)
   - Automated sample generation workflow
   - Real-time factor (RTF) measurement tools

## Training Workflow

### Prerequisites

1. Install dependencies:
   ```bash
   cd Grad-TTS
   pip install -r requirements.txt
   ```

2. Build monotonic alignment:
   ```bash
   cd model/monotonic_align
   python setup.py build_ext --inplace
   cd ../..
   ```

3. Prepare SWARA 1.0 dataset filelists in `resources/filelists/SWARA_1_0/`

### Base Model Training

Train base Romanian model:

```bash
export CUDA_VISIBLE_DEVICES=0
python train_multi_speaker.py
```

### Fine-tuning for Specific Speakers

Use the provided fine-tuning script:

```bash
chmod +x finetune.sh
./finetune.sh
```

This script fine-tunes the base model for:
- BAS speaker (10 and 950 sample variants)
- SGS speaker (10 and 950 sample variants)

## Inference Workflow

### Quick Start

1. Download or use trained checkpoints from `checkpts/` directory
2. Prepare text file with Romanian sentences (format: `filepath|text|speaker`)
3. Run Romanian inference:

```bash
python inference_RO.py \
  -f resources/filelists/synthesis.txt \
  -c checkpts/grad-tts-ro-bas-100.pt \
  -t 50 \
  --temperature 1.5
```

### Parameters

- `-f, --file`: Path to text file with synthesis targets
- `-c, --checkpoint`: Path to trained Grad-TTS checkpoint
- `-t, --timesteps`: Number of diffusion timesteps (default: 50)
- `-s, --speaker_id`: Speaker ID for multi-speaker models
- `--temperature`: Sampling temperature (default: 1.5)

### Output

Generated audio files are saved to timestamped directories under `/workspace/local/samples/grad-tts/{speaker_id}/{epochs}/`

### Checkpoint Naming Convention

Checkpoints follow the pattern: `grad-tts-ro-{speaker_id}-{epochs}.pt`
- `speaker_id`: Target speaker (bas, sgs, etc.)
- `epochs`: Number of training epochs

## Generated Samples

This repository includes pre-trained models and generated samples:

### Available Checkpoints

- `grad-tts-ro.pt`: Base Romanian model
- Fine-tuned speaker-specific models (created via finetune.sh)

### Sample Audio

Generated samples demonstrate:
- Romanian text-to-speech synthesis quality
- Speaker adaptation capabilities
- Various text lengths and complexities

Samples are generated using the inference pipeline and stored with RTF measurements for performance evaluation.

## Model Architecture Details

### Text Processing
- **Phonemization**: Espeak for Romanian language
- **Text Encoding**: Character and phoneme-based encoding
- **Speaker Embedding**: Configurable embedding dimensions

### Training Configuration
- **Dataset**: SWARA 1.0 Romanian speech corpus
- **Speakers**: Multi-speaker support (BAS, SGS)
- **Sampling Rate**: 22kHz
- **Vocoder**: HiFi-GAN for audio generation

## Technical Implementation

### Key Modifications from Original

1. **Text Processing Pipeline**
   - Replaced CMUDict with Espeak for Romanian phonemization
   - Added Romanian-specific text cleaning and normalization
   - Updated symbol set for Romanian phonemes

2. **Model Configuration**
   - Speaker embedding dimension: Configurable via `spk_emb_dim`
   - Multi-speaker support: `n_spks` parameter adjustment
   - Romanian phoneme set integration

3. **Training Enhancements**
   - Checkpoint resuming functionality
   - Validation step configuration
   - Batch size adaptation for few-shot learning

### File Structure

```
Grad-TTS/
├── checkpts/                 # Model checkpoints
│   ├── grad-tts-ro.pt       # Base Romanian model
│   └── hifigan.pt           # Vocoder model
├── resources/
│   └── filelists/SWARA_1_0/  # Romanian dataset filelists
├── inference_RO.py          # Romanian-specific inference
├── finetune.sh             # Speaker fine-tuning script
└── tools/text_processing/   # Romanian text processing
```

### Performance

- Real-time Factor (RTF) measurement included
- Support for various inference speeds via timestep control
- Temperature-controlled synthesis quality

## Usage Examples

### Example 1: Basic Romanian Synthesis

```bash
# Create text file
echo "filepath1|Salut, aceasta este o propoziție în română.|speaker1" > test.txt

# Run inference
python inference_RO.py -f test.txt -c checkpts/grad-tts-ro.pt -t 50
```

### Example 2: Speaker-Specific Fine-tuning

```bash
# Fine-tune for BAS speaker with limited data
python train.py \
  --checkpoint_path checkpts/grad-tts-ro.pt \
  --n_epochs 100 \
  --train_filelist_path resources/filelists/SWARA_1_0/finetune_meta_bas_10.csv \
  --batch_size 5 \
  --log_dir log_dir/bas_10
```

### Example 3: Batch Inference

```bash
# Use the shell script for automated fine-tuning
chmod +x finetune.sh
./finetune.sh
```

This will process all configured speaker/sample combinations automatically.

## Citation

If you use this Romanian adaptation, please cite both the original work and the SWARA dataset:

```bibtex
@article{popov2021grad,
  title={Grad-TTS: A Diffusion Probabilistic Model for Text-to-Speech},
  author={Popov, Vadim and Vovk, Ivan and Gogoryan, Vladimir and Sadekova, Tasnima and Kudinov, Mikhail},
  journal={International Conference on Machine Learning},
  year={2021}
}
```

## License

This project maintains the same license as the original Speech-Backbones repository. See individual component directories for specific license terms.