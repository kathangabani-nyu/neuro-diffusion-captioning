# fMRI-to-Caption Training Pipeline

Direct training pipeline for generating captions from fMRI brain signals.

## Overview

This pipeline implements an end-to-end fMRI→Caption model that:
1. Encodes fMRI brain signals into embeddings using DynaDiff pretrained brain encoder (FmriMLP)
2. Decodes brain embeddings into natural language captions using a Transformer decoder
3. Uses DynaDiff pretrained weights for brain encoder, trains caption decoder from scratch

## Architecture

### Brain Encoder
- **Component**: DynaDiff pretrained FmriMLP (346M parameters)
- **Input**: fMRI signals [B, num_voxels, 6] (variable voxel counts per subject)
- **Output**: Brain embeddings [B, 257, embed_dim]
- **Initialization**: Pretrained weights from DynaDiff checkpoint

### Caption Decoder
- **Component**: PyTorch `nn.TransformerDecoder` with cross-attention
- **Architecture**:
  - Token embeddings + positional embeddings
  - 4-6 layer TransformerDecoder with cross-attention to brain embeddings
  - Language model head for vocabulary prediction
- **Initialization**: Random (trained from scratch, ~50-100M parameters)

### Full Model
- **File**: `model/brain_caption_model.py`
- Combines brain encoder and caption decoder
- Supports gradient checkpointing for memory efficiency

## Training

### Configuration
- **Config File**: `config/train_config.yaml`
- **Key Settings**:
  - Batch size: 8 (with gradient accumulation: 8 steps = effective batch size 64)
  - Learning rate: 1e-4
  - Mixed precision: Enabled (FP16)
  - Gradient checkpointing: Enabled
  - Model size: ~1.05B parameters (embed_dim=512, n_layers=4)

### Training Script
```bash
sbatch slurm/train_caption.sbatch
```

### Features
- Image-disjoint train/val/test splits
- Automatic checkpointing (best model + periodic saves)
- GPU utilization monitoring
- Resume from checkpoint support

## Evaluation

### Caption Metrics
- BLEU-1, BLEU-2, BLEU-3, BLEU-4
- METEOR
- ROUGE-L
- CIDEr

### Image Metrics (via Stable Diffusion)
After generating captions, images are generated using Stable Diffusion and compared to ground truth:
- SSIM
- PixCorr
- AlexNet(2), AlexNet(5)
- CLIP-12
- Inception V3
- EfficientNet
- SWAV
- mIoU
- DreamSim

### Evaluation Script
```bash
sbatch slurm/evaluate_caption_with_images.sbatch
```

## Dataset

### Structure
- **Dataset Class**: `dataPrepartion/fmri_caption_dataset.py`
- **Features**:
  - Image-disjoint splits (no COCO image overlap between train/val/test)
  - Variable voxel count handling (different subjects have different voxel counts)
  - Qwen2-VL tokenizer integration
  - Custom collate function for padding

### Data Files
- `data/trial_mapping.json` - Maps NSD trials to COCO images
- `data/qwen_captions.json` - Qwen2-VL generated captions from original NSD images (ground truth)

## Key Implementation Details

### Variable Voxel Counts
- Different subjects have different numbers of voxels
- Custom collate function pads to max voxel count in batch
- `SubjectLayers` in FmriMLP handles variable counts by slicing to actual voxel count

### Memory Optimization
- Gradient checkpointing (recomputes activations during backward pass)
- Mixed precision training (FP16)
- Gradient accumulation (simulate larger batch size)
- Aggressive memory clearing (`torch.cuda.empty_cache()`)

### Tokenizer
- Uses Qwen2-VL tokenizer (same as caption generation)
- Handles BOS/EOS tokens with fallback logic

## Results

### Training Progress
- **Epoch 4**: Val Loss 6.11 (best model)
- **Epoch 9**: Train Loss 3.51 (43% reduction)
- **Status**: Training ongoing

### Evaluation
- Evaluation job running (91% complete, generating images)
- Results will be saved to `checkpoints_caption/*/eval_results_full.json`

## File Structure

```
fmri_caption_project/
├── model/
│   ├── transformer_captioner.py    # Caption decoder (Transformer)
│   ├── brain_encoder.py              # Brain encoder wrapper
│   ├── brain_caption_model.py       # Full model
│   └── fmri_mlp.py                  # FmriMLP brain encoder
├── train/
│   └── train_caption_model.py        # Training script
├── test/
│   ├── evaluate_caption_with_images.py  # Full evaluation
│   └── evaluate_caption_model.py        # Basic evaluation
├── dataPrepartion/
│   └── fmri_caption_dataset.py      # Dataset class
├── config/
│   └── train_config.yaml            # Training config
├── slurm/
│   ├── train_caption.sbatch         # Training SLURM script
│   └── evaluate_caption_with_images.sbatch  # Eval SLURM script
└── checkpoints_caption/             # Model checkpoints
```

## Dependencies

- PyTorch 2.4.0+
- transformers (for Qwen2-VL tokenizer)
- diffusers (for Stable Diffusion image generation)
- pycocoevalcap (for caption metrics)
- Existing `metrics/` code (for image metrics)

## Ground Truth

Ground truth captions are Qwen2-VL generated captions from original NSD images. We use high-quality vision-language model captions as ground truth for training and evaluation.

## Notes

- Brain encoder uses DynaDiff pretrained weights (346M parameters) for strong initialization
- Caption decoder is trained from scratch (~50-100M parameters)
- Ground truth captions from Qwen2-VL on original NSD images
- Image-disjoint splits ensure fair evaluation
- Variable voxel counts are handled automatically
- GPU memory optimized for 32GB GPUs

