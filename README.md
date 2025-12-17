# Neuro-Diffusion Captioning

End-to-end fMRI-to-caption model using DynaDiff pretrained brain encoder and Transformer decoder.

## Overview

Direct training pipeline for generating captions from fMRI brain signals:

1. **Brain Encoder**: DynaDiff pretrained FmriMLP (346M parameters) - encodes fMRI signals to CLIP space embeddings
2. **Caption Decoder**: Transformer decoder (trained from scratch) - generates captions from brain embeddings
3. **Training**: Caption loss only (cross-entropy on next-token prediction)
4. **Evaluation**: Comprehensive metrics on both captions and generated images

## Architecture

```
fMRI [B, voxels, TRs]
  ↓
DynaDiff Pretrained Brain Encoder (FmriMLP)
  ↓
Brain Embeddings [B, 257, 768] in CLIP Space
  ↓
Caption Decoder (Transformer)
  ↓
Generated Caption
  ↓
Cross-Entropy Loss
```

**Key Design Choices:**
- Leverages CLIP multimodal space for semantic alignment between brain signals and language
- Fine-tunes pretrained brain encoder while training decoder from scratch
- Direct optimization for caption quality (no image generation during training)
- Images generated post-hoc via Stable Diffusion for evaluation only

## Ground Truth

Ground truth captions are Qwen2-VL generated captions from original NSD images. We use high-quality vision-language model captions as ground truth for training and evaluation.

## Training

### Configuration
- Config: `config/train_config.yaml`
- Batch size: 8 (effective 64 with gradient accumulation)
- Learning rate: 1e-4
- Mixed precision: Enabled
- Gradient checkpointing: Enabled

### Run Training
```bash
sbatch slurm/train_caption.sbatch
```

## Evaluation

### Caption Metrics
- BLEU-1, BLEU-2, BLEU-3, BLEU-4
- METEOR
- ROUGE-L
- CIDEr

### Image Metrics
Images generated from captions using Stable Diffusion, then evaluated:
- SSIM, PixCorr
- AlexNet(2), AlexNet(5)
- CLIP-12
- Inception V3
- EfficientNet, SWAV, DreamSim, mIoU

### Run Evaluation
```bash
sbatch slurm/evaluate_caption_with_images.sbatch
```

## Dataset

- Image-disjoint train/val/test splits (no COCO image overlap)
- Variable voxel count handling per subject
- Qwen2-VL tokenizer integration
- Ground truth: Qwen2-VL captions from original NSD images

## Key Files

- `model/brain_caption_model.py` - Full model assembly
- `model/brain_encoder.py` - DynaDiff pretrained brain encoder
- `model/transformer_captioner.py` - Caption decoder
- `train/train_caption_model.py` - Training script
- `test/evaluate_caption_with_images.py` - Evaluation script
- `dataPrepartion/fmri_caption_dataset.py` - Dataset with image-disjoint splits

## Results

See `INTERMEDIATE_RESULTS_SUMMARY.md` for current evaluation results.

## Dependencies

- PyTorch 2.4.0+
- transformers (Qwen2-VL tokenizer)
- diffusers (Stable Diffusion for evaluation)
- pycocoevalcap (caption metrics)

See `requirements.txt` for complete list.

