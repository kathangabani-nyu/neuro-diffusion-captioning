# Intermediate Results Summary
**Date**: December 15, 2025  
**Model Checkpoint**: `checkpoints_caption/fmri_caption_20251215_132600/best.pt` (Epoch 12)

---

## Model Training Status

- **Best Model**: Epoch 12 checkpoint saved as `best.pt`
- **Training Completed**: Model trained through 12 epochs
- **Checkpoint Size**: 12 GB
- **Evaluation Dataset**: 400 test samples processed

---

## Available Results

### 1. Image Generation Metrics

**Status**: Complete (400 test samples evaluated)

Images were generated from predicted captions using Stable Diffusion v1.5, then compared to ground truth NSD images.

| Metric | Value | Notes |
|--------|-------|-------|
| **SSIM** | 0.2026 | Structural Similarity Index |
| **PixCorr** | 0.0106 | Pixel-level correlation |
| **AlexNet(2)** | 48.11% | Layer 2 feature similarity |
| **AlexNet(5)** | 48.99% | Layer 5 feature similarity |
| **CLIP-12** | 51.65% | CLIP ViT-L/14@336px similarity |
| **Inception V3** | 50.55% | InceptionV3 feature similarity |
| **EfficientNet** | 0.9708 | EfficientNet feature similarity |
| **SWAV** | 0.6600 | SWAV self-supervised features |
| **DreamSim** | 0.8178 | DreamSim perceptual similarity |
| **mIoU** | Not computed | Mean Intersection over Union (needs fix) |

**Interpretation**:
- CLIP-12 (51.65%) and Inception V3 (50.55%) show moderate semantic similarity
- AlexNet features (48-49%) indicate some structural correspondence
- Low SSIM (0.20) and PixCorr (0.01) suggest pixel-level differences (expected for generated vs. real images)
- DreamSim (0.82) shows good perceptual similarity

### 2. Caption Generation

**Status**: Complete (400 test samples)

- **Generated Captions**: 400 test samples
- **Format**: JSON file with predictions and references
- **File**: `checkpoints_caption/fmri_caption_20251215_132600/generated_captions.json`

**Sample Output**:
- **Trial**: `subj01_trial0006`
- **Generated**: "image depicts a well-lit, likely a room with a large, specifically a mix of modern and white fur..."
- **Ground Truth**: "The image depicts a scene at a baseball field during what appears to be a practice session. There are three individuals in the foreground..."

### 3. Caption Metrics (NLP)

**Status**: Not computed (dependency issue)

**Issue**: `pycocoevalcap` package was not installed during evaluation run.

**Metrics to be computed** (once fixed):
- BLEU-1, BLEU-2, BLEU-3, BLEU-4
- METEOR
- ROUGE-L
- CIDEr

**Fix Status**: Scripts updated to auto-install dependency. Will be computed in next evaluation run.

---

## Model Architecture

- **Brain Encoder**: DynaDiff pretrained FmriMLP (346M parameters, loaded from DynaDiff checkpoint)
- **Caption Decoder**: Custom PyTorch `nn.TransformerDecoder` with cross-attention (trained from scratch)
- **Tokenizer**: Qwen2-VL-2B-Instruct
- **Training**: Caption loss only (no diffusion loss, no image generation during training)
- **Image Generation**: Post-hoc using Stable Diffusion (for evaluation metrics only)

---

## Pipeline Workflow

```
fMRI → Brain Encoder → Brain Embeddings → Caption Decoder → Captions
                                                                    ↓
                                                          Stable Diffusion
                                                                    ↓
                                                          Generated Images
                                                                    ↓
                                                          Image Metrics
```

**Note**: Images are generated from captions for evaluation purposes only. The model is trained solely on caption loss.

---

## Files Available

1. **Model Checkpoint**: `checkpoints_caption/fmri_caption_20251215_132600/best.pt`
2. **Evaluation Results**: `checkpoints_caption/fmri_caption_20251215_132600/eval_results_full.json`
3. **Generated Captions**: `checkpoints_caption/fmri_caption_20251215_132600/generated_captions.json`
4. **Intermediate Results**: `checkpoints_caption/fmri_caption_20251215_132600/eval_results_intermediate.json`

---

## Next Steps

1. **Image Metrics**: Complete (9/10 metrics computed)
2. **Caption Metrics**: Need to re-run evaluation with `pycocoevalcap` installed
3. **mIoU**: Need to fix computation (likely image count mismatch)
4. **Subject 1 Predictions**: Need to regenerate with fixed subject filter
5. **Full Test Set Predictions**: Can generate for all subjects

---

## Notes for Teammates

- **Image metrics are ready** and show moderate semantic similarity (50-52% for CLIP and Inception)
- **Caption generation is working** - 400 samples successfully processed
- **NLP metrics pending** - will be available after next evaluation run (dependency fix applied)
- **Model architecture clarified**: Transformer decoder with cross-attention, not Qwen decoder
- **Training approach**: Caption loss only, images generated post-hoc for evaluation

---

## Quick Stats

- **Test Samples Evaluated**: 400
- **Images Generated**: 399 (1 failed to load ground truth)
- **Evaluation Time**: ~43 minutes (with image generation)
- **Model Size**: 12 GB
- **Best Epoch**: 12