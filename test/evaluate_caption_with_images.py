#!/usr/bin/env python3
"""
Enhanced evaluation script for fMRI → Caption model.

1. Generates captions from fMRI
2. Generates images from captions (using text-to-image model)
3. Loads ground truth COCO images
4. Computes image metrics: SSIM, PixCorr, AlexNet(2), AlexNet(5), CLIP-12, 
   InceptionV3, EfficientNet, SwAV, DreamSim, mIoU
5. Also computes caption metrics: BLEU, METEOR, ROUGE-L, CIDEr
"""

import argparse
import json
from pathlib import Path
from tqdm import tqdm
import sys
import time
from datetime import datetime

import torch
from torch.utils.data import DataLoader
import yaml
from PIL import Image
import numpy as np
import json

from transformers import AutoTokenizer

# Standard captioning metrics (pycocoevalcap)
try:
    from pycocoevalcap.bleu.bleu import Bleu
    from pycocoevalcap.meteor.meteor import Meteor
    from pycocoevalcap.rouge.rouge import Rouge
    from pycocoevalcap.cider.cider import Cider
    METRICS_AVAILABLE = True
except ImportError:
    print("Warning: pycocoevalcap not available. Install with: pip install pycocoevalcap")
    METRICS_AVAILABLE = False

# Image metrics
try:
    from metrics.image_metrics import compute_image_generation_metrics
    from metrics.mIOU.evaluate_img_gen import compute_miou
    IMAGE_METRICS_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Image metrics not available: {e}")
    IMAGE_METRICS_AVAILABLE = False

# Import dataset and model
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
from dataPrepartion.fmri_caption_dataset import FmriCaptionDataset
from model.brain_caption_model import BrainCaptionModel


def load_nsd_image(trial_mapping: dict, trial_key: str, nsddata_path: str = None) -> Image.Image:
    """
    Load NSD image using trial mapping.
    
    Args:
        trial_mapping: Trial mapping dictionary
        trial_key: Trial key (e.g., "subj01_trial0001")
        nsddata_path: Path to NSD data root (optional, not needed if image_path is in mapping)
    
    Returns:
        PIL.Image: Loaded image (resized to 512x512 for consistency)
    """
    # Parse trial key: "subj01_trial0001" -> subject="subj01", trial="0001"
    parts = trial_key.split("_trial")
    if len(parts) != 2:
        raise ValueError(f"Invalid trial key format: {trial_key}")
    
    subj_key = parts[0]
    trial_id = str(int(parts[1]))  # Convert "0006" -> "6" to match mapping
    
    # Get trial info from mapping
    if subj_key not in trial_mapping:
        raise KeyError(f"Subject {subj_key} not found in trial mapping")
    
    if trial_id not in trial_mapping[subj_key]:
        raise KeyError(f"Trial {trial_id} not found for {subj_key}")
    
    trial_info = trial_mapping[subj_key][trial_id]
    
    # Use image_path directly from trial_mapping (most reliable)
    if "image_path" in trial_info:
        img_path = Path(trial_info["image_path"])
        if img_path.exists():
            img = Image.open(img_path).convert("RGB")
            # Resize to 512x512 for consistency with generated images
            img = img.resize((512, 512), Image.Resampling.LANCZOS)
            return img
        else:
            raise FileNotFoundError(f"Image path from mapping does not exist: {img_path}")
    
    # Fallback: try NSD stimuli directory structure
    nsd_image_id = trial_info.get("nsd_image_id")
    if nsd_image_id is not None and nsddata_path:
        nsd_path = Path(nsddata_path)
        stimuli_paths = [
            nsd_path / "nsd_stimuli" / f"{nsd_image_id}.png",
            nsd_path / "nsd_stimuli" / f"{nsd_image_id}.jpg",
            nsd_path.parent / "nsd_stimuli" / f"{nsd_image_id}.png",
            nsd_path.parent / "nsd_stimuli" / f"{nsd_image_id}.jpg",
        ]
        for img_path in stimuli_paths:
            if img_path.exists():
                img = Image.open(img_path).convert("RGB")
                img = img.resize((512, 512), Image.Resampling.LANCZOS)
                return img
    
    raise FileNotFoundError(f"Could not find image for trial {trial_key}")


# Global Stable Diffusion pipeline (lazy loading)
_sd_pipeline = None

def get_stable_diffusion_pipeline(device: str = "cuda"):
    """Get or create Stable Diffusion pipeline (singleton)."""
    global _sd_pipeline
    if _sd_pipeline is None:
        try:
            from diffusers import StableDiffusionPipeline
            import torch
            
            print("Loading Stable Diffusion model...")
            # Use Stable Diffusion 1.5 (smaller, faster) or 2.1
            model_id = "runwayml/stable-diffusion-v1-5"  # Good balance of quality/speed
            
            _sd_pipeline = StableDiffusionPipeline.from_pretrained(
                model_id,
                torch_dtype=torch.float16 if device == "cuda" else torch.float32,
                safety_checker=None,  # Disable for faster generation
                requires_safety_checker=False,
            )
            _sd_pipeline = _sd_pipeline.to(device)
            _sd_pipeline.set_progress_bar_config(disable=True)
            print("Stable Diffusion loaded successfully!")
        except Exception as e:
            print(f"Error loading Stable Diffusion: {e}")
            print("Falling back to placeholder image generation")
            _sd_pipeline = None
    return _sd_pipeline

def generate_image_from_caption(caption: str, device: str = "cuda", num_inference_steps: int = 20):
    """
    Generate image from caption using Stable Diffusion.
    
    Args:
        caption: Text caption to generate image from
        device: Device to run on
        num_inference_steps: Number of diffusion steps (lower = faster, higher = better quality)
    
    Returns:
        PIL.Image: Generated image (512x512)
    """
    pipeline = get_stable_diffusion_pipeline(device)
    
    if pipeline is None:
        # Fallback: return placeholder
        print(f"Warning: Stable Diffusion not available. Caption: {caption[:50]}...")
        return Image.new("RGB", (512, 512), color=(128, 128, 128))
    
    try:
        # Generate image
        # Use negative prompt to improve quality
        negative_prompt = "blurry, distorted, low quality, watermark, text"
        
        with torch.no_grad():
            image = pipeline(
                prompt=caption,
                negative_prompt=negative_prompt,
                num_inference_steps=num_inference_steps,
                guidance_scale=7.5,  # Standard guidance scale
                height=512,
                width=512,
            ).images[0]
        
        return image
    except Exception as e:
        print(f"Error generating image from caption '{caption[:50]}...': {e}")
        # Return placeholder on error
        return Image.new("RGB", (512, 512), color=(128, 128, 128))


def evaluate(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Load checkpoint
    print(f"Loading model from {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location=device)
    config = checkpoint["config"]
    
    # Load tokenizer
    tokenizer_name = config["model"].get("tokenizer", "Qwen/Qwen2-VL-2B-Instruct")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    vocab_size = len(tokenizer)
    
    # Create model
    model = BrainCaptionModel(
        n_voxels=config["model"]["n_voxels"],
        n_timepoints=config["model"]["n_timepoints"],
        brain_hidden_dim=config["model"]["brain_hidden_dim"],
        n_mlp_blocks=config["model"]["n_mlp_blocks"],
        n_subjects=len(config["data"]["subjects"]),
        vocab_size=vocab_size,
        embed_dim=config["model"]["embed_dim"],
        n_decoder_layers=config["model"]["n_decoder_layers"],
        n_heads=config["model"]["n_heads"],
        decoder_ff_dim=config["model"].get("decoder_ff_dim"),
        decoder_dropout=config["model"].get("decoder_dropout", 0.1),
        max_seq_length=config["data"]["max_caption_length"],
        n_brain_tokens=config["model"]["n_brain_tokens"],
    ).to(device)
    
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    
    # Load test dataset
    test_dataset = FmriCaptionDataset(
        nsddata_path=config["data"]["nsddata_path"],
        trial_mapping_path=config["data"]["trial_mapping_path"],
        captions_path=config["data"]["captions_path"],
        subjects=config["data"]["subjects"],
        split="test",
        tokenizer_name=tokenizer_name,
        max_caption_length=config["data"]["max_caption_length"],
        train_ratio=config["data"].get("train_ratio", 0.8),
        val_ratio=config["data"].get("val_ratio", 0.1),
        test_ratio=config["data"].get("test_ratio", 0.1),
        seed=config["data"].get("seed", 42),
    )
    
    # Use custom collate function to handle None values
    def collate_fn_eval(batch):
        """Custom collate function that handles None values."""
        from dataPrepartion.fmri_caption_dataset import collate_fn_variable_fmri
        # Use the existing collate function but handle None coco_ids
        result = collate_fn_variable_fmri(batch)
        # Replace None coco_ids with -1 (or keep as list of None)
        if "coco_id" in result:
            result["coco_id"] = [cid if cid is not None else -1 for cid in result["coco_id"]]
        return result
    
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, collate_fn=collate_fn_eval)
    
    # Load trial mapping once (for image loading)
    trial_mapping = None
    if args.generate_images:
        print("Loading trial mapping for image loading...")
        with open(config["data"]["trial_mapping_path"], 'r') as f:
            trial_mapping = json.load(f)
    
    # Generate captions and images
    print("Generating captions and images...")
    print(f"Total test samples: {len(test_dataset)}")
    start_time = time.time()
    predictions = {}
    references = {}
    generated_images = []
    ground_truth_images = []
    coco_ids = []
    
    # Monitoring: save intermediate results every N samples
    save_interval = 50
    checkpoint_dir = Path(args.checkpoint).parent
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(test_loader, desc="Generating")):
            brain = batch["brain"].to(device)
            subject_ids = batch["subject_id"].to(device)
            original_voxel_counts = batch.get("original_voxel_counts")
            if original_voxel_counts is not None:
                original_voxel_counts = original_voxel_counts.to(device)
            trial_key = batch["trial_key"][0]
            gt_caption = batch["caption"][0]
            coco_id = batch.get("coco_id", [None])[0]
            
            # Generate caption
            generated_ids = model.generate(
                brain=brain,
                subject_ids=subject_ids,
                tokenizer=tokenizer,
                max_length=64,
                temperature=0.7,
                top_k=50,
                top_p=0.95,
                original_voxel_counts=original_voxel_counts,
            )
            
            generated_caption = tokenizer.decode(
                generated_ids[0],
                skip_special_tokens=True,
            )
            
            predictions[trial_key] = [generated_caption]
            references[trial_key] = [gt_caption]
            
            # Monitoring: print sample predictions periodically
            if (batch_idx + 1) % 20 == 0:
                elapsed = time.time() - start_time
                rate = (batch_idx + 1) / elapsed if elapsed > 0 else 0
                remaining = (len(test_dataset) - batch_idx - 1) / rate if rate > 0 else 0
                print(f"\n[Progress] {batch_idx + 1}/{len(test_dataset)} samples | "
                      f"Elapsed: {elapsed/60:.1f}min | "
                      f"Rate: {rate:.2f} samples/sec | "
                      f"ETA: {remaining/60:.1f}min")
                print(f"[Sample] Trial: {trial_key}")
                print(f"  Generated: {generated_caption[:80]}...")
                print(f"  Ground truth: {gt_caption[:80]}...")
            
            # Monitoring: save intermediate results periodically
            if (batch_idx + 1) % save_interval == 0:
                intermediate_path = checkpoint_dir / "eval_results_intermediate.json"
                intermediate_data = {
                    "timestamp": datetime.now().isoformat(),
                    "samples_processed": batch_idx + 1,
                    "total_samples": len(test_dataset),
                    "predictions_count": len(predictions),
                    "images_generated": len(generated_images),
                    "sample_predictions": dict(list(predictions.items())[-5:])  # Last 5 samples
                }
                with open(intermediate_path, "w") as f:
                    json.dump(intermediate_data, f, indent=2)
                print(f"[Checkpoint] Saved intermediate results to {intermediate_path}")
            
            # Generate image from caption
            if args.generate_images and trial_mapping is not None:
                gen_image = generate_image_from_caption(
                    generated_caption, 
                    device=device,
                    num_inference_steps=args.num_inference_steps,
                )
                generated_images.append(gen_image)
                
                # Save generated image to disk
                if args.save_images:
                    images_dir = checkpoint_dir / "generated_images"
                    images_dir.mkdir(exist_ok=True)
                    image_path = images_dir / f"{trial_key}_generated.png"
                    gen_image.save(image_path)
                
                # Load ground truth NSD image
                try:
                    gt_image = load_nsd_image(
                        trial_mapping,
                        trial_key,
                        config["data"]["nsddata_path"],
                    )
                    ground_truth_images.append(gt_image)
                    coco_ids.append(coco_id)
                    
                    # Save ground truth image to disk
                    if args.save_images:
                        images_dir = checkpoint_dir / "generated_images"
                        images_dir.mkdir(exist_ok=True)
                        gt_path = images_dir / f"{trial_key}_ground_truth.png"
                        gt_image.save(gt_path)
                except (FileNotFoundError, KeyError, ValueError) as e:
                    print(f"Warning: Could not load ground truth image for {trial_key}: {e}")
                    # Skip this sample for image metrics
                    if len(generated_images) > len(ground_truth_images):
                        generated_images.pop()  # Remove the generated image too
    
    # Compute caption metrics first (NLP evaluation)
    caption_results = {}
    if not METRICS_AVAILABLE:
        print("WARNING: pycocoevalcap not available. Caption metrics will not be computed.")
        print("Install with: pip install pycocoevalcap")
        caption_results = {"error": "pycocoevalcap not installed"}
    elif len(predictions) == 0:
        print("WARNING: No predictions available for caption metrics")
        caption_results = {"error": "No predictions available"}
    else:
        print(f"\nComputing caption metrics for {len(predictions)} samples...")
        try:
            bleu = Bleu()
            meteor = Meteor()
            rouge = Rouge()
            cider = Cider()
            
            # Compute scores
            print("  Computing BLEU...")
            bleu_scores, _ = bleu.compute_score(references, predictions)
            print("  Computing METEOR...")
            meteor_score, _ = meteor.compute_score(references, predictions)
            print("  Computing ROUGE-L...")
            rouge_score, _ = rouge.compute_score(references, predictions)
            print("  Computing CIDEr...")
            cider_score, _ = cider.compute_score(references, predictions)
            
            # Extract BLEU scores (returns list of 4 scores for BLEU-1, BLEU-2, BLEU-3, BLEU-4)
            if isinstance(bleu_scores, (list, tuple)) and len(bleu_scores) >= 4:
                caption_results["BLEU-1"] = float(bleu_scores[0])
                caption_results["BLEU-2"] = float(bleu_scores[1])
                caption_results["BLEU-3"] = float(bleu_scores[2])
                caption_results["BLEU-4"] = float(bleu_scores[3])
            else:
                caption_results["BLEU"] = float(bleu_scores) if not isinstance(bleu_scores, (list, tuple)) else float(bleu_scores[0])
            
            caption_results["METEOR"] = float(meteor_score)
            caption_results["ROUGE-L"] = float(rouge_score)
            caption_results["CIDEr"] = float(cider_score)
            
            # Print caption metrics
            print("\nCaption Metrics:")
            for metric_name, value in caption_results.items():
                print(f"  {metric_name}: {value:.4f}")
                
        except Exception as e:
            print(f"ERROR computing caption metrics: {e}")
            import traceback
            traceback.print_exc()
            caption_results = {"error": f"Caption metrics computation failed: {str(e)}"}
    
    # Compute image metrics
    image_results = {}
    if args.generate_images and IMAGE_METRICS_AVAILABLE and len(generated_images) > 0:
        print(f"Computing image metrics for {len(generated_images)} generated images and {len(ground_truth_images)} ground truth images...")
        try:
            # Compute standard image metrics
            image_results = compute_image_generation_metrics(
                preds=generated_images,
                trues=ground_truth_images,
                device=device,
            )
            
            # Map to expected metric names (exact names as requested)
            results_mapped = {
                "SSIM": image_results.get("ssim", 0.0),
                "PixCorr": image_results.get("pixcorr", 0.0),
                "AlexNet(2)": image_results.get("alexnet-2", 0.0) * 100,  # Convert to percentage
                "AlexNet(5)": image_results.get("alexnet-5", 0.0) * 100,
                "CLIP-12": image_results.get("clip-12", 0.0) * 100,
                "Inception V3": image_results.get("inceptionv3-avgpool", 0.0) * 100,  # Note: space in name
                "EfficientNet": image_results.get("efficientnet-avgpool", 0.0),
                "SWAV": image_results.get("swav-avgpool", 0.0),  # All caps as requested
                "DreamSim": image_results.get("dreamsim", 0.0),
            }
            
            # Compute mIoU if available
            if len(generated_images) > 0 and len(generated_images) == len(ground_truth_images):
                try:
                    print(f"Computing mIoU for {len(generated_images)} image pairs...")
                    miou = compute_miou(
                        generated_images,
                        ground_truth_images,
                        eval_res=512,
                    )
                    results_mapped["mIoU"] = float(miou) if miou is not None else None
                    print(f"mIoU computed: {miou}")
                except Exception as e:
                    print(f"Error computing mIoU: {e}")
                    import traceback
                    traceback.print_exc()
                    results_mapped["mIoU"] = None
            else:
                print(f"Warning: Cannot compute mIoU - generated_images={len(generated_images)}, ground_truth_images={len(ground_truth_images)}")
                results_mapped["mIoU"] = None
            
            image_results = results_mapped
            
            # Print results in the order requested
            metric_order = [
                "SSIM", "PixCorr", "AlexNet(2)", "AlexNet(5)", "CLIP-12",
                "Inception V3", "EfficientNet", "SWAV", "mIoU", "DreamSim"
            ]
            for metric in metric_order:
                if metric in image_results:
                    value = image_results[metric]
                    if value is not None:
                        if metric in ["AlexNet(2)", "AlexNet(5)", "CLIP-12", "Inception V3"]:
                            print(f"{metric}: {value:.2f}%")
                        elif metric == "mIoU":
                            print(f"{metric}: {value:.2f}")
                        else:
                            print(f"{metric}: {value:.4f}")
                        
        except Exception as e:
            print(f"Error computing image metrics: {e}")
            import traceback
            traceback.print_exc()
            image_results = {"error": str(e)}
    else:
        if not args.generate_images:
            print("Warning: --generate-images flag not set. Skipping image generation and metrics.")
        elif not IMAGE_METRICS_AVAILABLE:
            print("Warning: Image metrics not available. Skipping image metrics computation.")
        elif len(generated_images) == 0:
            print("Warning: No generated images available. Skipping image metrics computation.")
        image_results = {"error": "Image metrics not computed - see warnings above"}
    
    # Combine caption and image metrics
    all_results = {
        "caption_metrics": caption_results,
        "image_metrics": image_results
    }
    
    # Save results (both caption and image metrics)
    output_path = Path(args.checkpoint).parent / "eval_results_full.json"
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\nResults saved to {output_path}")
    if caption_results and "error" not in caption_results:
        print(f"Caption metrics computed successfully: {len(caption_results)} metrics")
    if image_results and "error" not in image_results:
        print(f"Image metrics computed successfully: {len(image_results)} metrics")
    
    # Save generated captions for analysis
    captions_path = Path(args.checkpoint).parent / "generated_captions.json"
    with open(captions_path, "w") as f:
        json.dump({
            "predictions": predictions,
            "references": references,
        }, f, indent=2)
    
    print(f"Generated captions saved to {captions_path}")
    
    return all_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate fMRI-to-Caption model with image metrics")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--generate-images", action="store_true", help="Generate images from captions using Stable Diffusion")
    parser.add_argument("--save-images", action="store_true", help="Save generated and ground truth images to disk")
    parser.add_argument("--num-inference-steps", type=int, default=20, help="Number of diffusion steps for image generation (default: 20)")
    args = parser.parse_args()
    evaluate(args)

