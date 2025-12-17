#!/usr/bin/env python3
"""
Generate predictions JSON for test dataset.

Generates caption predictions for test dataset samples and saves to JSON.
Can filter to specific subject if needed.
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

from transformers import AutoTokenizer

# Import dataset and model
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
from dataPrepartion.fmri_caption_dataset import FmriCaptionDataset
from model.brain_caption_model import BrainCaptionModel


def generate_predictions(args):
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
        result = collate_fn_variable_fmri(batch)
        if "coco_id" in result:
            result["coco_id"] = [cid if cid is not None else -1 for cid in result["coco_id"]]
        return result
    
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, collate_fn=collate_fn_eval)
    
    # Generate predictions
    print("Generating predictions...")
    print(f"Total test samples: {len(test_dataset)}")
    if args.subject_filter:
        print(f"Filtering to: {args.subject_filter}")
    start_time = time.time()
    results = []
    
    # Map subject IDs to subject names
    subject_id_to_name = {i: subj for i, subj in enumerate(config["data"]["subjects"])}
    print(f"Available subjects: {config['data']['subjects']}")
    print(f"Subject ID to name mapping: {subject_id_to_name}")
    
    if args.subject_filter:
        # Validate subject filter
        if args.subject_filter not in config["data"]["subjects"]:
            print(f"WARNING: Subject filter '{args.subject_filter}' not found in subjects list!")
            print(f"Available subjects: {config['data']['subjects']}")
            print("Proceeding anyway, but no matches may be found...")
    
    # Monitoring: save intermediate results
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
            subject_id = subject_ids[0].item()
            subject_name = subject_id_to_name.get(subject_id, f"subj{subject_id+1:02d}")
            
            # Filter to subject if requested
            if args.subject_filter and subject_name != args.subject_filter:
                continue
            
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
            
            results.append({
                "trial_key": trial_key,
                "subject_id": subject_id,
                "subject_name": subject_name,
                "generated_caption": generated_caption,
                "ground_truth_caption": gt_caption
            })
            
            # Monitoring: print sample predictions periodically
            if (batch_idx + 1) % 20 == 0:
                elapsed = time.time() - start_time
                rate = len(results) / elapsed if elapsed > 0 else 0
                remaining = (len(test_dataset) - batch_idx - 1) / rate if rate > 0 else 0
                print(f"\n[Progress] {len(results)} predictions generated | "
                      f"Elapsed: {elapsed/60:.1f}min | "
                      f"Rate: {rate:.2f} samples/sec | "
                      f"ETA: {remaining/60:.1f}min")
                if len(results) > 0:
                    last = results[-1]
                    print(f"[Sample] {last['trial_key']}: {last['generated_caption'][:80]}...")
            
            # Monitoring: save intermediate results periodically
            if (batch_idx + 1) % save_interval == 0 and len(results) > 0:
                intermediate_path = checkpoint_dir / "predictions_intermediate.json"
                intermediate_data = {
                    "timestamp": datetime.now().isoformat(),
                    "samples_processed": batch_idx + 1,
                    "predictions_count": len(results),
                    "subject_filter": args.subject_filter,
                    "results": results  # Save all results so far
                }
                with open(intermediate_path, "w") as f:
                    json.dump(intermediate_data, f, indent=2)
                print(f"[Checkpoint] Saved {len(results)} intermediate predictions to {intermediate_path}")
    
    # Save to JSON
    output_path = Path(args.output) if args.output else Path(args.checkpoint).parent / "test_predictions.json"
    if args.subject_filter:
        # Include subject name in filename if filtering
        output_path = Path(args.checkpoint).parent / f"test_predictions_{args.subject_filter}.json"
    
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\nGenerated {len(results)} predictions")
    print(f"Results saved to {output_path}")
    
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate predictions JSON for test dataset")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--output", type=str, default=None, help="Output JSON file path (default: checkpoint_dir/test_predictions.json)")
    parser.add_argument("--subject-filter", type=str, default=None, help="Filter to specific subject (e.g., 'subj01') or None for all subjects")
    args = parser.parse_args()
    
    generate_predictions(args)

