#!/usr/bin/env python3
"""
Main training script for fMRI → Caption model.

Usage:
    python train/train_caption_model.py --config config/train_config.yaml
"""

import argparse
import json
import math
import os
import subprocess
import threading
import time
from pathlib import Path
from datetime import datetime

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from dataPrepartion.fmri_caption_dataset import collate_fn_variable_fmri
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.cuda.amp import GradScaler, autocast
import yaml

from transformers import AutoTokenizer
from tqdm import tqdm

# Import dataset and model
import sys
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
from dataPrepartion.fmri_caption_dataset import FmriCaptionDataset
from model.brain_caption_model import BrainCaptionModel


def parse_args():
    parser = argparse.ArgumentParser(description="Train fMRI-to-Caption model")
    parser.add_argument("--config", type=str, required=True, help="Path to config YAML file")
    parser.add_argument("--resume", type=str, default=None, help="Resume from checkpoint")
    parser.add_argument("--debug", action="store_true", help="Debug mode with small data")
    return parser.parse_args()


def get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps):
    """Cosine learning rate schedule with linear warmup."""
    def lr_lambda(step):
        if step < warmup_steps:
            return float(step) / float(max(1, warmup_steps))
        progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return 0.5 * (1.0 + math.cos(math.pi * progress))
    return LambdaLR(optimizer, lr_lambda)


def monitor_gpu_utilization(interval=60, log_file=None):
    """Monitor GPU utilization in a background thread."""
    def _monitor():
        while True:
            try:
                result = subprocess.run(
                    ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,memory.total", "--format=csv,noheader,nounits"],
                    capture_output=True,
                    text=True,
                    timeout=5
                )
                if result.returncode == 0:
                    gpu_info = result.stdout.strip().split(', ')
                    if len(gpu_info) >= 3:
                        gpu_util = gpu_info[0]
                        mem_used = gpu_info[1]
                        mem_total = gpu_info[2]
                        msg = f"GPU Util: {gpu_util}%, Memory: {mem_used}/{mem_total} MB"
                        print(f"[GPU Monitor] {msg}")
                        if log_file:
                            with open(log_file, 'a') as f:
                                f.write(f"{datetime.now().isoformat()} - {msg}\n")
            except Exception as e:
                pass  # Silently fail
            time.sleep(interval)
    
    if torch.cuda.is_available():
        thread = threading.Thread(target=_monitor, daemon=True)
        thread.start()
        return thread
    return None


def train():
    args = parse_args()
    
    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Create output directory
    run_name = f"fmri_caption_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    save_dir = Path(config["training"]["save_dir"]) / run_name
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Save config
    with open(save_dir / "config.yaml", 'w') as f:
        yaml.dump(config, f)
    
    # Load tokenizer
    tokenizer_name = config["model"].get("tokenizer", "Qwen/Qwen2-VL-2B-Instruct")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    vocab_size = len(tokenizer)
    print(f"Tokenizer vocab size: {vocab_size}")
    
    # Create datasets
    print("Loading datasets...")
    train_dataset = FmriCaptionDataset(
        nsddata_path=config["data"]["nsddata_path"],
        trial_mapping_path=config["data"]["trial_mapping_path"],
        captions_path=config["data"]["captions_path"],
        subjects=config["data"]["subjects"],
        split="train",
        tokenizer_name=tokenizer_name,
        max_caption_length=config["data"]["max_caption_length"],
        train_ratio=config["data"].get("train_ratio", 0.8),
        val_ratio=config["data"].get("val_ratio", 0.1),
        test_ratio=config["data"].get("test_ratio", 0.1),
        seed=config["data"].get("seed", 42),
    )
    
    val_dataset = FmriCaptionDataset(
        nsddata_path=config["data"]["nsddata_path"],
        trial_mapping_path=config["data"]["trial_mapping_path"],
        captions_path=config["data"]["captions_path"],
        subjects=config["data"]["subjects"],
        split="val",
        tokenizer_name=tokenizer_name,
        max_caption_length=config["data"]["max_caption_length"],
        train_ratio=config["data"].get("train_ratio", 0.8),
        val_ratio=config["data"].get("val_ratio", 0.1),
        test_ratio=config["data"].get("test_ratio", 0.1),
        seed=config["data"].get("seed", 42),
    )
    
    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}")
    
    # Create data loaders
    batch_size = int(config["training"]["batch_size"])
    # Use multiple workers for parallel data loading to keep GPU busy
    # We fixed tensor storage issues, so multiprocessing should work now
    num_workers = int(config["training"].get("num_workers", 4))
    # Use persistent workers to avoid worker startup overhead
    persistent_workers = num_workers > 0
    # Pin memory for faster CPU→GPU transfers
    pin_memory = torch.cuda.is_available()
    # Prefetch multiple batches to keep GPU fed (read from config if available)
    prefetch_factor = config["training"].get("prefetch_factor", 2) if num_workers > 0 else None
    
    print(f"DataLoader config: batch_size={batch_size}, num_workers={num_workers}, "
          f"pin_memory={pin_memory}, persistent_workers={persistent_workers}, "
          f"prefetch_factor={prefetch_factor}")
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
        prefetch_factor=prefetch_factor,
        collate_fn=collate_fn_variable_fmri,  # Handle variable-length fMRI sequences
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
        prefetch_factor=prefetch_factor,
        collate_fn=collate_fn_variable_fmri,  # Handle variable-length fMRI sequences
    )
    
    # Create model with DynaDiff pretrained brain encoder
    use_gradient_checkpointing = config["training"].get("use_gradient_checkpointing", False)
    dynadiff_checkpoint = config["model"].get("dynadiff_checkpoint_path", None)
    
    if dynadiff_checkpoint:
        print(f"Building model with DynaDiff pretrained brain encoder from: {dynadiff_checkpoint}")
    else:
        print("Building model with DynaDiff pretrained brain encoder (checkpoint path not specified)")
    
    if use_gradient_checkpointing:
        print("Gradient checkpointing enabled - will save memory at cost of compute")
    
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
        use_gradient_checkpointing=use_gradient_checkpointing,
        dynadiff_checkpoint_path=dynadiff_checkpoint,
    ).to(device)
    
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {n_params / 1e6:.2f}M")
    
    # Optimizer and scheduler
    # Ensure learning_rate is a float (YAML might read 1e-4 as string)
    learning_rate = float(config["training"]["learning_rate"])
    weight_decay = float(config["training"].get("weight_decay", 0.01))
    
    optimizer = AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
        betas=(0.9, 0.999),
    )
    
    epochs = int(config["training"]["epochs"])
    gradient_accumulation_steps = int(config["training"].get("gradient_accumulation_steps", 1))
    # With gradient accumulation, we update weights less frequently
    batches_per_epoch = len(train_loader) // gradient_accumulation_steps
    total_steps = batches_per_epoch * epochs
    warmup_steps = min(1000, total_steps // 10)
    
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)
    scaler = GradScaler(enabled=config["training"].get("mixed_precision", True))
    
    print(f"Total training steps: {total_steps} (with gradient accumulation: {gradient_accumulation_steps})")
    print(f"Warmup steps: {warmup_steps}")
    
    # Resume from checkpoint
    start_epoch = 0
    global_step = 0
    best_val_loss = float("inf")
    
    if args.resume:
        print(f"Resuming from {args.resume}")
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint["model_state"])
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        scheduler.load_state_dict(checkpoint["scheduler_state"])
        start_epoch = checkpoint["epoch"]
        global_step = checkpoint["global_step"]
        best_val_loss = checkpoint.get("val_loss", float("inf"))
    
    # Training loop
    print("\n" + "="*60)
    print("Starting training - GPU should be actively utilized!")
    print("Monitor GPU usage with: nvidia-smi -l 1")
    print("="*60 + "\n")
    
    # Start GPU monitoring in background
    gpu_log_file = save_dir / "gpu_utilization.log"
    monitor_thread = monitor_gpu_utilization(interval=60, log_file=str(gpu_log_file))
    if monitor_thread:
        print(f"GPU monitoring started (logging to {gpu_log_file})")
    
    # Gradient accumulation
    gradient_accumulation_steps = int(config["training"].get("gradient_accumulation_steps", 1))
    effective_batch_size = batch_size * gradient_accumulation_steps
    print(f"Gradient accumulation: {gradient_accumulation_steps} steps")
    print(f"Effective batch size: {effective_batch_size} (batch_size={batch_size} × accumulation={gradient_accumulation_steps})")
    
    for epoch in range(start_epoch, epochs):
        model.train()
        epoch_loss = 0.0
        
        progress = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
        
        optimizer.zero_grad()  # Zero gradients at the start of epoch
        
        for batch_idx, batch in enumerate(progress):
            # Move tensors to GPU asynchronously for better performance
            brain = batch["brain"].to(device, non_blocking=pin_memory)
            subject_ids = batch["subject_id"].to(device, non_blocking=pin_memory)
            original_voxel_counts = batch.get("original_voxel_counts")
            if original_voxel_counts is not None:
                original_voxel_counts = original_voxel_counts.to(device, non_blocking=pin_memory)
            input_ids = batch["input_ids"].to(device, non_blocking=pin_memory)
            attention_mask = batch["attention_mask"].to(device, non_blocking=pin_memory)
            labels = batch["labels"].to(device, non_blocking=pin_memory)
            
            with autocast(enabled=config["training"].get("mixed_precision", True)):
                outputs = model(
                    brain=brain,
                    subject_ids=subject_ids,
                    caption_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                    original_voxel_counts=original_voxel_counts,
                )
                # Scale loss by accumulation steps to get average gradient
                loss = outputs.loss / gradient_accumulation_steps
            
            scaler.scale(loss).backward()
            
            epoch_loss += loss.item() * gradient_accumulation_steps  # Unscale for logging
            
            # Only update weights every gradient_accumulation_steps
            if (batch_idx + 1) % gradient_accumulation_steps == 0:
                # Gradient clipping
                scaler.unscale_(optimizer)
                max_grad_norm = float(config["training"].get("max_grad_norm", 1.0))
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    max_grad_norm
                )
                
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()  # Zero gradients after update
                scheduler.step()
                global_step += 1
                
                # Clear cache after each gradient update to reduce memory fragmentation
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            
            # Log GPU memory usage periodically
            if batch_idx % 10 == 0 and torch.cuda.is_available():
                mem_allocated = torch.cuda.memory_allocated(device) / 1024**3  # GB
                mem_reserved = torch.cuda.memory_reserved(device) / 1024**3  # GB
                mem_total = torch.cuda.get_device_properties(device).total_memory / 1024**3  # GB
                progress.set_postfix({
                    "loss": f"{loss.item() * gradient_accumulation_steps:.4f}",
                    "lr": f"{scheduler.get_last_lr()[0]:.2e}",
                    "gpu_mem": f"{mem_allocated:.1f}/{mem_total:.1f}GB",
                })
            else:
                progress.set_postfix({
                    "loss": f"{loss.item() * gradient_accumulation_steps:.4f}",
                    "lr": f"{scheduler.get_last_lr()[0]:.2e}",
                })
        
        avg_train_loss = epoch_loss / len(train_loader)
        print(f"Epoch {epoch+1} - Train Loss: {avg_train_loss:.4f}")
        
        # Validation
        model.eval()
        val_loss = 0.0
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validation"):
                # Move tensors to GPU asynchronously for better performance
                brain = batch["brain"].to(device, non_blocking=pin_memory)
                subject_ids = batch["subject_id"].to(device, non_blocking=pin_memory)
                original_voxel_counts = batch.get("original_voxel_counts")
                if original_voxel_counts is not None:
                    original_voxel_counts = original_voxel_counts.to(device, non_blocking=pin_memory)
                input_ids = batch["input_ids"].to(device, non_blocking=pin_memory)
                attention_mask = batch["attention_mask"].to(device, non_blocking=pin_memory)
                labels = batch["labels"].to(device, non_blocking=pin_memory)
                
                outputs = model(
                    brain=brain,
                    subject_ids=subject_ids,
                    original_voxel_counts=original_voxel_counts,
                    caption_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                )
                val_loss += outputs.loss.item()
        
        avg_val_loss = val_loss / len(val_loader)
        print(f"Epoch {epoch+1} - Val Loss: {avg_val_loss:.4f}")
        
        # Save checkpoint
        checkpoint = {
            "epoch": epoch + 1,
            "global_step": global_step,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "train_loss": avg_train_loss,
            "val_loss": avg_val_loss,
            "config": config,
        }
        
        # Save latest
        torch.save(checkpoint, save_dir / "latest.pt")
        
        # Save best
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(checkpoint, save_dir / "best.pt")
            print(f"New best model saved! Val Loss: {avg_val_loss:.4f}")
        
        # Save periodic (every N epochs)
        save_every = int(config["training"].get("save_every", 10))
        if (epoch + 1) % save_every == 0:
            torch.save(checkpoint, save_dir / f"epoch_{epoch+1}.pt")
            print(f"Saved checkpoint: epoch_{epoch+1}.pt")
        
        # Also save every epoch for safety (can be disabled if disk space is limited)
        # This ensures we don't lose progress if job is cancelled
        torch.save(checkpoint, save_dir / f"epoch_{epoch+1}_latest.pt")
    
    print("Training complete!")
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"Checkpoints saved to: {save_dir}")


if __name__ == "__main__":
    train()

