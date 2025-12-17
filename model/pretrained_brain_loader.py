#!/usr/bin/env python3
"""
Utility for loading pretrained DynaDiff brain encoder weights.

Provides functions to load and initialize brain encoder with
pretrained weights from DynaDiff checkpoints.
"""

import torch
import torch.nn as nn
from pathlib import Path
from typing import Optional, Dict, Any
import warnings


def extract_brain_module_state(
    checkpoint_path: str,
    device: str = "cpu"
) -> Dict[str, torch.Tensor]:
    """
    Extract brain module state dict from DynaDiff checkpoint.
    
    Args:
        checkpoint_path: Path to DynaDiff checkpoint file
        device: Device to load tensors on
        
    Returns:
        Dictionary containing brain module weights
    """
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"DynaDiff checkpoint not found: {checkpoint_path}")
    
    print(f"Loading DynaDiff checkpoint from: {checkpoint_path}")
    
    try:
        checkpoint_data = torch.load(checkpoint_path, map_location=device)
    except Exception as e:
        raise RuntimeError(f"Failed to load checkpoint: {e}")
    
    state_dict = None
    
    if isinstance(checkpoint_data, dict):
        if "state_dict" in checkpoint_data:
            state_dict = checkpoint_data["state_dict"]
        elif "model_state_dict" in checkpoint_data:
            state_dict = checkpoint_data["model_state_dict"]
        elif "model" in checkpoint_data:
            state_dict = checkpoint_data["model"]
        else:
            state_dict = checkpoint_data
    
    if state_dict is None:
        raise ValueError("Could not find state dict in checkpoint")
    
    return state_dict


def filter_brain_encoder_weights(
    state_dict: Dict[str, torch.Tensor],
    prefix: str = "mlp_modules.clip_image"
) -> Dict[str, torch.Tensor]:
    """
    Filter state dict to extract only brain encoder (FmriMLP) weights.
    
    Args:
        state_dict: Full checkpoint state dict
        prefix: Prefix used in checkpoint for brain module weights
        
    Returns:
        Filtered state dict with only brain encoder weights
    """
    brain_weights = {}
    
    for key, value in state_dict.items():
        if prefix in key:
            new_key = key.replace(f"{prefix}.", "")
            brain_weights[new_key] = value
    
    if len(brain_weights) == 0:
        warnings.warn(
            f"No weights found with prefix '{prefix}'. "
            "Trying alternative prefixes..."
        )
        for alt_prefix in ["brain_module", "fmri_mlp", "encoder"]:
            for key, value in state_dict.items():
                if alt_prefix in key.lower():
                    new_key = key.replace(f"{alt_prefix}.", "").replace(f"{alt_prefix}_", "")
                    brain_weights[new_key] = value
            if len(brain_weights) > 0:
                break
    
    return brain_weights


def load_dynadiff_weights_into_encoder(
    encoder: nn.Module,
    checkpoint_path: str,
    strict: bool = False,
    device: str = "cpu"
) -> nn.Module:
    """
    Load DynaDiff pretrained weights into brain encoder.
    
    Args:
        encoder: Brain encoder module to load weights into
        checkpoint_path: Path to DynaDiff checkpoint
        strict: Whether to strictly match all keys
        device: Device to load weights on
        
    Returns:
        Encoder with loaded weights
    """
    full_state = extract_brain_module_state(checkpoint_path, device)
    brain_weights = filter_brain_encoder_weights(full_state)
    
    if len(brain_weights) == 0:
        warnings.warn(
            "No brain encoder weights found in checkpoint. "
            "Encoder will use random initialization."
        )
        return encoder
    
    encoder_state = encoder.state_dict()
    matched_keys = []
    missing_keys = []
    unexpected_keys = []
    
    for key in encoder_state.keys():
        if key in brain_weights:
            matched_keys.append(key)
            encoder_state[key] = brain_weights[key]
        else:
            missing_keys.append(key)
    
    for key in brain_weights.keys():
        if key not in encoder_state:
            unexpected_keys.append(key)
    
    encoder.load_state_dict(encoder_state, strict=False)
    
    print(f"Loaded {len(matched_keys)} weights from DynaDiff checkpoint")
    if missing_keys and not strict:
        print(f"Warning: {len(missing_keys)} encoder keys not found in checkpoint (using random init)")
    if unexpected_keys:
        print(f"Warning: {len(unexpected_keys)} checkpoint keys not used")
    
    return encoder


def find_dynadiff_checkpoint(
    search_paths: list,
    checkpoint_name: Optional[str] = None
) -> Optional[str]:
    """
    Search for DynaDiff checkpoint in common locations.
    
    Args:
        search_paths: List of directories to search
        checkpoint_name: Optional specific checkpoint filename
        
    Returns:
        Path to checkpoint if found, None otherwise
    """
    if checkpoint_name:
        for search_path in search_paths:
            path = Path(search_path) / checkpoint_name
            if path.exists():
                return str(path)
    
    common_names = [
        "checkpoint.pt",
        "best.pt",
        "dynadiff.pt",
        "brain_encoder.pt",
        "model.pt"
    ]
    
    for search_path in search_paths:
        for name in common_names:
            path = Path(search_path) / name
            if path.exists():
                return str(path)
    
    return None

