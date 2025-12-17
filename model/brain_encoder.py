#!/usr/bin/env python3
"""
Brain Encoder: Wrapper around FmriMLP for fMRI-to-caption model.

Uses DynaDiff pretrained weights for initialization.
"""

import torch
import torch.nn as nn
from functools import partial
from typing import Optional

import sys
from pathlib import Path
# Add parent directory to path for imports
if str(Path(__file__).parent.parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).parent.parent))

from model.fmri_mlp import FmriMLP, FmriMLPConfig, Mean, SubjectLayers
from model.pretrained_brain_loader import load_dynadiff_weights_into_encoder


class BrainEncoder(nn.Module):
    """
    Brain encoder that maps fMRI signals to brain embeddings.
    
    Input:  [B, num_voxels, num_TRs] e.g., [B, 15724, 6]
    Output: [B, n_brain_tokens, embed_dim] e.g., [B, 257, 768]
    """
    
    def __init__(
        self,
        n_voxels: int = 15724,
        n_timepoints: int = 6,
        hidden_dim: int = 4096,
        n_mlp_blocks: int = 4,
        n_subjects: int = 4,  # subjects 1, 2, 5, 7
        embed_dim: int = 768,
        n_brain_tokens: int = 257,
        dynadiff_checkpoint_path: Optional[str] = None,
    ):
        super().__init__()
        
        self.n_voxels = n_voxels
        self.n_timepoints = n_timepoints
        self.n_brain_tokens = n_brain_tokens
        self.embed_dim = embed_dim
        self.dynadiff_checkpoint_path = dynadiff_checkpoint_path
        
        # Configure FmriMLP
        brain_config = FmriMLPConfig(
            hidden=hidden_dim,
            n_blocks=n_mlp_blocks,
            norm_type="ln",
            act_first=False,
            n_repetition_times=n_timepoints,
            time_agg="out_linear",
            subject_layers=True,
            n_subjects=n_subjects,
            subject_layers_dim="hidden",
            subject_layers_id=False,
            out_dim=n_brain_tokens * embed_dim,
            native_fmri_space=False,
            blurry_recon=False,
            use_tr_embeds=False,
            use_tr_layer=False,
        )
        
        # Build FmriMLP
        self.fmri_mlp = brain_config.build(
            n_in_channels=n_voxels,
            n_outputs=n_brain_tokens * embed_dim,
        )
        
        # Load DynaDiff pretrained weights if checkpoint provided
        if dynadiff_checkpoint_path is not None:
            self._load_pretrained_weights()
        else:
            self._init_weights()
    
    def _load_pretrained_weights(self):
        """Load DynaDiff pretrained weights into the encoder."""
        if self.dynadiff_checkpoint_path is None:
            return
        
        try:
            device = next(self.parameters()).device
            load_dynadiff_weights_into_encoder(
                self.fmri_mlp,
                self.dynadiff_checkpoint_path,
                strict=False,
                device=str(device)
            )
            print("Successfully loaded DynaDiff pretrained weights")
        except Exception as e:
            print(f"Warning: Failed to load DynaDiff weights: {e}")
            print("Falling back to random initialization")
            self._init_weights()
    
    def _init_weights(self):
        """Initialize weights for modules not loaded from checkpoint."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
            elif isinstance(module, SubjectLayers):
                pass
    
    def forward(self, brain: torch.Tensor, subject_ids: torch.Tensor, original_voxel_counts: torch.Tensor = None) -> torch.Tensor:
        """
        Encode fMRI signals to brain embeddings.
        
        Args:
            brain: [B, num_voxels, num_TRs] e.g., [B, 15724, 6]
            subject_ids: [B] subject indices (0-3 for subjects 1,2,5,7)
            original_voxel_counts: [B] original voxel counts before padding (optional)
        
        Returns:
            brain_embedding: [B, n_brain_tokens, embed_dim] e.g., [B, 257, 768]
        """
        B = brain.shape[0]
        
        # FmriMLP expects [B, C, T] format
        # brain is [B, V, T] where V=voxels, T=timepoints
        brain_input = brain  # Already in correct format
        
        # Forward through FmriMLP
        # FmriMLP returns dict with "MSELoss" key containing the output
        output = self.fmri_mlp(brain_input, subject_ids, original_voxel_counts=original_voxel_counts)
        
        if isinstance(output, dict):
            brain_flat = output["MSELoss"]  # [B, n_brain_tokens * embed_dim]
        else:
            brain_flat = output  # [B, n_brain_tokens * embed_dim]
        
        # Reshape to [B, n_brain_tokens, embed_dim]
        brain_embedding = brain_flat.view(B, self.n_brain_tokens, self.embed_dim)
        
        return brain_embedding

