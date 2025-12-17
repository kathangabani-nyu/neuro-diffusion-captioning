#!/usr/bin/env python3
"""
BrainCaptionModel: End-to-end fMRI → Caption model.

Combines brain encoder (FmriMLP) and caption decoder (Transformer)
to generate captions directly from fMRI signals.
"""

import torch
import torch.nn as nn

import sys
from pathlib import Path
# Add parent directory to path for imports
if str(Path(__file__).parent.parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).parent.parent))

from model.brain_encoder import BrainEncoder
from model.transformer_captioner import CaptionDecoder


class BrainCaptionModel(nn.Module):
    """
    Full model: fMRI → Brain Encoder → Caption Decoder → Caption
    
    Uses DynaDiff pretrained brain encoder (346M parameters) for strong
    initialization. Caption decoder is trained from scratch.
    """
    
    def __init__(
        self,
        # Brain encoder config
        n_voxels: int = 15724,
        n_timepoints: int = 6,
        brain_hidden_dim: int = 4096,
        n_mlp_blocks: int = 4,
        n_subjects: int = 4,  # subjects 1, 2, 5, 7
        
        # Caption decoder config
        vocab_size: int = 151936,  # Qwen2-VL vocab size
        embed_dim: int = 768,
        n_decoder_layers: int = 6,
        n_heads: int = 12,
        decoder_ff_dim: int = None,
        decoder_dropout: float = 0.1,
        max_seq_length: int = 512,
        
        # Brain embedding config
        n_brain_tokens: int = 257,  # Following DynaDiff's CLIP alignment
        
        # Training optimization
        use_gradient_checkpointing: bool = False,
        
        # DynaDiff checkpoint
        dynadiff_checkpoint_path: str = None,
    ):
        super().__init__()
        
        self.n_brain_tokens = n_brain_tokens
        self.embed_dim = embed_dim
        self.use_gradient_checkpointing = use_gradient_checkpointing
        
        # Brain encoder with DynaDiff pretrained weights
        self.brain_encoder = BrainEncoder(
            n_voxels=n_voxels,
            n_timepoints=n_timepoints,
            hidden_dim=brain_hidden_dim,
            n_mlp_blocks=n_mlp_blocks,
            n_subjects=n_subjects,
            embed_dim=embed_dim,
            n_brain_tokens=n_brain_tokens,
            dynadiff_checkpoint_path=dynadiff_checkpoint_path,
        )
        
        # Caption decoder
        self.caption_decoder = CaptionDecoder(
            vocab_size=vocab_size,
            embed_dim=embed_dim,
            n_layers=n_decoder_layers,
            n_heads=n_heads,
            ff_dim=decoder_ff_dim,
            dropout=decoder_dropout,
            max_seq_length=max_seq_length,
            use_gradient_checkpointing=use_gradient_checkpointing,
        )
    
    def forward(
        self,
        brain: torch.Tensor,  # [B, 15724, 6]
        subject_ids: torch.Tensor,  # [B]
        caption_ids: torch.Tensor,  # [B, L]
        attention_mask: torch.Tensor = None,  # [B, L]
        labels: torch.Tensor = None,  # [B, L] for loss computation
        original_voxel_counts: torch.Tensor = None,  # [B] original voxel counts before padding
    ):
        """
        Forward pass with teacher forcing.
        
        Returns:
            BrainCaptionOutput with loss and logits
        """
        # 1. Encode fMRI → brain embedding
        brain_embedding = self.brain_encoder(brain, subject_ids, original_voxel_counts)  # [B, 257, 768]
        
        # 2. Decode brain embedding → caption logits
        logits = self.caption_decoder(
            caption_ids=caption_ids,
            brain_embedding=brain_embedding,
            attention_mask=attention_mask,
        )  # [B, L, vocab_size]
        
        # 3. Compute loss if labels provided
        loss = None
        if labels is not None:
            # Shift for next-token prediction
            shift_logits = logits[:, :-1, :].contiguous()  # [B, L-1, vocab_size]
            shift_labels = labels[:, 1:].contiguous()  # [B, L-1]
            
            loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
            loss = loss_fct(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
            )
        
        return BrainCaptionOutput(
            loss=loss,
            logits=logits,
            brain_embedding=brain_embedding,
        )
    
    @torch.no_grad()
    def generate(
        self,
        brain: torch.Tensor,  # [B, 15724, 6]
        subject_ids: torch.Tensor,  # [B]
        tokenizer,
        max_length: int = 64,
        temperature: float = 1.0,
        top_k: int = 50,
        top_p: float = 0.95,
        original_voxel_counts: torch.Tensor = None,  # [B] original voxel counts before padding
    ) -> torch.Tensor:
        """
        Generate captions from fMRI signals.
        
        Returns:
            generated_ids: [B, generated_length]
        """
        # Encode brain
        brain_embedding = self.brain_encoder(brain, subject_ids, original_voxel_counts)  # [B, 257, 768]
        
        # Generate caption
        generated_ids = self.caption_decoder.generate(
            brain_embedding=brain_embedding,
            tokenizer=tokenizer,
            max_length=max_length,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
        )
        
        return generated_ids


class BrainCaptionOutput:
    """Container for model outputs."""
    def __init__(self, loss=None, logits=None, brain_embedding=None):
        self.loss = loss
        self.logits = logits
        self.brain_embedding = brain_embedding

