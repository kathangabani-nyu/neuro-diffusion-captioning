#!/usr/bin/env python3
"""
Caption Decoder: Transformer decoder for generating captions from brain embeddings.

Uses cross-attention to condition on brain embeddings and causal masking
for autoregressive generation.
"""

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint


class CaptionDecoder(nn.Module):
    """
    Transformer decoder for caption generation.
    
    Architecture:
    - Token embedding + positional embedding
    - 6-layer TransformerDecoder with cross-attention to brain embeddings
    - LM head for vocabulary prediction
    """
    
    def __init__(
        self,
        vocab_size: int,
        embed_dim: int = 768,
        n_layers: int = 6,
        n_heads: int = 12,
        ff_dim: int = None,
        dropout: float = 0.1,
        max_seq_length: int = 512,
        use_gradient_checkpointing: bool = False,
    ):
        super().__init__()
        
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.n_layers = n_layers
        self.max_seq_length = max_seq_length
        self.use_gradient_checkpointing = use_gradient_checkpointing
        
        if ff_dim is None:
            ff_dim = embed_dim * 4
        
        # Token and positional embeddings
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.pos_embedding = nn.Embedding(max_seq_length, embed_dim)
        
        # Transformer decoder layers
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=embed_dim,
            nhead=n_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=n_layers)
        
        # Language model head
        self.lm_head = nn.Linear(embed_dim, vocab_size, bias=False)
        
        # Optional: tie weights (share embedding and output weights)
        # self.lm_head.weight = self.token_embedding.weight
        
        self.dropout = nn.Dropout(dropout)
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights."""
        nn.init.normal_(self.token_embedding.weight, std=0.02)
        nn.init.normal_(self.pos_embedding.weight, std=0.02)
        nn.init.normal_(self.lm_head.weight, std=0.02)
    
    def forward(
        self,
        caption_ids: torch.Tensor,  # [B, L]
        brain_embedding: torch.Tensor,  # [B, n_brain_tokens, embed_dim]
        attention_mask: torch.Tensor = None,  # [B, L]
    ) -> torch.Tensor:
        """
        Forward pass with teacher forcing.
        
        Args:
            caption_ids: Tokenized caption [B, L]
            brain_embedding: Brain embedding [B, n_brain_tokens, embed_dim]
            attention_mask: Attention mask for padding [B, L]
        
        Returns:
            logits: [B, L, vocab_size]
        """
        B, L = caption_ids.shape
        device = caption_ids.device
        
        # Embed tokens
        token_emb = self.token_embedding(caption_ids)  # [B, L, embed_dim]
        
        # Add positional embeddings
        positions = torch.arange(L, device=device).unsqueeze(0).expand(B, -1)
        pos_emb = self.pos_embedding(positions)  # [B, L, embed_dim]
        
        # Combine embeddings
        caption_embedding = token_emb + pos_emb
        caption_embedding = self.dropout(caption_embedding)
        
        # Create causal mask for autoregressive decoding
        causal_mask = nn.Transformer.generate_square_subsequent_mask(L, device=device)
        
        # Transformer decoder
        # tgt = caption embedding (what we're generating)
        # memory = brain embedding (what we're conditioning on)
        if self.use_gradient_checkpointing and self.training:
            # Use gradient checkpointing to save memory (trades compute for memory)
            # Create a function that calls the decoder with proper arguments
            def decoder_forward(tgt, memory, tgt_mask):
                return self.decoder(
                    tgt=tgt,
                    memory=memory,
                    tgt_mask=tgt_mask,
                    tgt_is_causal=True,
                )
            decoder_output = checkpoint(
                decoder_forward,
                caption_embedding,
                brain_embedding,
                causal_mask,
                use_reentrant=False,
            )
        else:
            decoder_output = self.decoder(
                tgt=caption_embedding,
                memory=brain_embedding,
                tgt_mask=causal_mask,
                tgt_is_causal=True,
            )  # [B, L, embed_dim]
        
        # Project to vocabulary
        logits = self.lm_head(decoder_output)  # [B, L, vocab_size]
        
        return logits
    
    @torch.no_grad()
    def generate(
        self,
        brain_embedding: torch.Tensor,  # [B, n_brain_tokens, embed_dim]
        tokenizer,
        max_length: int = 64,
        temperature: float = 1.0,
        top_k: int = 50,
        top_p: float = 0.95,
        bos_token_id: int = None,
        eos_token_id: int = None,
    ) -> torch.Tensor:
        """
        Autoregressive caption generation.
        
        Returns:
            generated_ids: [B, generated_length]
        """
        B = brain_embedding.shape[0]
        device = brain_embedding.device
        
        if bos_token_id is None:
            bos_token_id = getattr(tokenizer, 'bos_token_id', None)
            if bos_token_id is None:
                # Try pad_token_id or use 1 as fallback
                bos_token_id = getattr(tokenizer, 'pad_token_id', 1)
        if eos_token_id is None:
            eos_token_id = getattr(tokenizer, 'eos_token_id', None)
            if eos_token_id is None:
                # Use 2 as fallback
                eos_token_id = 2
        
        # Start with BOS token
        generated = torch.full((B, 1), bos_token_id, dtype=torch.long, device=device)
        
        for step in range(max_length - 1):
            # Get embeddings for current sequence
            L = generated.size(1)
            token_emb = self.token_embedding(generated)  # [B, L, embed_dim]
            positions = torch.arange(L, device=device).unsqueeze(0).expand(B, -1)
            pos_emb = self.pos_embedding(positions)
            caption_emb = token_emb + pos_emb
            
            # Decode
            causal_mask = nn.Transformer.generate_square_subsequent_mask(L, device=device)
            decoder_out = self.decoder(
                caption_emb, brain_embedding, tgt_mask=causal_mask, tgt_is_causal=True
            )
            
            # Get next token logits (last position only)
            next_token_logits = self.lm_head(decoder_out[:, -1, :]) / temperature  # [B, vocab_size]
            
            # Top-k filtering
            if top_k > 0:
                indices_to_remove = next_token_logits < torch.topk(next_token_logits, top_k)[0][..., -1, None]
                next_token_logits[indices_to_remove] = float('-inf')
            
            # Top-p (nucleus) sampling
            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                next_token_logits[indices_to_remove] = float('-inf')
            
            # Sample next token
            probs = torch.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)  # [B, 1]
            
            # Append to generated sequence
            generated = torch.cat([generated, next_token], dim=1)
            
            # Stop if all sequences have EOS
            if (next_token == eos_token_id).all():
                break
        
        return generated

