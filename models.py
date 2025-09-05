import torch
import torch.nn as nn
import math
from copy import deepcopy

# ========== Small helper: Transformer encoder layer with cross-attention ==========
class HarmonyEncoderLayerWithCross(nn.Module):
    """
    One layer for harmony encoder:
      - self-attn (harmony -> harmony)
      - cross-attn (harmony queries -> melody keys/values)
      - feed-forward
    Stores last attention weights for diagnostics.
    """
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation='gelu', batch_first=True):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=batch_first)
        self.cross_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=batch_first)

        # feedforward
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.activation = nn.GELU() if activation == 'gelu' else nn.ReLU()
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        # placeholders for attention visualization
        self.last_self_attn = None  # shape (B, nhead, Lh, Lh) if requested
        self.last_cross_attn = None # shape (B, nhead, Lh, Lm) if requested

    def forward(self, x_h, melody_kv, attn_mask=None, key_padding_mask=None, melody_key_padding_mask=None):
        """
        x_h: (B, Lh, d_model) harmony input
        melody_kv: (B, Lm, d_model) melody encoded (keys & values)
        attn_mask: optional for self-attn
        key_padding_mask: optional for self-attn
        melody_key_padding_mask: optional for cross-attn (for melody padding)
        """
        # Self-attention
        h2, self_w = self.self_attn(x_h, x_h, x_h,
                                    attn_mask=attn_mask,
                                    key_padding_mask=key_padding_mask,
                                    need_weights=True,
                                    average_attn_weights=False)
        # self_w : (B, nhead, Lh, Lh)  if batch_first and average_attn_weights=False
        self.last_self_attn = self_w.detach() if isinstance(self_w, torch.Tensor) else None

        x_h = x_h + self.dropout1(h2)
        x_h = self.norm1(x_h)

        # Cross-attention: queries = harmony (x_h), keys/values = melody_kv
        c2, cross_w = self.cross_attn(x_h, melody_kv, melody_kv,
                                      key_padding_mask=melody_key_padding_mask,
                                      need_weights=True,
                                      average_attn_weights=False)
        self.last_cross_attn = cross_w.detach() if isinstance(cross_w, torch.Tensor) else None

        x_h = x_h + self.dropout2(c2)
        x_h = self.norm2(x_h)

        # Feed-forward
        ff = self.linear2(self.dropout3(self.activation(self.linear1(x_h))))
        x_h = x_h + self.dropout3(ff)
        x_h = self.norm3(x_h)

        return x_h

# ========== Stacked encoder modules ==========
class SimpleTransformerStack(nn.Module):
    """A small wrapper that stacks standard nn.TransformerEncoderLayer layers (no cross-attn)."""
    def __init__(self, layer, num_layers):
        super().__init__()
        self.layers = nn.ModuleList([deepcopy(layer) for _ in range(num_layers)])

    def forward(self, x, src_key_padding_mask=None):
        # x: (B, L, D) assumed batch_first=True
        for layer in self.layers:
            x = layer(x, src_key_padding_mask=src_key_padding_mask)
        return x

class HarmonyTransformerStack(nn.Module):
    """Stack of HarmonyEncoderLayerWithCross"""
    def __init__(self, layer, num_layers):
        super().__init__()
        self.layers = nn.ModuleList([deepcopy(layer) for _ in range(num_layers)])

    def forward(self, x_h, melody_kv, h_key_padding_mask=None, melody_key_padding_mask=None):
        # x_h: (B, Lh, D)
        for layer in self.layers:
            x_h = layer(x_h, melody_kv,
                        key_padding_mask=h_key_padding_mask,
                        melody_key_padding_mask=melody_key_padding_mask)
        return x_h

# ========== Dual-encoder model ==========
class DualGridMLMMelHarm(nn.Module):
    def __init__(self,
                 chord_vocab_size,
                 d_model=512,
                 nhead=8,
                 num_layers_mel=4,
                 num_layers_harm=6,
                 dim_feedforward=2048,
                 pianoroll_dim=12,      # e.g., PCP only
                 melody_length=80,
                 harmony_length=80,
                 max_stages=10,
                 dropout=0.1,
                 device='cpu'):
        super().__init__()
        self.device = device
        self.to(device)

        self.d_model = d_model
        self.melody_length = melody_length
        self.harmony_length = harmony_length

        # Projections
        self.melody_proj = nn.Linear(pianoroll_dim, d_model)   # project PCP (and bar flag) -> d_model
        self.harmony_embedding = nn.Embedding(chord_vocab_size, d_model)

        # Positional embeddings (separate for clarity)
        self.mel_pos = nn.Parameter(torch.randn(1, melody_length, d_model))
        self.harm_pos = nn.Parameter(torch.randn(1, harmony_length, d_model))

        # Stage embedding (for harmony encoder)
        self.max_stages = max_stages
        self.stage_embedding_dim = 64
        self.stage_embedding = nn.Embedding(self.max_stages, self.stage_embedding_dim)
        self.stage_proj = nn.Linear(d_model + self.stage_embedding_dim, d_model)

        # Melody encoder: use standard transformer encoder layers
        encoder_layer_mel = nn.TransformerEncoderLayer(d_model=d_model,
                                                       nhead=nhead,
                                                       dim_feedforward=dim_feedforward,
                                                       dropout=dropout,
                                                       activation='gelu',
                                                       batch_first=True)
        self.melody_encoder = SimpleTransformerStack(encoder_layer_mel, num_layers_mel)

        # Harmony encoder: layers with cross-attention into melody encoded output
        harm_layer = HarmonyEncoderLayerWithCross(d_model=d_model, nhead=nhead,
                                                  dim_feedforward=dim_feedforward,
                                                  dropout=dropout, activation='gelu', batch_first=True)
        self.harmony_encoder = HarmonyTransformerStack(harm_layer, num_layers_harm)

        # Output head for chords
        self.output_head = nn.Linear(d_model, chord_vocab_size)

        # Norms / dropout
        self.input_norm = nn.LayerNorm(d_model)
        self.output_norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, melody_grid, harmony_tokens=None, stage_indices=None,
                melody_key_padding_mask=None, harm_key_padding_mask=None):
        """
        melody_grid: (B, Lm, pianoroll_dim)  -> melody features (PCP + bar flag etc.)
        harmony_tokens: (B, Lh) token ids, or None (then zeros used)
        stage_indices: (B,) or (B,1) ints in [0, max_stages-1] - used by harmony encoder
        Returns:
            harmony_logits: (B, Lh, V)
        """
        B = melody_grid.size(0)
        device = melody_grid.device

        # ---- Melody encoding ----
        mel = self.melody_proj(melody_grid)                    # (B, Lm, d_model)
        mel = mel + self.mel_pos[:, :self.melody_length, :].to(device)
        mel = self.input_norm(mel)
        mel = self.dropout(mel)

        mel_encoded = self.melody_encoder(mel, src_key_padding_mask=melody_key_padding_mask)  # (B, Lm, d_model)

        # ---- Harmony embedding + optional stage conditioning ----
        if harmony_tokens is not None:
            harm = self.harmony_embedding(harmony_tokens)      # (B, Lh, d_model)
        else:
            harm = torch.zeros(B, self.harmony_length, self.d_model, device=device)

        # add harmony positional encodings
        harm = harm + self.harm_pos[:, :self.harmony_length, :].to(device)

        # stage conditioning (concatenate stage embedding to features before projecting back)
        if stage_indices is not None:
            if stage_indices.dim() == 1:
                stage_indices = stage_indices.unsqueeze(-1)
            stage_idx = stage_indices.squeeze(-1).long().to(device)   # (B,)
            stage_emb = self.stage_embedding(stage_idx)               # (B, stage_emb_dim)
            # expand along sequence and concat
            stage_emb_seq = stage_emb.unsqueeze(1).repeat(1, self.harmony_length, 1)  # (B, Lh, S)
            harm = torch.cat([harm, stage_emb_seq], dim=-1)           # (B, Lh, d_model+S)
            harm = self.stage_proj(harm)                              # back to (B, Lh, d_model)

        harm = self.input_norm(harm)
        harm = self.dropout(harm)

        # ---- Harmony encoder: self + cross-attn into melody ----
        harm_encoded = self.harmony_encoder(harm, mel_encoded,
                                            h_key_padding_mask=harm_key_padding_mask,
                                            melody_key_padding_mask=melody_key_padding_mask)  # (B, Lh, d_model)

        harm_encoded = self.output_norm(harm_encoded)

        # ---- Output logits (only harmony positions) ----
        harmony_logits = self.output_head(harm_encoded)  # (B, Lh, V)

        return harmony_logits

    # optionally add helpers to extract attention maps across layers:
    def get_attention_maps(self):
        """
        Returns lists of per-layer attention tensors for self and cross attentions.
            self_attns = [layer.last_self_attn, ...]
            cross_attns = [layer.last_cross_attn, ...]
        Each element can be None (if not computed) or a tensor (B, nhead, Lh, Lh)/(B, nhead, Lh, Lm).
        """
        self_attns = []
        cross_attns = []
        for layer in self.harmony_encoder.layers:
            self_attns.append(layer.last_self_attn)
            cross_attns.append(layer.last_cross_attn)
        return self_attns, cross_attns
