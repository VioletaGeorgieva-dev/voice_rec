"""
V6 Model — Encoder-Decoder TTS with MioCodec + Speaker Embedding
=================================================================
Architecture (V6 Small):
  - Text Encoder: 4-layer bidirectional Transformer (d=384, 6 heads, ff=1536)
    Learned positional embeddings, RMSNorm, SwiGLU
  - Audio Decoder: 8-layer causal Transformer (d=384, 6 heads, ff=1536)
    RoPE, cross-attention to encoder at every layer, RMSNorm, SwiGLU
  - Speaker Projection: Linear(128, 384) — MioCodec global_embedding → decoder dim

Key design:
  - enc_d == dec_d == 384 → no projection layer needed
  - Speaker embedding (128-dim) injected into decoder as additive bias
  - Tied decoder embeddings (lm_head = token_embedding.weight)
  - Gradient checkpointing in decoder during training
  - KV-cache for inference
  - ~38M params total

Target inference: RTF ~0.25-0.30 on RTX 5090
"""

import math
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict
from dataclasses import dataclass

from config import (
    TOTAL_VOCAB_SIZE, ENCODER_VOCAB_SIZE, DECODER_VOCAB_SIZE,
    ENC_D_MODEL, ENC_N_HEADS, ENC_N_LAYERS, ENC_D_FF,
    DEC_D_MODEL, DEC_N_HEADS, DEC_N_LAYERS, DEC_D_FF,
    MAX_TEXT_LEN, MAX_AUDIO_LEN, DROPOUT,
    PAD_TOKEN_ID, NUM_AUDIO_TOKENS, AUDIO_OFFSET,
    SPEAKER_EMB_DIM,
)


# ── Shared Components ──────────────────────────────────────────

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight


class RotaryPositionalEmbedding(nn.Module):
    def __init__(self, dim: int, max_seq_len: int = 4096, base: float = 10000.0):
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self._build_cache(max_seq_len)

    def _build_cache(self, seq_len: int):
        t = torch.arange(seq_len, device=self.inv_freq.device, dtype=self.inv_freq.dtype)
        freqs = torch.outer(t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos(), persistent=False)
        self.register_buffer("sin_cached", emb.sin(), persistent=False)

    def forward(self, seq_len: int) -> Tuple[torch.Tensor, torch.Tensor]:
        if seq_len > self.max_seq_len:
            self._build_cache(seq_len)
            self.max_seq_len = seq_len
        return self.cos_cached[:seq_len], self.sin_cached[:seq_len]


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin):
    cos = cos.unsqueeze(0).unsqueeze(0)
    sin = sin.unsqueeze(0).unsqueeze(0)
    return (q * cos + rotate_half(q) * sin,
            k * cos + rotate_half(k) * sin)


class SwiGLUFFN(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float):
        super().__init__()
        self.gate_proj = nn.Linear(d_model, d_ff, bias=False)
        self.up_proj   = nn.Linear(d_model, d_ff, bias=False)
        self.down_proj = nn.Linear(d_ff, d_model, bias=False)
        self.dropout   = nn.Dropout(dropout)

    def forward(self, x):
        return self.dropout(self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x)))


# ── Encoder (Bidirectional) ────────────────────────────────────

class EncoderSelfAttention(nn.Module):
    """Bidirectional self-attention for text encoder (NO causal mask)."""
    def __init__(self, d_model: int, n_heads: int, dropout: float):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        assert d_model % n_heads == 0

        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.o_proj = nn.Linear(d_model, d_model, bias=False)
        self.resid_dropout = nn.Dropout(dropout)

    def forward(self, x, key_padding_mask=None):
        B, T, _ = x.shape
        q = self.q_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)

        attn_mask = None
        if key_padding_mask is not None:
            attn_mask = key_padding_mask.unsqueeze(1).unsqueeze(2)  # [B, 1, 1, T]
            attn_mask = attn_mask.float() * torch.finfo(q.dtype).min

        attn_out = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=attn_mask,
            dropout_p=self.resid_dropout.p if self.training else 0.0,
            is_causal=False,
        )
        attn_out = attn_out.transpose(1, 2).contiguous().view(B, -1, self.d_model)
        return self.resid_dropout(self.o_proj(attn_out))


class EncoderBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float):
        super().__init__()
        self.attn_norm = RMSNorm(d_model)
        self.attention = EncoderSelfAttention(d_model, n_heads, dropout)
        self.ffn_norm  = RMSNorm(d_model)
        self.ffn       = SwiGLUFFN(d_model, d_ff, dropout)

    def forward(self, x, key_padding_mask=None):
        x = x + self.attention(self.attn_norm(x), key_padding_mask)
        x = x + self.ffn(self.ffn_norm(x))
        return x


class TextEncoder(nn.Module):
    """
    Bidirectional Transformer encoder for text.
    Input: text token IDs (special + chars, vocab 155)
    Output: contextualized text representations [B, T_text, d_model]
    """
    def __init__(self, vocab_size=ENCODER_VOCAB_SIZE, d_model=ENC_D_MODEL,
                 n_heads=ENC_N_HEADS, n_layers=ENC_N_LAYERS, d_ff=ENC_D_FF,
                 max_len=MAX_TEXT_LEN, dropout=DROPOUT):
        super().__init__()
        self.d_model = d_model
        self.token_embedding = nn.Embedding(vocab_size, d_model, padding_idx=PAD_TOKEN_ID)
        self.pos_embedding = nn.Embedding(max_len, d_model)
        self.embed_dropout = nn.Dropout(dropout)

        self.layers = nn.ModuleList([
            EncoderBlock(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])
        self.final_norm = RMSNorm(d_model)

    def forward(self, input_ids, attention_mask=None):
        B, T = input_ids.shape
        pos = torch.arange(T, device=input_ids.device).unsqueeze(0)
        h = self.embed_dropout(self.token_embedding(input_ids) + self.pos_embedding(pos))

        key_padding_mask = None
        if attention_mask is not None:
            key_padding_mask = (attention_mask == 0)

        for layer in self.layers:
            h = layer(h, key_padding_mask)

        return self.final_norm(h)


# ── Decoder (Causal with Cross-Attention + Speaker) ────────────

class DecoderSelfAttention(nn.Module):
    """Causal self-attention with RoPE and KV-cache."""
    def __init__(self, d_model: int, n_heads: int, dropout: float, max_len: int):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        assert d_model % n_heads == 0

        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.o_proj = nn.Linear(d_model, d_model, bias=False)
        self.resid_dropout = nn.Dropout(dropout)
        self.rope = RotaryPositionalEmbedding(self.head_dim, max_len)

    def forward(self, x, past_kv=None, use_cache=False):
        B, T, _ = x.shape
        q = self.q_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)

        # RoPE
        if past_kv is not None:
            offset = past_kv[0].shape[2]
            cos, sin = self.rope(offset + T)
            cos, sin = cos[offset:offset + T], sin[offset:offset + T]
        else:
            cos, sin = self.rope(T)

        q, k = apply_rotary_pos_emb(q, k, cos, sin)

        if past_kv is not None:
            k = torch.cat([past_kv[0], k], dim=2)
            v = torch.cat([past_kv[1], v], dim=2)

        new_kv = (k, v) if use_cache else None

        is_causal = (past_kv is None) and (T > 1)
        attn_mask = None
        if past_kv is not None and T > 1:
            # Cached prefixes need an offset mask when decoding multiple new tokens.
            query_positions = torch.arange(T, device=x.device) + past_kv[0].shape[2]
            key_positions = torch.arange(k.shape[2], device=x.device)
            attn_mask = key_positions.unsqueeze(0) <= query_positions.unsqueeze(1)
        attn_out = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=attn_mask,
            dropout_p=self.resid_dropout.p if self.training else 0.0,
            is_causal=is_causal,
        )
        attn_out = attn_out.transpose(1, 2).contiguous().view(B, -1, self.d_model)
        return self.resid_dropout(self.o_proj(attn_out)), new_kv


class CrossAttention(nn.Module):
    """Cross-attention: decoder queries attend to encoder keys/values."""
    def __init__(self, d_model: int, n_heads: int, dropout: float):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        assert d_model % n_heads == 0

        # Q from decoder, K/V from encoder — same dim since enc_d == dec_d
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.o_proj = nn.Linear(d_model, d_model, bias=False)
        self.resid_dropout = nn.Dropout(dropout)

    def forward(self, x, encoder_output, encoder_mask=None, cached_kv=None, use_cache=False):
        B, T, _ = x.shape
        q = self.q_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)

        if cached_kv is not None:
            k, v = cached_kv
        else:
            T_enc = encoder_output.shape[1]
            k = self.k_proj(encoder_output).view(B, T_enc, self.n_heads, self.head_dim).transpose(1, 2)
            v = self.v_proj(encoder_output).view(B, T_enc, self.n_heads, self.head_dim).transpose(1, 2)

        new_kv = (k, v) if use_cache else None

        attn_mask = None
        if encoder_mask is not None:
            attn_mask = (encoder_mask == 0).unsqueeze(1).unsqueeze(2)
            attn_mask = attn_mask.float() * torch.finfo(q.dtype).min

        attn_out = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=attn_mask,
            dropout_p=self.resid_dropout.p if self.training else 0.0,
            is_causal=False,
        )
        attn_out = attn_out.transpose(1, 2).contiguous().view(B, -1, self.d_model)
        return self.resid_dropout(self.o_proj(attn_out)), new_kv


class DecoderBlock(nn.Module):
    """Decoder block: self-attention → cross-attention → FFN"""
    def __init__(self, d_model: int, n_heads: int, d_ff: int,
                 dropout: float, max_len: int):
        super().__init__()
        self.self_attn_norm = RMSNorm(d_model)
        self.self_attention = DecoderSelfAttention(d_model, n_heads, dropout, max_len)

        self.cross_attn_norm = RMSNorm(d_model)
        self.cross_attention = CrossAttention(d_model, n_heads, dropout)

        self.ffn_norm = RMSNorm(d_model)
        self.ffn = SwiGLUFFN(d_model, d_ff, dropout)

    def forward(self, x, encoder_output, encoder_mask=None,
                past_self_kv=None, past_cross_kv=None, use_cache=False):
        # 1. Causal self-attention
        h = self.self_attn_norm(x)
        attn_out, new_self_kv = self.self_attention(h, past_self_kv, use_cache)
        x = x + attn_out

        # 2. Cross-attention to encoder
        h = self.cross_attn_norm(x)
        cross_out, new_cross_kv = self.cross_attention(
            h, encoder_output, encoder_mask, past_cross_kv, use_cache)
        x = x + cross_out

        # 3. FFN
        x = x + self.ffn(self.ffn_norm(x))

        return x, new_self_kv, new_cross_kv


class AudioDecoder(nn.Module):
    """
    Causal Transformer decoder with cross-attention + speaker embedding.
    Speaker embedding is added once to the token embeddings (like a global bias).
    """
    def __init__(self, vocab_size=DECODER_VOCAB_SIZE, d_model=DEC_D_MODEL,
                 n_heads=DEC_N_HEADS, n_layers=DEC_N_LAYERS, d_ff=DEC_D_FF,
                 max_len=MAX_AUDIO_LEN, dropout=DROPOUT,
                 speaker_emb_dim=SPEAKER_EMB_DIM):
        super().__init__()
        self.config_d_model = d_model
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.embed_dropout = nn.Dropout(dropout)

        # Speaker embedding projection: 128 → d_model (normalized)
        self.speaker_proj = nn.Linear(speaker_emb_dim, d_model, bias=False)
        self.register_buffer('spk_scale', torch.ones(1))  # fixed scale, not learnable

        self.layers = nn.ModuleList([
            DecoderBlock(d_model, n_heads, d_ff, dropout, max_len)
            for _ in range(n_layers)
        ])
        self.final_norm = RMSNorm(d_model)

        # LM head — tied with token embedding
        self.lm_head = None  # tied

    def forward(self, input_ids, encoder_output, encoder_mask=None,
                speaker_emb=None, labels=None,
                past_key_values=None, use_cache=False):
        """
        input_ids:      [B, T_dec]
        encoder_output: [B, T_enc, d_model]
        encoder_mask:   [B, T_enc]
        speaker_emb:    [B, 128] — MioCodec global_embedding
        labels:         [B, T_dec] — for training
        """
        h = self.token_embedding(input_ids)

        # Inject speaker embedding — normalized, additive, broadcast over time
        if speaker_emb is not None:
            spk = self.speaker_proj(speaker_emb)  # [B, d_model]
            spk = F.normalize(spk, dim=-1) * self.spk_scale  # normalize to unit norm
            h = h + spk.unsqueeze(1)  # [B, 1, d_model] broadcast

        h = self.embed_dropout(h)

        new_kvs = [] if use_cache else None
        for i, layer in enumerate(self.layers):
            past_self_kv = past_key_values[i][0] if past_key_values else None
            past_cross_kv = past_key_values[i][1] if past_key_values else None

            if self.training and not use_cache:
                h, self_kv, cross_kv = torch.utils.checkpoint.checkpoint(
                    layer, h, encoder_output, encoder_mask,
                    past_self_kv, past_cross_kv, use_cache,
                    use_reentrant=False)
            else:
                h, self_kv, cross_kv = layer(
                    h, encoder_output, encoder_mask,
                    past_self_kv, past_cross_kv, use_cache)

            if use_cache:
                new_kvs.append((self_kv, cross_kv))

        h = self.final_norm(h)

        # Tied embeddings
        logits = F.linear(h, self.token_embedding.weight)

        result = {"logits": logits}
        if use_cache:
            result["past_key_values"] = new_kvs

        if labels is not None:
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=-100,
            )
            result["loss"] = loss

        return result


# ── Full Encoder-Decoder Model ─────────────────────────────────

@dataclass
class V6Config:
    # Encoder
    enc_vocab_size: int = ENCODER_VOCAB_SIZE
    enc_d_model: int = ENC_D_MODEL
    enc_n_heads: int = ENC_N_HEADS
    enc_n_layers: int = ENC_N_LAYERS
    enc_d_ff: int = ENC_D_FF
    max_text_len: int = MAX_TEXT_LEN
    # Decoder
    dec_vocab_size: int = DECODER_VOCAB_SIZE
    dec_d_model: int = DEC_D_MODEL
    dec_n_heads: int = DEC_N_HEADS
    dec_n_layers: int = DEC_N_LAYERS
    dec_d_ff: int = DEC_D_FF
    max_audio_len: int = MAX_AUDIO_LEN
    # Speaker
    speaker_emb_dim: int = SPEAKER_EMB_DIM
    # Shared
    dropout: float = DROPOUT


class TTSEncoderDecoder(nn.Module):
    """
    V6 Encoder-Decoder TTS with MioCodec + Speaker Embedding.

    Forward flow:
    1. Text → Encoder → contextualized text representations [B, T_text, d_model]
    2. Audio tokens + speaker_emb → Decoder (with cross-attn) → logits
    """
    def __init__(self, config: V6Config):
        super().__init__()
        self.config = config

        # Text encoder (bidirectional)
        self.encoder = TextEncoder(
            vocab_size=config.enc_vocab_size,
            d_model=config.enc_d_model,
            n_heads=config.enc_n_heads,
            n_layers=config.enc_n_layers,
            d_ff=config.enc_d_ff,
            max_len=config.max_text_len,
            dropout=config.dropout,
        )

        # enc_d == dec_d → identity projection (no extra params)
        assert config.enc_d_model == config.dec_d_model, \
            f"V6 requires enc_d == dec_d, got {config.enc_d_model} vs {config.dec_d_model}"

        # Audio decoder (causal with cross-attention + speaker embedding)
        self.decoder = AudioDecoder(
            vocab_size=config.dec_vocab_size,
            d_model=config.dec_d_model,
            n_heads=config.dec_n_heads,
            n_layers=config.dec_n_layers,
            d_ff=config.dec_d_ff,
            max_len=config.max_audio_len,
            dropout=config.dropout,
            speaker_emb_dim=config.speaker_emb_dim,
        )

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def get_num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def encode(self, enc_ids, enc_mask=None):
        """Run encoder. Returns [B, T_enc, d_model]."""
        return self.encoder(enc_ids, enc_mask)

    def forward(self, enc_ids, dec_ids, enc_mask=None, dec_labels=None,
                speaker_emb=None):
        """
        Full forward: encoder → decoder → loss.

        Args:
            enc_ids:      [B, T_enc] — text token IDs
            dec_ids:      [B, T_dec] — audio token IDs (decoder input)
            enc_mask:     [B, T_enc] — 1=real, 0=pad
            dec_labels:   [B, T_dec] — decoder labels (-100 for masked)
            speaker_emb:  [B, 128] — MioCodec global_embedding
        """
        # 1. Encode text
        enc_out = self.encoder(enc_ids, enc_mask)  # [B, T_enc, d_model]

        # 2. Decode audio with cross-attention + speaker
        dec_out = self.decoder(dec_ids, enc_out, enc_mask,
                               speaker_emb=speaker_emb, labels=dec_labels)

        result = {"logits": dec_out["logits"]}
        if "loss" in dec_out:
            result["loss"] = dec_out["loss"]

        return result


# ── Factory functions ──────────────────────────────────────────

def create_model(device="cuda", dropout_override=None) -> TTSEncoderDecoder:
    """Create V6 encoder-decoder TTS model."""
    kwargs = {}
    if dropout_override is not None:
        kwargs["dropout"] = dropout_override
    config = V6Config(**kwargs)
    model = TTSEncoderDecoder(config)

    n = model.get_num_params()
    enc_n = sum(p.numel() for p in model.encoder.parameters())
    dec_n = sum(p.numel() for p in model.decoder.parameters())

    print(f"V6 Encoder-Decoder TTS with MioCodec + Speaker Embedding")
    print(f"   Total params:  {n:,} ({n/1e6:.1f}M)")
    print(f"   Encoder:       {enc_n:,} ({enc_n/1e6:.1f}M)")
    print(f"   Decoder:       {dec_n:,} ({dec_n/1e6:.1f}M)")
    print(f"   Enc: d={config.enc_d_model}, h={config.enc_n_heads}, "
          f"L={config.enc_n_layers}, ff={config.enc_d_ff}")
    print(f"   Dec: d={config.dec_d_model}, h={config.dec_n_heads}, "
          f"L={config.dec_n_layers}, ff={config.dec_d_ff}")
    print(f"   Speaker: {config.speaker_emb_dim}-dim → {config.dec_d_model}")
    print(f"   Dropout: {config.dropout}")

    model = model.to(device)
    return model


def save_checkpoint(model, optimizer, scheduler, step, loss, path, best_val_loss=None):
    """Save full training checkpoint."""
    os.makedirs(path, exist_ok=True)
    model_to_save = model._orig_mod if hasattr(model, "_orig_mod") else model

    torch.save({
        "model_state_dict": model_to_save.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
        "step": step,
        "loss": loss,
        "best_val_loss": best_val_loss,
        "config": {
            "enc_vocab_size": model_to_save.config.enc_vocab_size,
            "enc_d_model": model_to_save.config.enc_d_model,
            "enc_n_heads": model_to_save.config.enc_n_heads,
            "enc_n_layers": model_to_save.config.enc_n_layers,
            "enc_d_ff": model_to_save.config.enc_d_ff,
            "max_text_len": model_to_save.config.max_text_len,
            "dec_vocab_size": model_to_save.config.dec_vocab_size,
            "dec_d_model": model_to_save.config.dec_d_model,
            "dec_n_heads": model_to_save.config.dec_n_heads,
            "dec_n_layers": model_to_save.config.dec_n_layers,
            "dec_d_ff": model_to_save.config.dec_d_ff,
            "max_audio_len": model_to_save.config.max_audio_len,
            "speaker_emb_dim": model_to_save.config.speaker_emb_dim,
            "dropout": model_to_save.config.dropout,
        },
    }, f"{path}/checkpoint.pt")
    print(f"Saved: {path} (step {step}, loss {loss:.4f})")


def load_for_inference(checkpoint_path: str, device="cuda") -> TTSEncoderDecoder:
    """Load model from checkpoint for inference."""
    if os.path.isfile(checkpoint_path):
        ckpt_file = checkpoint_path
    else:
        ckpt_file = os.path.join(checkpoint_path, "checkpoint.pt")
    print(f"Loading from {ckpt_file}...")
    ckpt = torch.load(ckpt_file, map_location=device, weights_only=False)

    cfg = ckpt["config"]
    config = V6Config(
        enc_vocab_size=cfg["enc_vocab_size"],
        enc_d_model=cfg["enc_d_model"],
        enc_n_heads=cfg["enc_n_heads"],
        enc_n_layers=cfg["enc_n_layers"],
        enc_d_ff=cfg["enc_d_ff"],
        max_text_len=cfg["max_text_len"],
        dec_vocab_size=cfg["dec_vocab_size"],
        dec_d_model=cfg["dec_d_model"],
        dec_n_heads=cfg["dec_n_heads"],
        dec_n_layers=cfg["dec_n_layers"],
        dec_d_ff=cfg["dec_d_ff"],
        max_audio_len=cfg["max_audio_len"],
        speaker_emb_dim=cfg.get("speaker_emb_dim", SPEAKER_EMB_DIM),
        dropout=cfg["dropout"],
    )
    model = TTSEncoderDecoder(config)
    model.load_state_dict(ckpt["model_state_dict"])
    model = model.to(device).eval()

    n = model.get_num_params()
    print(f"Loaded! {n/1e6:.1f}M params, step {ckpt['step']}, loss {ckpt['loss']:.4f}")
    return model
