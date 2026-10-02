"""
V6 Config — Encoder-Decoder TTS with MioCodec + Speaker Embedding
==================================================================
Vocab layout:
  [0..8]        = 9 special tokens
  [9..154]      = ~146 text chars (BG + EN + digits + punct)
  [155..12954]  = 12,800 audio tokens (MioCodec, 1 codebook)
  Total = 12,955

Architecture:
  Encoder: 4L bidirectional, d=384, 6 heads — text understanding
  Decoder: 8L causal + cross-attention, d=384, 6 heads — audio generation
  Speaker: 128-dim global_embedding → Linear(128, 384) → added to decoder

Key differences from V5:
  - MioCodec (25fps, 1CB, 12800) instead of NanoCodec (12.5fps, 4CB, 16128)
  - d=384 for both encoder and decoder (V5: enc=512, dec=768)
  - 8 decoder layers (V5: 18)
  - Speaker embedding injection (V5: discrete speaker tokens)
  - max_text=256, max_audio=1024 (V5: 512/2048) — OPTIMIZED FOR LONG STORIES
  - ~40M params (V5: 250M)
  - Expected RTF ~0.15-0.25 (V5: 1.1)
"""

# ── MioCodec 25Hz ──────────────────────────────────────────────
CODEC_MODEL_NAME     = "Aratako/MioCodec-25Hz-24kHz"
CODEC_SAMPLE_RATE    = 24_000
CODEC_NUM_CODEBOOKS  = 1
CODEC_CODEBOOK_SIZE  = 12_800
CODEC_FRAME_RATE     = 25.0
CODEC_TOKENS_PER_SEC = 25  # 25fps × 1 codebook
TOKENS_PER_FRAME     = 1
SPEAKER_EMB_DIM      = 128  # MioCodec global_embedding dimension

# ── Character set (same as V5) ─────────────────────────────────
BG_LOWER  = "абвгдежзийклмнопрстуфхцчшщъьюя"
BG_UPPER  = "АБВГДЕЖЗИЙКЛМНОПРСТУФХЦЧШЩЪЬЮЯ"
EN_LOWER  = "abcdefghijklmnopqrstuvwxyz"
EN_UPPER  = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
DIGITS    = "0123456789"
PUNCT     = '.,!?;:-–—…"\'()[]{}«»„"" '
EXTRA     = "\n\t"

_ALL_CHARS: list[str] = []
_seen: set[str] = set()
for _src in [BG_LOWER, BG_UPPER, EN_LOWER, EN_UPPER, DIGITS, PUNCT, EXTRA]:
    for _ch in _src:
        if _ch not in _seen:
            _ALL_CHARS.append(_ch)
            _seen.add(_ch)

# ── Special tokens (indices 0..8) ──────────────────────────────
SPECIAL_TOKENS = {
    "<pad>":             0,
    "<start_of_text>":   1,
    "<end_of_text>":     2,
    "<start_of_speech>": 3,
    "<end_of_speech>":   4,
    "<spk_0>":           5,  # kept for compatibility, but speaker embedding is primary
    "<spk_1>":           6,
    "<spk_2>":           7,
    "<spk_3>":           8,
}
NUM_SPECIAL_TOKENS = len(SPECIAL_TOKENS)     # 9

# ── Vocab offsets ───────────────────────────────────────────────
TEXT_CHARS       = _ALL_CHARS
TEXT_VOCAB_SIZE  = len(TEXT_CHARS)             # ~146
TEXT_OFFSET      = NUM_SPECIAL_TOKENS         # 9
AUDIO_OFFSET     = TEXT_OFFSET + TEXT_VOCAB_SIZE  # 155
NUM_AUDIO_TOKENS = CODEC_CODEBOOK_SIZE            # 12,800
TOTAL_VOCAB_SIZE = AUDIO_OFFSET + NUM_AUDIO_TOKENS  # 12,955

# Encoder needs only text vocab; decoder needs full vocab
ENCODER_VOCAB_SIZE = AUDIO_OFFSET      # 155 (special + text)
DECODER_VOCAB_SIZE = TOTAL_VOCAB_SIZE  # 12,955 (full)

# ── Convenience IDs ─────────────────────────────────────────────
PAD_TOKEN_ID             = SPECIAL_TOKENS["<pad>"]
START_OF_TEXT_TOKEN_ID   = SPECIAL_TOKENS["<start_of_text>"]
END_OF_TEXT_TOKEN_ID     = SPECIAL_TOKENS["<end_of_text>"]
START_OF_SPEECH_TOKEN_ID = SPECIAL_TOKENS["<start_of_speech>"]
END_OF_SPEECH_TOKEN_ID   = SPECIAL_TOKENS["<end_of_speech>"]
SPK_0_TOKEN_ID           = SPECIAL_TOKENS["<spk_0>"]
SPK_1_TOKEN_ID           = SPECIAL_TOKENS["<spk_1>"]

# ── Helper functions ────────────────────────────────────────────
def audio_token_id(code: int) -> int:
    """MioCodec code → global token ID."""
    return AUDIO_OFFSET + code

def decode_audio_token(token_id: int) -> int:
    """Global token ID → MioCodec code."""
    return token_id - AUDIO_OFFSET

def is_audio_token(token_id: int) -> bool:
    return AUDIO_OFFSET <= token_id < AUDIO_OFFSET + NUM_AUDIO_TOKENS

def is_special_token(token_id: int) -> bool:
    return 0 <= token_id < NUM_SPECIAL_TOKENS

def is_text_token(token_id: int) -> bool:
    return TEXT_OFFSET <= token_id < AUDIO_OFFSET

# ── V6 Model Config ────────────────────────────────────────────
# Encoder: 4 bidirectional layers
ENC_D_MODEL    = 384
ENC_N_HEADS    = 6
ENC_N_LAYERS   = 4
ENC_D_FF       = 1536

# Decoder: 8 causal layers with cross-attention
DEC_D_MODEL    = 384
DEC_N_HEADS    = 6
DEC_N_LAYERS   = 8
DEC_D_FF       = 1536

MAX_TEXT_LEN   = 256         # Max text tokens (chars) — covers ~17s speech
MAX_AUDIO_LEN  = 1024        # Max audio tokens — 1024/25 = 41s (OPTIMIZED FOR STORIES)
DROPOUT        = 0.0

# ── Long-form synthesis config (for stories) ───────────────────
STORY_CHUNK_LEN = 200        # Target chunk length in tokens (adaptive for stories)
STORY_MIN_CHUNK = 80         # Minimum chunk length to avoid too many splits
STORY_OVERLAP   = 20         # Token overlap between chunks for context preservation

# ── Training defaults ──────────────────────────────────────────
BATCH_SIZE     = 16          # Smaller model = bigger batch
GRAD_ACCUM     = 4           # effective = 64
LR             = 3e-4
WEIGHT_DECAY   = 0.1
WARMUP_STEPS   = 1000
NUM_EPOCHS     = 5

# ── Print summary ──────────────────────────────────────────────
if __name__ == "__main__":
    print(f"V6 Vocab Layout:")
    print(f"  Special:  [0, {NUM_SPECIAL_TOKENS-1}]  ({NUM_SPECIAL_TOKENS} tokens)")
    print(f"  Text:     [{TEXT_OFFSET}, {AUDIO_OFFSET-1}]  ({TEXT_VOCAB_SIZE} chars)")
    print(f"  Audio:    [{AUDIO_OFFSET}, {TOTAL_VOCAB_SIZE-1}]  ({NUM_AUDIO_TOKENS} tokens)")
    print(f"  TOTAL:    {TOTAL_VOCAB_SIZE}")
    print()
    print(f"V6 Encoder: d={ENC_D_MODEL}, heads={ENC_N_HEADS}, L={ENC_N_LAYERS}, ff={ENC_D_FF}")
    print(f"V6 Decoder: d={DEC_D_MODEL}, heads={DEC_N_HEADS}, L={DEC_N_LAYERS}, ff={DEC_D_FF}")
    print(f"V6 Codec:   MioCodec {CODEC_FRAME_RATE}fps, {CODEC_NUM_CODEBOOKS}CB × {CODEC_CODEBOOK_SIZE}")
    print(f"V6 Speaker: {SPEAKER_EMB_DIM}-dim global_embedding")
    print(f"V6 Limits:  max_text={MAX_TEXT_LEN}, max_audio={MAX_AUDIO_LEN}")
    print(f"V6 Stories: chunk_len={STORY_CHUNK_LEN}, overlap={STORY_OVERLAP}")

