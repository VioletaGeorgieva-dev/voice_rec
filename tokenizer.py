"""
V6 Tokenizer — char-level for Bulgarian TTS with MioCodec
==========================================================
Same character set as V5, but adapted for:
  - MioCodec single codebook (no interleaving)
  - Speaker embedding (no speaker tokens in encoder input)
"""

import re
import unicodedata
import torch
from typing import Optional

from config import (
    TEXT_CHARS, TEXT_OFFSET, AUDIO_OFFSET,
    SPECIAL_TOKENS, NUM_SPECIAL_TOKENS, CODEC_CODEBOOK_SIZE,
    TOTAL_VOCAB_SIZE,
    PAD_TOKEN_ID, START_OF_TEXT_TOKEN_ID, END_OF_TEXT_TOKEN_ID,
    START_OF_SPEECH_TOKEN_ID, END_OF_SPEECH_TOKEN_ID,
    is_audio_token, is_special_token, is_text_token,
)


class TTSTokenizer:
    def __init__(self):
        self.char2id: dict[str, int] = {}
        self.id2char: dict[int, str] = {}
        for i, ch in enumerate(TEXT_CHARS):
            tid = TEXT_OFFSET + i
            self.char2id[ch] = tid
            self.id2char[tid] = ch

        self._special_id_to_name = {v: k for k, v in SPECIAL_TOKENS.items()}
        self.vocab_size = TOTAL_VOCAB_SIZE
        self.text_vocab_size = len(TEXT_CHARS)

    def normalize_text(self, text: str) -> str:
        text = unicodedata.normalize('NFC', text)
        # Preserve Bulgarian stressed и without changing trained token IDs.
        text = text.translate(str.maketrans({'ѝ': 'и', 'Ѝ': 'И', '“': '"', '”': '"', '‘': "'", '’': "'"}))
        text = re.sub(r'\s+', ' ', text).strip()
        text = re.sub(r'[–—]', '-', text)
        text = re.sub(r'[«»„""]', '"', text)
        return text

    def encode_text(self, text: str) -> list[int]:
        text = self.normalize_text(text)
        return [self.char2id[ch] for ch in text if ch in self.char2id]

    def decode_text(self, ids: list[int]) -> str:
        return "".join(self.id2char.get(t, "") for t in ids if is_text_token(t))

    # ── Encoder-Decoder methods ──────────────────────────────

    def build_encoder_input(self, text: str) -> torch.Tensor:
        """
        Encoder input: <sot> text_chars <eot>
        No speaker token — speaker info comes from embedding.
        """
        text_ids = self.encode_text(text)
        seq = [START_OF_TEXT_TOKEN_ID] + text_ids + [END_OF_TEXT_TOKEN_ID]
        return torch.tensor(seq, dtype=torch.long)

    def build_decoder_input(self, audio_codes: torch.Tensor) -> torch.Tensor:
        """
        Decoder input: <sos> [audio_codes + AUDIO_OFFSET] <eos>
        audio_codes: raw MioCodec codes in [0, 12799]
        """
        seq = (
            [START_OF_SPEECH_TOKEN_ID]
            + (audio_codes + AUDIO_OFFSET).tolist()
            + [END_OF_SPEECH_TOKEN_ID]
        )
        return torch.tensor(seq, dtype=torch.long)

    def build_decoder_prefix(self) -> torch.Tensor:
        """For inference: just <sos> to start generation."""
        return torch.tensor([START_OF_SPEECH_TOKEN_ID], dtype=torch.long)

    def extract_audio_codes(self, sequence: torch.Tensor) -> Optional[torch.Tensor]:
        """Extract raw MioCodec codes from a token sequence."""
        mask = torch.tensor([is_audio_token(t.item()) for t in sequence])
        if not mask.any():
            return None
        return sequence[mask] - AUDIO_OFFSET

    def describe(self, seq: torch.Tensor, max_tok: int = 30) -> str:
        parts = []
        for t in seq[:max_tok]:
            tid = t.item()
            if is_special_token(tid):
                parts.append(self._special_id_to_name.get(tid, f"<sp_{tid}>"))
            elif is_text_token(tid):
                ch = self.id2char.get(tid, "?")
                parts.append(ch if ch != " " else "·")
            elif is_audio_token(tid):
                code = tid - AUDIO_OFFSET
                parts.append(f"♪{code}")
            else:
                parts.append(f"?{tid}")
        r = " ".join(parts)
        if len(seq) > max_tok:
            r += f" ... [{len(seq) - max_tok} more]"
        return r
