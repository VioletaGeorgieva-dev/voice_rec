"""
V6 Codec — MioCodec 25Hz wrapper
==================================
Single codebook, 12800 codes, 25fps, 24kHz.
Supports global_embedding for voice cloning.
"""

import torch
import numpy as np
import soundfile as sf
from pathlib import Path
from typing import Optional, Union

from config import (
    CODEC_MODEL_NAME, CODEC_SAMPLE_RATE,
    CODEC_CODEBOOK_SIZE, CODEC_FRAME_RATE,
)


class CodecV6:
    def __init__(self, device: str = "cuda"):
        self.device = device
        self.sample_rate = CODEC_SAMPLE_RATE  # 24000
        self.codebook_size = CODEC_CODEBOOK_SIZE  # 12800
        self.frame_rate = CODEC_FRAME_RATE  # 25.0
        self._load_model()

    def _load_model(self):
        from miocodec import MioCodecModel
        self.model = MioCodecModel.from_pretrained(CODEC_MODEL_NAME)
        self.model = self.model.to(self.device).eval()
        print(f"MioCodec loaded: {CODEC_MODEL_NAME}, {self.sample_rate}Hz, "
              f"{self.frame_rate}fps, {self.codebook_size} codes")

    @torch.no_grad()
    def encode(self, wav_path: str | Path) -> dict:
        """
        Encode wav file → MioCodec codes + global_embedding.
        """
        data, sr = sf.read(str(wav_path), dtype='float32')
        if data.ndim == 2:
            data = data.mean(axis=1)  # soundfile returns [samples, channels]
        waveform = torch.from_numpy(data)
        return self.encode_waveform(waveform, sr)

    @torch.no_grad()
    def encode_waveform(self, waveform: torch.Tensor, sr: int) -> dict:
        """
        Encode directly from waveform tensor.
        waveform: [samples] or [channels, samples]
        sr: int
        """
        if sr <= 0:
            raise ValueError('Sample rate must be positive.')
        if waveform.dim() not in (1, 2) or waveform.numel() == 0:
            raise ValueError('Expected non-empty [samples] or [channels, samples] audio.')
        waveform = waveform.float()
        if not torch.isfinite(waveform).all():
            raise ValueError('Reference audio contains non-finite samples.')
        if waveform.dim() == 2:
            waveform = waveform.mean(0)  # documented channel-first tensor layout
            
        if sr != self.sample_rate:
            import torchaudio
            waveform = torchaudio.functional.resample(waveform, sr, self.sample_rate)

        audio = waveform.to(self.device).float()

        # MioCodec encode returns (content_token_indices, global_embedding)
        result = self.model.encode(audio)
        codes = result.content_token_indices.reshape(-1).cpu()    # [num_frames]
        global_emb = result.global_embedding.squeeze().cpu()       # [128]

        return {
            'codes': codes,
            'global_embedding': global_emb,
        }

    @torch.no_grad()
    def decode(self, codes: torch.Tensor,
               global_embedding: torch.Tensor) -> torch.Tensor:
        """
        Decode MioCodec codes → waveform.

        Args:
            codes: [num_frames] — token indices in [0, 12799]
            global_embedding: [128] — speaker embedding

        Returns:
            waveform: [samples] float32
        """
        codes = codes.to(self.device, dtype=torch.long).reshape(-1)
        global_embedding = global_embedding.to(self.device)

        # MioCodec expects flat tensors: codes [num_frames], emb [128]
        if codes.dim() > 1:
            codes = codes.squeeze()
        if global_embedding.dim() > 1:
            global_embedding = global_embedding.squeeze()

        audio = self.model.decode(
            global_embedding=global_embedding,
            content_token_indices=codes,
        )
        return audio.squeeze().cpu().float()

    def encode_to_tokens(self, wav_path: str) -> dict:
        """Convenience: encode and return codes + embedding."""
        return self.encode(wav_path)

    def tokens_to_wav(self, codes: torch.Tensor,
                      global_embedding: torch.Tensor,
                      output: Optional[str] = None) -> torch.Tensor:
        """Decode tokens to wav, optionally save."""
        wav = self.decode(codes, global_embedding)
        if output:
            sf.write(output, wav.numpy(), self.sample_rate)
        return wav

    def get_stats(self, wav_path: str) -> dict:
        """Get encoding stats for a wav file."""
        result = self.encode(wav_path)
        data, sr = sf.read(str(wav_path), dtype='float32')
        dur = len(data) / sr if data.ndim == 1 else data.shape[0] / sr
        n_tokens = len(result['codes'])
        return {
            "duration_sec": dur,
            "num_tokens": n_tokens,
            "tokens_per_sec": n_tokens / dur if dur > 0 else 0,
            "global_emb_shape": tuple(result['global_embedding'].shape),
        }
