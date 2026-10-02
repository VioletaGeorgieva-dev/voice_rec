"""
BgTTS-38M Web Server — Gradio Interface
========================================
Voice cloning TTS with Bulgarian + English support.
"""

import sys
import os
import torch
import numpy as np
import tempfile
import time
import soundfile as sf

# Add parent dir to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import (
    AUDIO_OFFSET, NUM_AUDIO_TOKENS, END_OF_SPEECH_TOKEN_ID,
    START_OF_SPEECH_TOKEN_ID, CODEC_SAMPLE_RATE, CODEC_FRAME_RATE,
)
from tokenizer import TTSTokenizer
from codec import CodecV6
from model import load_for_inference
from inference import generate, _split_text

# ── Global state ──────────────────────────────────────────────
MODEL = None
TOKENIZER = None
CODEC = None
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CHECKPOINT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "checkpoint_inference.pt")


def load_model():
    """Load model, tokenizer, codec once at startup."""
    global MODEL, TOKENIZER, CODEC
    print(f"Loading model from {CHECKPOINT_PATH} on {DEVICE}...")
    MODEL = load_for_inference(CHECKPOINT_PATH, device=DEVICE)
    TOKENIZER = TTSTokenizer()
    CODEC = CodecV6(device=DEVICE)
    print("Model loaded!")


def synthesize_speech(text, ref_audio, temperature, top_k, top_p, rep_penalty):
    """
    Generate speech from text using reference audio for voice cloning.
    
    Returns: (sample_rate, audio_array) tuple for Gradio
    """
    if not text or not text.strip():
        return None, "Въведи текст за синтез."
    
    if ref_audio is None:
        return None, "Качи примерен запис за гласа."
    
    # Encode reference audio for speaker embedding
    sr_ref, audio_ref = ref_audio
    audio_ref = audio_ref.astype(np.float32)
    if audio_ref.size == 0 or not np.isfinite(audio_ref).all():
        return None, "Записът е празен или съдържа невалидни семпли."
    if sr_ref <= 0 or np.max(np.abs(audio_ref)) == 0:
        return None, "Записът трябва да съдържа звук и да има валидна честота."
    if audio_ref.max() > 1.0 or audio_ref.min() < -1.0:
        audio_ref = audio_ref / max(abs(audio_ref.max()), abs(audio_ref.min()))
    
    waveform = torch.from_numpy(audio_ref)
    if waveform.dim() == 2:
        waveform = waveform.mean(1)
    
    result = CODEC.encode_waveform(waveform, sr_ref)
    speaker_emb = result['global_embedding'].to(DEVICE)
    
    # Split text into chunks
    chunks = _split_text(text, TOKENIZER, max_len=min(250, MODEL.config.max_text_len))
    
    t0 = time.time()
    all_codes = []
    for chunk in chunks:
        codes = generate(
            MODEL, TOKENIZER, chunk, speaker_emb,
            max_new_tokens=512,
            temperature=temperature,
            top_k=int(top_k),
            top_p=top_p,
            rep_penalty=rep_penalty,
            device=DEVICE
        )
        if codes is None or len(codes) == 0:
            return None, "Не е генериран звук за част от текста. Опитай с по-кратки изречения."
        all_codes.append(codes)
    
    gen_time = time.time() - t0
    
    if not all_codes:
        return None, "Не е генериран звук."
    
    codes = torch.cat(all_codes)
    audio_dur = len(codes) / CODEC_FRAME_RATE
    rtf = gen_time / audio_dur if audio_dur > 0 else float('inf')
    
    # Decode to waveform
    wav = CODEC.decode(codes, speaker_emb)
    wav_np = wav.numpy()
    
    info = f"✅ {len(codes)} tokens | {audio_dur:.1f}s audio | {gen_time:.1f}s gen | RTF: {rtf:.3f}"
    
    return (CODEC_SAMPLE_RATE, wav_np), info


def build_ui():
    """Build Gradio interface."""
    import gradio as gr
    
    with gr.Blocks(title="BgTTS-38M V2 — Bulgarian Text-to-Speech") as app:
        gr.HTML('<h1 class="main-title">🎙️ BgTTS-38M</h1>')
        gr.HTML('<p class="subtitle">Bulgarian + English Text-to-Speech with Voice Cloning | 38M params | 153MB</p>')
        
        with gr.Row():
            with gr.Column(scale=2):
                text_input = gr.Textbox(
                    label="Текст / Text",
                    placeholder="Въведете текст на български или английски...\nEnter text in Bulgarian or English...",
                    lines=5,
                    max_lines=15,
                )
                
                ref_audio = gr.Audio(
                    label="🎤 Reference Voice (за клониране на глас)",
                    type="numpy",
                    sources=["upload", "microphone"],
                )
                
                with gr.Row():
                    generate_btn = gr.Button("🔊 Генерирай / Generate", variant="primary", size="lg")
                    clear_btn = gr.Button("🗑️ Изчисти", size="lg")
                
            with gr.Column(scale=1):
                with gr.Accordion("⚙️ Настройки / Settings", open=False):
                    temperature = gr.Slider(
                        minimum=0.05, maximum=1.5, value=0.3, step=0.05,
                        label="Temperature",
                        info="По-ниска = по-чисто, по-висока = по-разнообразно"
                    )
                    top_k = gr.Slider(
                        minimum=1, maximum=500, value=250, step=10,
                        label="Top-K"
                    )
                    top_p = gr.Slider(
                        minimum=0.1, maximum=1.0, value=0.95, step=0.05,
                        label="Top-P (Nucleus)"
                    )
                    rep_penalty = gr.Slider(
                        minimum=1.0, maximum=2.0, value=1.1, step=0.05,
                        label="Repetition Penalty"
                    )
                
                output_audio = gr.Audio(
                    label="🔊 Резултат / Output",
                    type="numpy",
                    interactive=False,
                )
                
                info_text = gr.Textbox(
                    label="ℹ️ Информация",
                    interactive=False,
                    lines=2,
                )
        
        # Examples
        gr.Examples(
            examples=[
                ["Българският език е изключително богат и мелодичен."],
                ["Artificial intelligence has reached a fascinating stage."],
                ["Когато говорим за истински multitasking, способността ми да превключвам плавно между български и English е от огромно значение."],
                ["Здравейте! Казвам се Ани и мога да говоря на български и английски."],
                ["The quick brown fox jumps over the lazy dog."],
            ],
            inputs=[text_input],
            label="📝 Примери / Examples",
        )
        
        # Event handlers
        generate_btn.click(
            fn=synthesize_speech,
            inputs=[text_input, ref_audio, temperature, top_k, top_p, rep_penalty],
            outputs=[output_audio, info_text],
        )
        
        clear_btn.click(
            fn=lambda: (None, None, ""),
            outputs=[text_input, output_audio, info_text],
        )
    
    return app


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", default=CHECKPOINT_PATH)
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=7860)
    p.add_argument("--share", action="store_true")
    p.add_argument("--device", default=DEVICE)
    args = p.parse_args()
    
    CHECKPOINT_PATH = args.checkpoint
    DEVICE = args.device
    
    load_model()
    app = build_ui()
    app.launch(
        server_name=args.host,
        server_port=args.port,
        share=args.share,
    )
