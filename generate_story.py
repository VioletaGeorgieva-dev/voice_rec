from pathlib import Path
from inference import synthesize

INPUT_FILE = "lowecyt_i_valshebniqt_prysten.txt"
OUTPUT_FILE = "lowecyt_i_valshebniqt.wav"
REFERENCE_FILE = "reference.wav"
CHECKPOINT = "checkpoint_inference.pt"

text = Path(INPUT_FILE).read_text(encoding="utf-8").strip()

print(f"Зареден текст: {len(text)} символа")
print("Започвам генерирането на приказката...")
print()

synthesize(
    checkpoint=CHECKPOINT,
    text=text,
    output=OUTPUT_FILE,
    speaker_wav=REFERENCE_FILE,
    temperature=0.3,
    top_k=250,
    top_p=0.95,
    rep_penalty=1.1,
    max_tokens=512,
    device="cpu"
)

print()
print(f"ГОТОВО: {OUTPUT_FILE}")