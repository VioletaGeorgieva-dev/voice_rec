import subprocess

INPUT_FILE = "slivi_za_smet.txt"
OUTPUT_FILE = "slivi_za_smet.wav"
REFERENCE_FILE = "reference.wav"
CHECKPOINT = "checkpoint_inference.pt"

with open(INPUT_FILE, "r", encoding="utf-8") as f:
    text = f.read().strip()

command = [
    "python",
    "inference.py",
    "--checkpoint", CHECKPOINT,
    "--text", text,
    "--speaker-wav", REFERENCE_FILE,
    "--output", OUTPUT_FILE,
    "--device", "cpu",
]

print("Започвам генерирането на приказката...")
print("Това може да отнеме няколко минути.")

subprocess.run(command, check=True)

print()
print(f"Готово! Файлът е: {OUTPUT_FILE}")