from pathlib import Path
from inference import synthesize

# =========================
# НАСТРОЙКИ
# =========================

INPUT_FILE = "юначното_петле.txt"
REFERENCE_FILE = "reference.wav"
CHECKPOINT = "checkpoint_inference.pt"


# =========================
# ОСНОВНА ПРОГРАМА
# =========================

def main():
    input_path = Path(INPUT_FILE)

    if not input_path.exists():
        print(f"Грешка: не е намерен файлът {INPUT_FILE}")
        return

    text = input_path.read_text(encoding="utf-8").strip()

    if not text:
        print("Грешка: текстовият файл е празен.")
        return

    # Папка за готовите WAV файлове
    output_dir = Path(input_path.stem)
    output_dir.mkdir(exist_ok=True)

    output_file = output_dir / "story.wav"

    print(f"Зареден текст: {len(text)} символа")
    print()
    print("Започвам генерирането...")
    print()

    synthesize(
        checkpoint=CHECKPOINT,
        text=text,
        output=str(output_file),
        speaker_wav=REFERENCE_FILE,
        temperature=0.65,  # Премахва цикленето и повторението на гласни ("Иииии")
        top_k=250,
        top_p=0.95,
        rep_penalty=1.2,   # Предотвратява повторението на аудио токени
        max_tokens=1024,   # Гарантира достатъчно време (~41 сек) за всеки chunk
        device="cpu"
    )

    print()
    print("=" * 60)
    print("ПРИКАЗКАТА Е ГОТОВА!")
    print(f"Файловете са в папката: {output_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()