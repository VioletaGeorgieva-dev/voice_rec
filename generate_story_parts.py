from pathlib import Path

from inference import synthesize


# =========================
# НАСТРОЙКИ
# =========================

INPUT_FILE = "lowecyt_i_valshebniqt_prysten.txt"
REFERENCE_FILE = "references.wav"
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

    # Това е само базово име.
    # inference.py автоматично ще направи:
    #
    # 001.wav
    # 002.wav
    # 003.wav
    # ...
    #
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
        temperature=0.3,
        top_k=250,
        top_p=0.95,
        rep_penalty=1.1,
        max_tokens=512,
        device="cpu"
    )

    print()
    print("=" * 60)
    print("ПРИКАЗКАТА Е ГОТОВА!")
    print(f"Файловете са в папката: {output_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()