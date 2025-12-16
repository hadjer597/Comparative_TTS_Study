import subprocess
import os

TEXT_TO_SAY = "Speech synthesis models generate natural voice signals."
OUTPUT_DIR = "resultsW"
OUTPUT_WAV = os.path.join(OUTPUT_DIR, "concatenative_tts.wav")


ESPEAK_PATH = "C:\Program Files (x86)\eSpeak\command_line\espeak.exe"


subprocess.run([ESPEAK_PATH, "-w", OUTPUT_WAV, TEXT_TO_SAY], check=True)
print(f"✅ Concatenative TTS audio saved to: {OUTPUT_WAV}")
