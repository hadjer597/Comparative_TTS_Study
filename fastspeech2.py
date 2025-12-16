from TTS.api import TTS
import os

TEXT_TO_SAY = "Speech synthesis models generate natural voice signals."
OUTPUT_DIR = "resultsW"
OUTPUT_WAV = os.path.join(OUTPUT_DIR, "fastpitch_tts.wav")


MODEL_NAME = "tts_models/en/ljspeech/fast_pitch"

print(f"Loading FastPitch Model: {MODEL_NAME}...")
tts_model = TTS(model_name=MODEL_NAME, progress_bar=True)

tts_model.tts_to_file(text=TEXT_TO_SAY, file_path=OUTPUT_WAV)

print(f"✅ FastPitch audio saved to: {OUTPUT_WAV}")
