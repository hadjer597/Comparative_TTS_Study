from TTS.api import TTS
import os

TEXT_TO_SAY = "Speech synthesis models generate natural voice signals."
OUTPUT_DIR = "resultsW"
OUTPUT_WAV = os.path.join(OUTPUT_DIR, "vits_tts.wav")



MODEL_NAME = "tts_models/en/ljspeech/vits"
print(f"Loading VITS Model: {MODEL_NAME}...")
tts_model = TTS(model_name=MODEL_NAME, progress_bar=True)

tts_model.tts_to_file(text=TEXT_TO_SAY, file_path=OUTPUT_WAV)
print(f"✅ VITS audio saved to: {OUTPUT_WAV}")
