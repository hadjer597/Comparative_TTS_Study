from TTS.api import TTS

text = "Speech synthesis models generate natural voice signals."

tts = TTS(model_name="tts_models/en/ljspeech/tacotron2-DDC", progress_bar=False)


tts.tts_to_file(text=text, file_path="tacotron2_tts.wav")

print("Tacotron2 TTS audio generated successfully.")
