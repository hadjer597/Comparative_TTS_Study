import pyttsx3
import os

text = "Speech synthesis models generate natural voice signals."

output_dir_name = "resultsW"
output_file_name = "parametric_tts.wav"

full_output_path = os.path.join(output_dir_name, output_file_name)



engine = pyttsx3.init()

engine.save_to_file(text, full_output_path)
engine.runAndWait()

print(f"Parametric TTS audio generated successfully at: {full_output_path}")