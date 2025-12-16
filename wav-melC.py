import os
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np

INPUT_WAV = "resultsW/concatenative.wav"
OUTPUT_DIR = "results"
WAVEFORM_IMG = os.path.join(OUTPUT_DIR, "concatenative_waveform.png")
MEL_IMG = os.path.join(OUTPUT_DIR, "concatenative_mel.png")

# تحميل الصوت
y, sr = librosa.load(INPUT_WAV, sr=None)

# ---------------- Waveform ----------------
plt.figure(figsize=(10, 3))
librosa.display.waveshow(y, sr=sr)
plt.title("Waveform - Concatenative ")
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
plt.tight_layout()
plt.savefig(WAVEFORM_IMG, dpi=300)
plt.show()
print(f"✅ Waveform image saved to: {WAVEFORM_IMG}")

# ---------------- Mel Spectrogram ----------------
S = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=1024, hop_length=256, n_mels=80)
S_dB = librosa.power_to_db(S, ref=np.max)

plt.figure(figsize=(10, 3))
librosa.display.specshow(S_dB, sr=sr, hop_length=256, x_axis="time", y_axis="mel", cmap="magma")
plt.colorbar(format="%+2.0f dB")
plt.title("Mel Spectrogram - Concatenative ")
plt.xlabel("Time (s)")
plt.ylabel("Mel Frequency")
plt.tight_layout()
plt.savefig(MEL_IMG, dpi=300)
plt.show()
print(f"✅ Mel Spectrogram image saved to: {MEL_IMG}")
