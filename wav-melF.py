import os
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf


wav_file = "resultsW/fastpitch_tts.wav"
y, sr = librosa.load(wav_file, sr=None)

# ---------------- Waveform ----------------
plt.figure(figsize=(10, 3))
librosa.display.waveshow(y, sr=sr)
plt.title("Waveform - FastPitch")
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
plt.tight_layout()
plt.savefig("results/fastpitch_waveform.png", dpi=300)
plt.show()

# ---------------- Mel Spectrogram ----------------
S = librosa.feature.melspectrogram(
    y=y,
    sr=sr,
    n_fft=1024,
    hop_length=256,
    n_mels=80
)
S_dB = librosa.power_to_db(S, ref=np.max)

plt.figure(figsize=(10, 3))
librosa.display.specshow(
    S_dB,
    sr=sr,
    hop_length=256,
    x_axis="time",
    y_axis="mel",
    cmap="magma"
)
plt.colorbar(format="%+2.0f dB")
plt.title("Mel Spectrogram - FastPitch")
plt.xlabel("Time (s)")
plt.ylabel("Mel Frequency")
plt.tight_layout()
plt.savefig("results/fastpitch_mel.png", dpi=300)
plt.show()
