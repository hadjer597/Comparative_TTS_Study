import time
import psutil
import librosa
import numpy as np
import soundfile as sf

# ---------------- الإعدادات ----------------
WAV_PATH = "resultsW/concatenative_tts.wav"

# ⚠️ أدخلي زمن التوليد الذي سجلتيه عند إنشاء الصوت
ELAPSED_TIME = 0.10  # مثال

# ---------------- تحميل الصوت ----------------
y, sr = librosa.load(WAV_PATH, sr=None)

# ---------------- المعايير الزمنية ----------------
audio_duration = librosa.get_duration(y=y, sr=sr)
rtf = ELAPSED_TIME / audio_duration

cpu_usage = psutil.cpu_percent(interval=1)
ram_usage = psutil.virtual_memory().used / (1024 ** 2)  # MB

info = sf.info(WAV_PATH)
sample_rate = info.samplerate

# ---------------- Pitch (F0) ----------------
f0, voiced_flag, voiced_prob = librosa.pyin(
    y,
    fmin=50,
    fmax=300,
    sr=sr
)

mean_pitch = np.nanmean(f0)
pitch_variability = np.nanstd(f0)

# ---------------- Energy ----------------
frame_energy = librosa.feature.rms(y=y)[0]
mean_energy = np.mean(frame_energy)
energy_variability = np.std(frame_energy)

# ---------------- Spectral Features ----------------
spectral_centroid = np.mean(
    librosa.feature.spectral_centroid(y=y, sr=sr)
)
spectral_bandwidth = np.mean(
    librosa.feature.spectral_bandwidth(y=y, sr=sr)
)

# ---------------- SNR ----------------
signal_power = np.mean(y ** 2)
noise_power = np.mean((y - np.mean(y)) ** 2)
snr = 10 * np.log10(signal_power / noise_power)

# ---------------- Log-Mel Energy ----------------
mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=80)
mel_db = librosa.power_to_db(mel)
log_mel_energy = np.mean(np.abs(mel_db))

# ---------------- Jitter & Shimmer (تقريبي) ----------------
periods = 1 / f0
jitter = np.nanmean(np.abs(np.diff(periods)))

amplitudes = np.abs(y)
shimmer = np.mean(np.abs(np.diff(amplitudes)))

# ---------------- طباعة النتائج ----------------
print("\n📊 Concatenative TTS Metrics:\n")

print(f"Elapsed Time (s): {ELAPSED_TIME:.2f}")
print(f"Audio Duration (s): {audio_duration:.2f}")
print(f"Real-Time Factor (RTF): {rtf:.2f}")
print(f"CPU Usage (%): {cpu_usage}")
print(f"RAM Usage (MB): {ram_usage:.2f}")
print(f"Sample Rate (Hz): {sample_rate}")

print("\n🔊 Acoustic Quality:")
print(f"Mean Pitch (Hz): {mean_pitch:.2f}")
print(f"Pitch Variability (Std): {pitch_variability:.2f}")
print(f"Mean Energy: {mean_energy:.4f}")
print(f"Energy Variability: {energy_variability:.4f}")

print("\n🌈 Spectral Metrics:")
print(f"Spectral Centroid: {spectral_centroid:.2f}")
print(f"Spectral Bandwidth: {spectral_bandwidth:.2f}")

print("\n📐 Advanced Metrics:")
print(f"SNR (dB): {snr:.2f}")
print(f"Log-Mel Energy: {log_mel_energy:.2f}")
print(f"Jitter (approx): {jitter:.6f}")
print(f"Shimmer (approx): {shimmer:.6f}")
