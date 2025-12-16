import matplotlib.pyplot as plt
import librosa
import librosa.display
import numpy as np

# =========================================================================
# 1. SETUP: File Paths and Model Names (Customize this section)
# =========================================================================

# **تنبيه: يجب استبدال هذه المسارات بمسارات ملفات WAV الفعلية لديكِ**
audio_files = {
    'Concatenative': 'resultsW/concatenative_tts.wav',
    'Parametric': 'resultsW/parametric_tts.wav',
    'Tacotron 2': 'resultsW/tacotron2_tts.wav',
    'FastPitch': 'resultsW/fastpitch_tts.wav',
    'VITS': 'resultsW/vits_tts.wav'
}

# تحديد الترتيب
model_order = list(audio_files.keys())

# =========================================================================
# 2. Figure 7.2: Integrated Mel Spectrogram Plot (Enhanced)
# =========================================================================

# 2 صفوف و 3 أعمدة (2 rows, 3 columns)
fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(15, 8))
plt.suptitle('Figure 7.2: Comparison of Mel Spectrogram (Spectral Richness Analysis)', 
             fontsize=14, y=1.05, fontweight='bold')

# إعدادات موحدة للمطياف (لضمان المقارنة العادلة)
SR = 22050
FMAX = 8000     # توحيد الحد الأقصى للتردد
N_MELS = 128
VMIN_DB = -60   # توحيد الحد الأدنى لشدة اللون
VMAX_DB = 0     # توحيد الحد الأقصى لشدة اللون

for i, name in enumerate(model_order):
    path = audio_files[name]
    row = i // 3
    col = i % 3

    # تحميل الصوت
    y, sr = librosa.load(path, sr=SR)
    
    # حساب Mel Spectrogram وتحويله إلى dB
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=N_MELS)
    S_dB = librosa.power_to_db(S, ref=np.max)

    # الرسم وتوحيد المدى اللوني (VMIN/VMAX)
    img = librosa.display.specshow(S_dB, x_axis='time', y_axis='mel', sr=sr, fmax=FMAX, 
                                   ax=axes[row, col], cmap='magma', vmin=VMIN_DB, vmax=VMAX_DB)

    axes[row, col].set(title=name)
    axes[row, col].label_outer()
    
# إزالة المحور الفرعي السادس الفارغ (لأن لدينا 5 رسوم فقط)
fig.delaxes(axes[1, 2]) 

# إضافة شريط ألوان موحد (Color Bar)
# يتم وضعه خارج التخطيط الشبكي ليكون مفتاحًا لجميع الرسوم
cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7]) # [left, bottom, width, height]
fig.colorbar(img, cax=cbar_ax, format="%+2.f dB")

plt.tight_layout(rect=[0, 0, 0.9, 1.0])
plt.savefig('Figure_7_2_MelSpectrogram_Comparison_Enhanced.png', dpi=300)
# plt.show()
plt.close()
print("Figure 7.2 (Mel Spectrogram) code executed successfully.")