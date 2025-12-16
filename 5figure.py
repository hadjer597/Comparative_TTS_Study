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

# تحديد الترتيب (لضمان تطابق الألوان والتسميات)
model_order = list(audio_files.keys())

# =========================================================================
# 2. Generating Figure 7.1: Integrated Waveform Plot (Enhanced)
# =========================================================================

# 5 صفوف وعمود واحد (5 rows, 1 column)
fig, axes = plt.subplots(nrows=5, ncols=1, figsize=(10, 10), sharex=True)
plt.suptitle('Figure 7.1: Comparison of Waveform (Zoomed In) across the Five Techniques', 
             fontsize=14, y=1.02, fontweight='bold')

# نطاق التركيز الزمني للموجة (Zoom In) - الإجراء الحاسم
TIME_START = 0.5  # بداية النطاق بالثواني
TIME_END = 1.5    # نهاية النطاق بالثواني

for i, name in enumerate(model_order):
    path = audio_files[name]
    
    # 1. تحميل ملف الصوت
    y, sr = librosa.load(path, sr=None)
    
    # 2. تحديد نطاق الرسم (التركيز)
    start_sample = librosa.time_to_samples(TIME_START, sr=sr)
    end_sample = librosa.time_to_samples(TIME_END, sr=sr)
    y_zoomed = y[start_sample:end_sample]

    # 3. إنشاء محور زمن جديد للجزء المُركَّز
    time_zoomed = librosa.samples_to_time(np.arange(len(y_zoomed)), sr=sr)
    
    # 4. الرسم (استخدام المحور الزمني المُركَّز)
    axes[i].plot(time_zoomed + TIME_START, y_zoomed, color='darkblue', linewidth=0.8)
    
    # 5. التوحيد والتخصيص
    axes[i].set_ylim(-1.0, 1.0) # توحيد المحور Y لجميع الرسوم
    axes[i].set_title(f'{name}', fontsize=10, loc='left', pad=-10)
    axes[i].set_ylabel('Amplitude', fontsize=8)
    axes[i].tick_params(axis='both', which='major', labelsize=7)
    
    # إزالة تسمية المحور X من الرسوم العلوية
    if i < 4:
        axes[i].set_xlabel('')
    else:
        axes[i].set_xlabel('Time (s)')

# ضبط التخطيط وتصدير الصورة بدقة عالية
plt.tight_layout(rect=[0, 0, 1, 1.0])
plt.savefig('Figure_7_1_Waveform_Comparison_Enhanced.png', dpi=300)
# plt.show() # ألغِ التعليق لعرض الصورة فوراً
plt.close()

print("Figure 7.1 code executed successfully. Check the output file: Figure_7_1_Waveform_Comparison_Enhanced.png")