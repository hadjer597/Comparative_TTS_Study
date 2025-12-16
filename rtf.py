import matplotlib.pyplot as plt
import numpy as np

# =========================================================================
# 1. SETUP: RTF Data (القيم النهائية الصحيحة والمرتبة للمقارنة)
# =========================================================================

# البيانات الكمية المحدثة لـ RTF (مرتبة من الأسرع للأبطأ)
rtf_data = {
    'Concatenative': 0.03,
    'Parametric': 0.12,
    'VITS': 0.90,       # النموذج العصبي الوحيد < 1.0
    'FastPitch': 1.30,  
    'Tacotron 2': 1.31   
}

techniques = list(rtf_data.keys())
rtf_values = list(rtf_data.values())
x_pos = np.arange(len(techniques))

# =========================================================================
# 2. Figure 7.3: Comparative Bar Chart for RTF (الرسم البياني الشريطي لـ RTF)
# =========================================================================

# تحديد الألوان: أخضر لما هو أسرع من الوقت الحقيقي (< 1.0)، أحمر لما هو أبطأ (> 1.0)
colors = ['green' if v < 1.0 else 'red' for v in rtf_values]

plt.figure(figsize=(10, 6))
bars = plt.bar(x_pos, rtf_values, color=colors, alpha=0.8)

# تسليط الضوء على قيم Non-Real-Time (Tacotron 2 و FastPitch) بخطوط مائلة
for i, bar in enumerate(bars):
    if rtf_values[i] > 1.0:
        bar.set_hatch('//') 

# إضافة خط الوقت الحقيقي المرجعي عند RTF = 1.0
plt.axhline(y=1.0, color='darkorange', linestyle='--', linewidth=2.0, 
            label='Real-Time Threshold (RTF=1.0)')

# تخصيص الرسم البياني
plt.xticks(x_pos, techniques, rotation=0)
plt.ylabel('Real-Time Factor (RTF)', fontweight='bold', fontsize=12)
plt.title('Figure 7.3: Comparison of Generation Efficiency (RTF) across Techniques', 
          fontsize=14, pad=15, fontweight='bold')
plt.legend(loc='upper right')
plt.grid(axis='y', linestyle=':', alpha=0.6)

# إضافة قيم RTF فوق كل شريط لتسهيل القراءة الأكاديمية
for i, v in enumerate(rtf_values):
    # تنسيق القيمة (2 رقم عشري)
    plt.text(x_pos[i], v + 0.05, f'{v:.2f}', ha='center', fontweight='bold', fontsize=10)

plt.ylim(0, 1.5) # تحديد المحور Y لتركيز المقارنة حول 1.0
plt.tight_layout()
plt.savefig('Figure_7_3_RTF_BarChart_Enhanced.png', dpi=300)
# plt.show()
plt.close()
print("Figure 7.3 (RTF Bar Chart) code executed successfully. Check your directory for the output image.")