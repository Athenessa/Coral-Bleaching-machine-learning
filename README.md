# A global coral bleaching-machine learning
This project is about machine learning 
ปัจจัยด้านความร้อนและสภาพแวดล้อมของแนวปะการังใช้ทำนายระดับการฟอกขาวได้แค่ไหน และนอกจากความร้อนแล้ว อะไรช่วยให้ปะการังรอด

Notebook นี้ทำ: โหลดข้อมูล → ตรวจคุณภาพ → clean → ตรวจ target → เลือก feature → export ไฟล์สะอาด + data dictionary ให้ส่วน modeling

ข้อมูล: van Woesik & Kratochwill (2022) A global coral-bleaching database, 1980–2020, Scientific Data 9:20. BCO-DMO dataset 773466 → ไฟล์ global_bleaching_environmental.csv (~16 MB)

หลักการ: ไม่แก้ raw data · ตรวจก่อนแก้ · ค่าผิดปกติให้ flag ก่อน ไม่ลบอัตโนมัติ · ไม่ impute ที่นี่ (ให้ทำใน Pipeline หลัง train/test split เพื่อกัน leakage)

## Decision Tree
### SEA
- Accuracy : 0.74
- F1-score : 0.61
- Precision : 0.60
- Recall : 0.61

### GLOBAL
- Accuracy : 0.82
- F1-score : 0.84
- Precision : 0.85
- Recall : 0.84

## Random Forest
### SEA
- Accuracy : 0.81
- F1-score : 0.69
- Precision : 0.72
- Recall : 0.67

### GLOBAL
- Accuracy : 0.86
- F1-score : 0.88
- Precision : 0.90
- Recall : 0.87
