# A global coral bleaching-machine learning
This project is about machine learning 
ปัจจัยด้านความร้อนและสภาพแวดล้อมของแนวปะการังใช้ทำนายระดับการฟอกขาวได้แค่ไหน และนอกจากความร้อนแล้ว อะไรช่วยให้ปะการังรอด

Notebook นี้ทำ: โหลดข้อมูล → ตรวจคุณภาพ → clean → ตรวจ target → เลือก feature → export ไฟล์สะอาด + data dictionary ให้ส่วน modeling

ข้อมูล: van Woesik & Kratochwill (2022) A global coral-bleaching database, 1980–2020, Scientific Data 9:20. BCO-DMO dataset 773466 → ไฟล์ global_bleaching_environmental.csv (~16 MB)

หลักการ: ไม่แก้ raw data · ตรวจก่อนแก้ · ค่าผิดปกติให้ flag ก่อน ไม่ลบอัตโนมัติ · ไม่ impute ที่นี่ (ให้ทำใน Pipeline หลัง train/test split เพื่อกัน leakage)

## Decision Tree
### SEA
- Accuracy: 0.7057 (70.57%)   
- Macro Avg (Precision / Recall / F1): 0.54 / 0.58 / 0.56   
- Severe Class F1-Score: 0.35 

### GLOBAL
- Accuracy: 0.7492 (74.92%)
- Macro Avg (Precision / Recall / F1): 0.73 / 0.72 / 0.73
- Severe Class F1-Score: 0.65 

## Random Forest
### SEA
- Accuracy: 0.7669 (76.69%)
- Macro Avg (Precision / Recall / F1): 0.65 / 0.64 / 0.64
- Severe Class F1-Score: 0.49

### GLOBAL
- Accuracy: 0.7981 (79.81%)
- Macro Avg (Precision / Recall / F1): 0.79 / 0.76 / 0.77
- Severe Class F1-Score: 0.69

