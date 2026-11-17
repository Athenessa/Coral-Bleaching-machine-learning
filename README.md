# A global coral bleaching-machine learning
This project is about machine learning 
ปัจจัยด้านความร้อนและสภาพแวดล้อมของแนวปะการังใช้ทำนายระดับการฟอกขาวได้แค่ไหน และนอกจากความร้อนแล้ว อะไรช่วยให้ปะการังรอด

Notebook นี้ทำ: โหลดข้อมูล → ตรวจคุณภาพ → clean → ตรวจ target → เลือก feature → export ไฟล์สะอาด + data dictionary ให้ส่วน modeling

ข้อมูล: van Woesik & Kratochwill (2022) A global coral-bleaching database, 1980–2020, Scientific Data 9:20. BCO-DMO dataset 773466 → ไฟล์ global_bleaching_environmental.csv (~16 MB)

หลักการ: ไม่แก้ raw data · ตรวจก่อนแก้ · ค่าผิดปกติให้ flag ก่อน ไม่ลบอัตโนมัติ · ไม่ impute ที่นี่ (ให้ทำใน Pipeline หลัง train/test split เพื่อกัน leakage)

## Decision Tree
### SEA

            precision    recall  f1-score   support

        mild       0.52      0.52      0.52       233
        none       0.81      0.80      0.81       529
     severe       0.29      0.43      0.35        23
    accuracy                           0.71       785
    
   macro avg       0.54      0.58      0.56       785
weighted avg       0.71      0.71      0.71       785


### GLOBAL
               precision    recall  f1-score   support

        mild       0.75      0.73      0.74      1573
        none       0.78      0.80      0.79      1481
      severe       0.66      0.64      0.65       447

    accuracy                           0.75      3501
   macro avg       0.73      0.72      0.73      3501
weighted avg       0.75      0.75      0.75      3501

Accuracy: 0.7492
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

