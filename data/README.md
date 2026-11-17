This folder contains DATA
สำหรับทีม modeling ⚠️
1. Split ตาม site_id เพราะ site เดียวกันถูกสำรวจหลายครั้ง ถ้าสุ่มแบ่งธรรมดาข้อมูลจะรั่ว ให้ใช้ GroupShuffleSplit / GroupKFold หรือแบ่งตามปี
2.Impute + encode ใน Pipeline หลัง split เท่านั้น
3.Target เบ้มาก ค่า 0 เยอะ ถ้าทำ regression ลองใช้ log1p ถ้าทำ classification ใช้ F1 / ROC-AUC และ class_weight="balanced"
4.data_source วัด target ไม่เหมือนกัน บางแหล่งเป็นค่าจากรหัสระดับการฟอกขาว (มีค่าไม่ซ้ำแค่ไม่กี่ค่า) ให้ลองใส่เป็น feature หรือกรองแหล่งนั้นออกแล้วเทียบผล
5.Feature ซ้ำซ้อน เช่น tsa_dhw กับ ssta_dhw ดูได้ใน figures/feature_correlation.png ซ้ำซ้อนมากเกินไปจะทำให้ feature importance ถูกแบ่งกระจาย
คำถามของโปรเจกต์: เทียบโมเดลที่เทรนทั้งโลกกับโมเดลที่เทรนเฉพาะ SEA แล้วดูว่านอกจากความร้อนแล้วมีปัจจัยอะไรช่วยให้ปะการังรอด
