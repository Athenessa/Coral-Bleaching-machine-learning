# 🪸 Coral Bleaching Prediction Using Machine Learning naka

**เปรียบเทียบโมเดลที่ train ด้วยข้อมูลทั่วโลก (Global) กับข้อมูลเฉพาะ Southeast Asia (SEA) เพื่อศึกษาปัจจัยที่เกี่ยวข้องกับการฟอกขาวของปะการัง**

## Environmental Problem

Coral bleaching เป็นหนึ่งในปัญหาสำคัญที่ส่งผลกระทบต่อแนวปะการัง เมื่อปะการังเผชิญกับความเครียดจากสภาพแวดล้อม โดยเฉพาะอุณหภูมิน้ำทะเลที่สูงผิดปกติ ปะการังอาจสูญเสียสาหร่ายที่อาศัยอยู่ในเนื้อเยื่อ ทำให้ปะการังเปลี่ยนเป็นสีขาว และถ้าความเครียดเกิดขึ้นต่อเนื่องเป็นเวลานานก็อาจนำไปสู่การตายของปะการังได้

ความร้อนเป็นปัจจัยสำคัญที่ทำให้เกิด bleaching แต่ไม่ได้อธิบายผลลัพธ์ของทุกพื้นที่ได้ทั้งหมด เพราะบางแนวปะการังอาจมีระดับการฟอกขาวต่างกัน แม้จะเผชิญกับสภาพความร้อนใกล้เคียงกัน ปัจจัยอื่น ๆ เช่น ความลึก ความขุ่นของน้ำ การเกิดพายุ ลักษณะพื้นที่ และตำแหน่งทางภูมิศาสตร์ ก็อาจมีส่วนเกี่ยวข้องด้วย

โปรเจกต์นี้จึงนำ Machine Learning มาใช้ศึกษาว่าปัจจัยด้านสิ่งแวดล้อมและสภาพพื้นที่สามารถช่วยทำนายระดับการฟอกขาวของปะการังได้มากแค่ไหน และเมื่อนำโมเดลที่ train จากข้อมูลทั่วโลกมาเทียบกับโมเดลที่ train เฉพาะ SEA ผลลัพธ์จะแตกต่างกันอย่างไร

## Dataset Source

ใช้ข้อมูลจาก van Woesik and Kratochwill (2022), *A global coral-bleaching database, 1980–2020*

- **Journal:** Scientific Data, 9, 20 (2022)
- **Original dataset:** [BCO-DMO Dataset 773466](https://www.bco-dmo.org/dataset/773466)
- **Study period:** 1980–2020
- **Cleaned datasets:** `data/coral_clean_global.csv` และ `data/coral_clean_sea.csv`
- **Data dictionary:** [`data/data_dictionary.csv`](data/data_dictionary.csv)

ข้อมูลประกอบด้วยบันทึกการฟอกขาวของปะการังและตัวแปรด้านสิ่งแวดล้อม เวลา และตำแหน่งทางภูมิศาสตร์ ตัวอย่าง feature ที่ใช้ ได้แก่ Accumulated heat stress (DHW), Sea Surface Temperature, Temperature, Depth, Turbidity, Cyclone Frequency, Wind Speed, Exposure, พิกัด และข้อมูลภูมิภาค

สำหรับ Classification models ใช้ `bleach_class` เป็น target โดยแบ่งเป็น 3 classes ดังนี้

| Class | Definition |
|---|---|
| `none` | 0% bleaching |
| `mild` | มากกว่า 0% และไม่เกิน 30% |
| `severe` | มากกว่า 30% |

## Research Questions

**คำถามหลักของโปรเจกต์:** เทียบโมเดลที่ train ทั้งโลกกับโมเดลที่ train เฉพาะ SEA แล้วดูว่านอกจากความร้อนแล้วมีปัจจัยอะไรที่เกี่ยวข้องกับการรอดของปะการัง

คำถามย่อยที่ใช้ในการศึกษา ได้แก่

1. โมเดลที่ train ด้วยข้อมูล Global จะทำนายข้อมูล SEA ได้ดีกว่าโมเดลที่ train เฉพาะ SEA หรือไม่?
2. นอกจาก Heat stress แล้ว Feature ด้านสิ่งแวดล้อมอะไรบ้างที่ช่วยอธิบายหรือทำนายระดับการฟอกขาว?
3. Machine Learning model แบบใดให้ผลการทำนายดีที่สุด?
4. Regression และ K-Nearest Neighbors (KNN) ให้ผลเป็นอย่างไรเมื่อเทียบกับโมเดลอื่นในโปรเจกต์?

## Methods

### Data Preparation

ใน repository มีทั้งข้อมูล Global และ SEA ที่ผ่านการ clean มาแล้ว รวมถึง cleaning script และ data dictionary ก่อนนำข้อมูลไป train model ต้องระวังเรื่องข้อมูลซ้ำจาก site เดิมที่ถูกสำรวจหลายครั้ง และ class imbalance ของ target ด้วย

แนวทางที่ใช้ใน modeling workflow ที่มีรายงานผลแล้ว ได้แก่

- **Group-based split:** แบ่ง train/test ตาม `site_id` เพื่อป้องกันไม่ให้ข้อมูลจาก reef site เดียวกันไปอยู่ทั้งใน train และ test set
- **Pipeline:** ทำ Imputation, Encoding และ Scaling ภายใน Pipeline หลังจาก split ข้อมูลแล้ว เพื่อลดปัญหา Data Leakage
- **Evaluation metrics:** ใช้ Macro-F1, Balanced Accuracy และ F1-score ราย class ร่วมกับ Accuracy เพราะข้อมูล `severe` มีจำนวนน้อย ถ้าดู Accuracy อย่างเดียวอาจทำให้เข้าใจผิดว่าโมเดลทำได้ดี
- **Feature correlation:** ตรวจสอบ feature ที่มีความสัมพันธ์กัน เช่น ตัวแปร Heat stress ที่อาจให้ข้อมูลซ้ำซ้อนกัน ก่อนนำ Feature Importance ไปตีความ

### Machine Learning Models

โมเดลที่มีผลการทดลองอยู่ใน repository แล้ว ได้แก่

- Decision Tree
- Random Forest
- XGBoost
- LightGBM
- Support Vector Machine (SVM)
- Neural Network (MLP)

ส่วน Regression และ KNN เป็นส่วนที่กำลังดำเนินการเพิ่มเติม แต่ยังไม่มีผลการประเมินฉบับสมบูรณ์ใน repository จึงยังไม่นำมารวมในตารางเปรียบเทียบผลด้านล่าง

### Model Evaluation

สำหรับ Classification models จะทำนาย `bleach_class` ทั้ง 3 classes คือ `none`, `mild` และ `severe` โดยใช้ Accuracy, Macro-F1, Balanced Accuracy และ F1-score ของแต่ละ class ในการประเมิน

Workflow ล่าสุดของ XGBoost, LightGBM, SVM และ Neural Network ใช้การแบ่งข้อมูลตาม site และ GroupKFold สำหรับ model selection นอกจากนี้ยังมีการเปรียบเทียบ Global-trained model กับ SEA-only model โดยประเมินผลบน SEA test subset เหมือนกัน

## Main Results

### Classification Model Performance

ตารางนี้เป็นผลที่รายงานไว้ใน `MODEL_RESULTS.md` จากการทดสอบแบบแบ่งข้อมูลตาม reef site

| Model | Macro-F1 | Balanced Accuracy | Accuracy | F1-score (severe) |
|---|---:|---:|---:|---:|
| Majority-class baseline | 0.230 | 0.333 | 0.526 | 0.000 |
| **XGBoost** | **0.746** | **0.751** | 0.769 | **0.705** |
| LightGBM | 0.744 | 0.746 | 0.769 | 0.697 |
| SVM (RBF) | 0.701 | 0.719 | 0.743 | 0.614 |
| Neural Network (MLP) | 0.670 | 0.643 | 0.724 | 0.562 |

ส่วนผลจาก Notebook รุ่นก่อนหน้าที่ทดลอง Decision Tree และ Random Forest มีดังนี้

| Training region | Model | Accuracy | Macro-F1 | F1-score (severe) |
|---|---|---:|---:|---:|
| Global | Decision Tree | 0.7492 | 0.73 | 0.65 |
| Global | Random Forest | 0.7981 | 0.77 | 0.69 |
| SEA | Decision Tree | 0.7057 | 0.56 | 0.35 |
| SEA | Random Forest | 0.7669 | 0.64 | 0.49 |

**หมายเหตุ:** ผลทั้งสองตารางมาจากคนละ workflow จึงไม่ควรนำคะแนนมาเทียบกันตรง ๆ เพราะ Notebook ของ Decision Tree และ Random Forest ใช้ random train/test split หลังจาก drop missing values ส่วน workflow ล่าสุดใช้การ split ตาม reef site ซึ่งช่วยลดโอกาสที่ข้อมูลจาก site เดียวกันจะรั่วไหลไปอยู่ทั้งสองชุด

### Global vs. Southeast Asia (SEA)

เปรียบเทียบ Macro-F1 ของโมเดลทั้งสองแบบบน SEA test subset ได้ผลดังนี้

| Training data | Macro-F1 on SEA test |
|---|---:|
| **Global data** | **0.691** |
| SEA data only | 0.666 |

จากผลการทดลองครั้งนี้ โมเดลที่ train ด้วยข้อมูล Global ทำคะแนนบน SEA test ได้สูงกว่าโมเดลที่ train เฉพาะ SEA เล็กน้อย อาจเป็นเพราะข้อมูลจากภูมิภาคอื่นช่วยให้โมเดลเห็นตัวอย่างของ severe bleaching มากขึ้นด้วย

อย่างไรก็ตาม ใน SEA test มี severe cases เพียง 24 จาก 849 observations หรือประมาณ 2.8% เท่านั้น ดังนั้นผลต่างของคะแนนนี้ยังมีความไม่แน่นอนสูง และยังสรุปไม่ได้ว่า Global model จะดีกว่า SEA model ในทุกกรณี

## Best Model

จาก Classification results ที่มีอยู่ในปัจจุบัน **XGBoost เป็นโมเดลที่ให้ผลดีที่สุด** ในกลุ่มที่ใช้ site-based evaluation โดยได้ Macro-F1 เท่ากับ **0.746** และ F1-score ของ `severe` เท่ากับ **0.705** ส่วน LightGBM ได้ Macro-F1 เท่ากับ 0.744 ซึ่งมีผลใกล้เคียงกันมาก

ข้อสรุปนี้จำกัดอยู่เฉพาะ Classification models ที่มีผลรายงานแล้วเท่านั้น เมื่อ Regression และ KNN มีผลการทดลองครบถ้วนแล้ว ควรนำมาเปรียบเทียบอีกครั้งด้วยวิธีแบ่งข้อมูลและ metric ที่เหมาะสมก่อนสรุป Best Model ของทั้งโปรเจกต์

## Feature Importance

ใช้ **Permutation Importance** บน XGBoost test set เพื่อดูว่าเมื่อสลับค่าของ feature ทีละตัวแล้ว Macro-F1 ลดลงมากแค่ไหน และใช้ **Partial Dependence** เพื่อดูแนวโน้มความสัมพันธ์ระหว่าง feature กับความน่าจะเป็นที่โมเดลทำนายว่าเป็น severe bleaching

Feature ที่มีประเด็นน่าสนใจจากผลที่รายงานไว้ใน `MODEL_RESULTS.md` ได้แก่

| Feature | สิ่งที่โมเดลพบ | Environmental interpretation |
|---|---|---|
| `date_year` | ความน่าจะเป็นของ severe bleaching เพิ่มขึ้นในช่วงปี 2014–2016 | ช่วงเวลาดังกล่าวตรงกับ Global Coral Bleaching Event ครั้งที่ 3 โดย Year อาจสะท้อนเหตุการณ์ระดับโลกที่ตัวแปรความร้อนเฉพาะพื้นที่อธิบายได้ไม่หมด |
| `longitude`, `latitude` | ผลการทำนายแตกต่างกันไปตามตำแหน่งทางภูมิศาสตร์ | แต่ละพื้นที่อาจมีสภาพแวดล้อมและการได้รับความร้อนต่างกัน ทั้งนี้อาจมีผลจากความแตกต่างของการสำรวจด้วย |
| `tsa_dhw` | เมื่อ Accumulated heat stress เพิ่มขึ้น ความน่าจะเป็นที่โมเดลทำนายว่า severe ก็เพิ่มขึ้น ก่อนที่แนวโน้มจะเริ่มคงที่ | เป็น Environmental feature ที่สำคัญที่สุดในผลการวิเคราะห์นี้ แสดงให้เห็นความสำคัญของความร้อนสะสม |
| `depth_m` | จุดสำรวจในน้ำตื้นมีโอกาสถูกทำนายเป็น severe สูงกว่าจุดสำรวจที่ลึกกว่า | ความลึกอาจเกี่ยวข้องกับปริมาณแสงและการได้รับความร้อนที่แตกต่างกัน |
| `cyclone_frequency` | พื้นที่ที่มี Cyclone frequency สูงมีความน่าจะเป็นของ severe ลดลงเล็กน้อย | การกวนผสมน้ำอาจช่วยลดความร้อนได้ในบางสถานการณ์ แต่ผลในโมเดลนี้มีขนาดเล็ก |
| `ssta_frequency` | เมื่อเกิดสัปดาห์ที่มีอุณหภูมิสูงผิดปกติบ่อยขึ้น ความเสี่ยงที่โมเดลทำนายก็เพิ่มขึ้น | การเผชิญความร้อนซ้ำ ๆ อาจทำให้ปะการังมีเวลาฟื้นตัวน้อยลง |
| `turbidity` | โมเดลแสดงแนวโน้มเล็กน้อยว่าพื้นที่น้ำขุ่นมีความเสี่ยงที่ทำนายต่ำลง | ความขุ่นอาจช่วยลดแสงที่ลงไปถึงปะการังได้ แต่ผลที่พบยังค่อนข้างน้อย |
| `exposure` | กลุ่ม Exposed และ Sheltered มีผลการทำนายใกล้เคียงกัน | ในการวิเคราะห์นี้ยังไม่พบความแตกต่างที่ชัดเจนจาก feature นี้ |

นอกจากนี้ยังมีการทำ **Ablation Analysis** โดยลองตัด feature บางกลุ่มออก แล้วดูว่า model performance เปลี่ยนไปอย่างไร

| Feature set | Macro-F1 | F1-score (severe) |
|---|---:|---:|
| All 21 features | 0.746 | 0.705 |
| Without year | 0.719 | 0.659 |
| Environmental features only (ตัด year, coordinates และ ocean realm ออก) | 0.702 | 0.633 |

ผลนี้แสดงให้เห็นว่าแม้จะใช้เฉพาะ Environmental features โมเดลก็ยังมีความสามารถในการทำนายอยู่ ส่วนข้อมูลด้านเวลาและภูมิศาสตร์ช่วยเพิ่มประสิทธิภาพให้โมเดลได้อีกระดับหนึ่ง

## Environmental Interpretation

จากผลที่มีอยู่ ความร้อนสะสมยังเป็นปัจจัยสำคัญที่เกี่ยวข้องกับการฟอกขาวของปะการัง แต่ก็ยังมีปัจจัยอื่นที่ช่วยอธิบายความแตกต่างของระดับการฟอกขาวได้ด้วย

- **Accumulated heat stress:** `tsa_dhw` เป็น Environmental feature ที่สำคัญที่สุดในผล XGBoost ที่รายงาน เมื่อ DHW เพิ่มขึ้น โมเดลก็ทำนายโอกาสเกิด severe bleaching สูงขึ้น
- **Depth:** จุดสำรวจที่ลึกกว่ามีโอกาสถูกทำนายเป็น severe ต่ำกว่าจุดสำรวจน้ำตื้น อาจเป็นไปได้ว่าแนวปะการังในระดับความลึกต่างกันได้รับแสงและความร้อนไม่เท่ากัน แต่ผลนี้ยังไม่ได้ยืนยันว่าความลึกเป็นสาเหตุโดยตรง
- **Time and location:** ช่วงปี 2014–2016 มีความเสี่ยงที่โมเดลทำนายสูงขึ้น แสดงให้เห็นว่าบริบทของเหตุการณ์ระดับโลกและความแตกต่างระหว่างพื้นที่อาจมีส่วนสำคัญ นอกเหนือจากค่าความร้อนเฉพาะจุด
- **Cyclone and turbidity:** ทั้งสอง feature มีแนวโน้มของผลที่ค่อนข้างเล็ก จึงยังไม่ควรสรุปว่าปัจจัยเหล่านี้ช่วยป้องกันการฟอกขาวได้โดยตรง แต่สามารถนำไปศึกษาเพิ่มเติมได้

โดยรวมแล้ว ผลการทดลองชี้ให้เห็นว่าการฟอกขาวของปะการังไม่ได้เกี่ยวข้องกับอุณหภูมิเพียงอย่างเดียว แต่ยังมีปัจจัยด้านสิ่งแวดล้อม เวลา และตำแหน่งทางภูมิศาสตร์เข้ามาเกี่ยวข้องด้วย อย่างไรก็ตาม ผลจาก Machine Learning แสดงความสัมพันธ์ที่ช่วยในการทำนาย ไม่ใช่หลักฐานยืนยันเหตุและผล จึงควรใช้เป็นข้อมูลประกอบการศึกษาและการวางแผนติดตามแนวปะการัง ไม่ใช่ใช้ตัดสินใจเรื่องพื้นที่ฟื้นฟูหรือการปิดแหล่งท่องเที่ยวเพียงอย่างเดียว

## Limitations

- จำนวน `severe` observations มีค่อนข้างน้อย โดยเฉพาะใน SEA test subset
- ผลจาก Feature Importance และ Partial Dependence แสดงความสัมพันธ์ที่โมเดลเรียนรู้ ไม่ได้ยืนยันเหตุและผล
- `date_year` และพิกัดอาจสะท้อนความแตกต่างของการเก็บข้อมูลหรือโปรแกรมสำรวจในแต่ละพื้นที่ด้วย
- Heat stress จาก Satellite data อาจไม่ตรงกับอุณหภูมิที่ปะการังแต่ละ colony ได้รับจริง
- Feature บางตัวมีความสัมพันธ์กัน ซึ่งอาจส่งผลต่อการแบ่ง Feature Importance
- Notebook แต่ละชุดใช้วิธี train/test split ต่างกัน จึงไม่ควรเปรียบเทียบคะแนนข้าม workflow โดยตรง
- ยังไม่มีผล Regression และ KNN ที่สรุปไว้ใน repository

## Repository Structure

| File / Folder | Description |
|---|---|
| `data/` | Cleaned Global และ SEA datasets, cleaning script และ data dictionary |
| `NOTE/` | Notebook สำหรับ Decision Tree และ Random Forest ทั้ง Global และ SEA |
| `Boosting_SVM_NN.ipynb` | Notebook สำหรับ XGBoost, LightGBM, SVM, Neural Network และ model interpretation |
| `MODEL_RESULTS.md` | รายละเอียดผลการทดลอง การเปรียบเทียบโมเดล และการตีความผล |

## References

van Woesik, R., & Kratochwill, C. (2022). A global coral-bleaching database, 1980–2020. *Scientific Data, 9*, 20. [https://doi.org/10.1038/s41597-021-01101-8](https://doi.org/10.1038/s41597-021-01101-8)

BCO-DMO. *A global coral-bleaching database, 1980–2020* (Dataset 773466). [https://www.bco-dmo.org/dataset/773466](https://www.bco-dmo.org/dataset/773466)
