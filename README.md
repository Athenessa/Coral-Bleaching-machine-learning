# 🪸 Coral Bleaching Prediction Using Machine Learning

**ทำนายระดับการฟอกขาวของปะการัง (none / mild / severe) จากความร้อนสะสม สภาพแนวปะการัง ตำแหน่ง และเวลา แล้วเปรียบเทียบโมเดลที่ train ด้วยข้อมูลทั่วโลก (Global) กับข้อมูลเฉพาะ Southeast Asia (SEA)**

> 2107541 Machine Learning for Environmental Applications · Project 2

## Environmental Problem

Coral bleaching เป็นหนึ่งในปัญหาสำคัญที่ส่งผลกระทบต่อแนวปะการัง เมื่อปะการังเผชิญกับความเครียดจากสภาพแวดล้อม โดยเฉพาะอุณหภูมิน้ำทะเลที่สูงผิดปกติ ปะการังจะขับสาหร่ายซูแซนเทลลี (zooxanthellae) ที่อาศัยอยู่ในเนื้อเยื่อออกไป ทำให้ปะการังเปลี่ยนเป็นสีขาว และถ้าความเครียดเกิดขึ้นต่อเนื่องเป็นเวลานานก็อาจนำไปสู่การตายของปะการังได้

ความร้อนเป็นปัจจัยหลักที่ทำให้เกิด bleaching แต่ไม่ได้อธิบายผลลัพธ์ของทุกพื้นที่ได้ทั้งหมด เพราะบางแนวปะการังฟอกขาวต่างกัน แม้จะเผชิญกับความร้อนใกล้เคียงกัน ปัจจัยอื่น เช่น ความลึก ความขุ่นของน้ำ การเกิดพายุ การเปิดรับคลื่น และตำแหน่งทางภูมิศาสตร์ ก็อาจมีส่วนเกี่ยวข้องด้วย

โปรเจกต์นี้ใช้ Machine Learning ศึกษาว่าปัจจัยด้านสิ่งแวดล้อมและสภาพพื้นที่ช่วยทำนายระดับการฟอกขาวได้มากแค่ไหน ปัจจัยใดสำคัญรองจากความร้อน และโมเดลที่ train จากข้อมูลทั่วโลกใช้กับแนวปะการังใน SEA ได้ดีกว่าโมเดลที่ train เฉพาะ SEA หรือไม่ ผลที่ได้นำไปใช้ประกอบการเลือกพื้นที่ฟื้นฟูปะการังและการวางแผนเตือนภัยแหล่งดำน้ำได้

## Dataset Source

ใช้ข้อมูลจาก van Woesik and Kratochwill (2022), *A global coral-bleaching database, 1980–2020*

- **Journal:** Scientific Data, 9, 20 (2022)
- **Original dataset:** [BCO-DMO Dataset 773466](https://www.bco-dmo.org/dataset/773466)
- **Study period:** 1980–2020
- **Cleaned datasets:** `data/coral_clean_global.csv` และ `data/coral_clean_sea.csv`
- **Cleaning script:** `data/clean_data.py`
- **Data dictionary:** [`data/data_dictionary.csv`](data/data_dictionary.csv)

| Dataset | Rows (survey events) | Sites | Years | none | mild | severe |
|---|---:|---:|---|---:|---:|---:|
| Global (`coral_clean_global.csv`) | 23,271 | 11,068 | 1980–2020 | 9,205 (39.6%) | 10,023 (43.1%) | 4,043 (17.4%) |
| SEA (`coral_clean_sea.csv`) | 4,699 | 1,832 | 1983–2020 | 3,131 (66.6%) | 1,309 (27.9%) | 259 (5.5%) |

ข้อมูล SEA มาจาก 10 ประเทศ ได้แก่ Malaysia, Indonesia, Philippines, Thailand, Vietnam, Cambodia, Brunei, Myanmar, East Timor และ Singapore ข้อมูลทั้งหมดมาจาก 9 แหล่งสำรวจ (Reef_Check, Donner, AGRRA, FRRP, Kumagai, McClanahan, Safaie, Nuryana, Setiawan) ซึ่งวัดการฟอกขาวด้วยวิธีต่างกัน

**Features (21 ตัว)** ตาม `data_dictionary.csv`

| กลุ่ม | Features |
|---|---|
| Heat stress | `tsa_dhw`, `ssta_dhw`, `tsa`, `ssta`, `ssta_frequency`, `tsa_frequency`, `climsst`, `temperature_c`, `temperature_maximum` |
| Reef / site condition | `depth_m`, `turbidity`, `distance_to_shore`, `exposure`, `cyclone_frequency`, `windspeed` |
| Location | `latitude_degrees`, `longitude_degrees`, `ocean_name`, `realm_name` |
| Time | `date_year`, `date_month` |

**Target:** `bleach_class` (classification 3 classes) สร้างจาก `percent_bleaching`

| Class | Definition |
|---|---|
| `none` | 0% bleaching |
| `mild` | มากกว่า 0% และไม่เกิน 30% |
| `severe` | มากกว่า 30% |

ในไฟล์ยังมี `percent_bleaching` (regression) และ `bleached` (binary 0/1) ให้ใช้ได้ แต่ผลที่รายงานใน README นี้ใช้ `bleach_class` ทั้งหมด

## Research Questions

**คำถามหลัก:** เทียบโมเดลที่ train ทั้งโลกกับโมเดลที่ train เฉพาะ SEA แล้วดูว่านอกจากความร้อนแล้วมีปัจจัยอะไรเกี่ยวข้องกับระดับการฟอกขาวของปะการัง

1. Machine Learning model แบบใดทำนายระดับการฟอกขาวในแนวปะการังที่โมเดลไม่เคยเห็นได้ดีที่สุด?
2. นอกจาก heat stress แล้ว feature ด้านสิ่งแวดล้อมใดช่วยอธิบายหรือทำนายระดับการฟอกขาว และมีทิศทางอย่างไร?
3. ถ้าตัดข้อมูลเวลาและตำแหน่งออก เหลือแต่ตัวแปรสิ่งแวดล้อม โมเดลยังทำนายได้หรือไม่?
4. โมเดลที่ train ด้วยข้อมูล Global ทำนายแนวปะการังใน SEA ได้ดีกว่าโมเดลที่ train เฉพาะ SEA หรือไม่?

## Methods

### Data Preparation (`data/clean_data.py`)

- อ่านค่า `nd` เป็น missing, แปลงชื่อคอลัมน์และชนิดข้อมูล, แปลงอุณหภูมิจาก Kelvin เป็น °C
- ค่าที่อยู่นอกช่วงทางกายภาพ (เช่น `percent_bleaching` นอก 0–100, `depth_m` นอก 0–100 m) เปลี่ยนเป็น NaN โดยไม่ลบทั้งแถว
- ลบแถวซ้ำและ `sample_id` ซ้ำ, ลบแถวที่ไม่มี target (ไม่ impute target)
- สร้าง `bleach_class`, `bleached`, `is_sea`, `is_thailand` และตัด feature ที่ missing เกิน 50%

### Modeling Workflows

โปรเจกต์มี 2 workflow ที่ตั้งค่าต่างกัน จึง**ไม่ควรเทียบคะแนนข้าม workflow โดยตรง**

| | Workflow A: Decision Tree & Random Forest | Workflow B: XGBoost, LightGBM, SVM, MLP |
|---|---|---|
| Notebook | `NOTE/DT&RF_XAI(global) (เเก้ล่าสุด).ipynb`, `NOTE/DT&RF_XAI(sea) (เเก้ล่าสุด).ipynb` | `Boosting_SVM_NN.ipynb` |
| ข้อมูล | Global และ SEA แยกกัน ใช้ทุก data source | Global หลังตัด Donner, Nuryana, Setiawan เหลือ 17,493 แถว |
| Missing values | `dropna()` เหลือ 17,502 แถว (Global), 3,921 แถว (SEA) | Impute median / most frequent ใน Pipeline |
| Features | 21 features + `country_name` แบบ one-hot (`get_dummies`) รวม 110 คอลัมน์ (Global), 30 คอลัมน์ (SEA) | 21 features, StandardScaler + OneHotEncoder ใน `ColumnTransformer` |
| Train/test split | `train_test_split` แบบสุ่ม 80/20 (`random_state=42`) | `GroupShuffleSplit` 80/20 ตาม `site_id`: train 13,944 แถว (6,163 sites), test 3,549 แถว (1,541 sites), ไม่มี site ซ้ำกัน |
| Hyperparameter tuning | ไม่ได้ทำ (ใช้ค่า default) | `RandomizedSearchCV` + `GroupKFold(5)` ตาม site, scoring = macro-F1 |
| Class imbalance | ไม่ได้จัดการ | `class_weight="balanced"` (LightGBM, SVM), balanced `sample_weight` (XGBoost) |
| Explainability | Impurity feature importance, SHAP (TreeExplainer บน 100 test samples) | Permutation importance, partial dependence, ablation, Global vs SEA |

**ทำไม Workflow B แบ่งตาม site:** site เดียวกันถูกสำรวจหลายครั้ง ถ้าสุ่มแบ่งแถว site เดียวกันจะอยู่ทั้งใน train และ test โมเดลจะจำ site ได้และผลจะดูดีเกินจริง (data leakage)

**ทำไมตัด Donner, Nuryana, Setiawan ใน Workflow B:** แหล่งเหล่านี้ไม่มีแถว 0% เลย (บันทึกเฉพาะเหตุการณ์ที่ฟอกขาว) ถ้าเก็บไว้ โมเดลจะเรียนรู้ bias จากวิธีเก็บข้อมูล ไม่ใช่จากธรรมชาติ ส่วน Workflow A ยังรวมแหล่งเหล่านี้อยู่ สัดส่วน `severe` ใน test ของ Workflow A จึงสูงกว่า (447/3,501 = 12.8% เทียบกับ 213/3,549 = 6.0%)

### Machine Learning Models (6 models + baseline)

| Model | Workflow | Search space (Workflow B) | Best params |
|---|---|---|---|
| Majority-class baseline | B | – | – |
| Decision Tree | A | ไม่ได้ tune | default |
| Random Forest | A | ไม่ได้ tune | default |
| XGBoost | B | n_estimators 150–700, max_depth 3–8, learning_rate 0.01–0.3, subsample, colsample_bytree, min_child_weight, reg_lambda (20 iter) | n_estimators 674, max_depth 7, learning_rate 0.063, subsample 0.72, colsample_bytree 0.80, min_child_weight 4, reg_lambda 0.14 |
| LightGBM | B | n_estimators 150–700, num_leaves 15–127, learning_rate 0.01–0.3, min_child_samples 10–100, ... (20 iter) | n_estimators 660, num_leaves 56, learning_rate 0.027, min_child_samples 68 |
| SVM (RBF) | B | C 0.1–100, gamma 0.001–1 (12 iter, tune บน subsample 5,000 แถว แล้ว refit บน train ทั้งหมด) | C 6.36, gamma 0.133 |
| Neural Network (MLP) | B | hidden layers (64) ถึง (128, 64, 32), alpha, learning_rate_init, batch size, early stopping (15 iter) | hidden (128, 64), alpha 0.0007, learning_rate_init 0.0002, batch 64 |

### Model Evaluation

ใช้ **Macro-F1** เป็น metric หลัก ร่วมกับ Balanced Accuracy, Accuracy, F1 และ recall ของแต่ละ class เพราะ `severe` มีน้อย (6% ใน Workflow B) ถ้าดู Accuracy อย่างเดียว โมเดลที่ไม่เคยทาย `severe` เลยก็ยังได้ Accuracy 0.526 (baseline) และสำหรับการเตือนภัย การพลาด `severe` (false negative) มีต้นทุนสูงกว่าการเตือนเกิน จึงดู F1 และ recall ของ `severe` ประกอบด้วย

### Course Requirements Checklist

| Requirement | ทำที่ไหน |
|---|---|
| Environmental dataset | BCO-DMO global coral-bleaching database |
| Problem definition | Environmental Problem, Research Questions |
| Data preparation / preprocessing | `data/clean_data.py`, Pipeline ใน `Boosting_SVM_NN.ipynb` |
| Feature engineering | `bleach_class`, `is_sea`, one-hot encoding, scaling, ablation feature sets |
| ≥ 5 models compared | DT, RF, XGBoost, LightGBM, SVM, MLP (+ baseline) |
| Hyperparameter tuning | `RandomizedSearchCV` + `GroupKFold` (Workflow B) |
| Evaluation | Macro-F1, Balanced Accuracy, per-class F1, confusion matrix |
| Feature importance (explainable ML) | Permutation importance, partial dependence, SHAP, impurity importance |
| Environmental interpretation | Environmental Interpretation |

## Main Results

### Workflow B: site-based test (แนวปะการังที่โมเดลไม่เคยเห็น, 3,549 แถว)

| Model | CV Macro-F1 | Test Macro-F1 | Balanced Accuracy | Accuracy | F1 none | F1 mild | F1 severe |
|---|---:|---:|---:|---:|---:|---:|---:|
| Majority-class baseline | – | 0.230 | 0.333 | 0.526 | 0.690 | 0.000 | 0.000 |
| **XGBoost** | 0.711 ± 0.011 | **0.746** | **0.751** | **0.769** | **0.812** | 0.720 | **0.705** |
| LightGBM | 0.714 ± 0.014 | 0.744 | 0.746 | **0.769** | **0.812** | **0.722** | 0.697 |
| SVM (RBF) | 0.625 ± 0.019 | 0.701 | 0.719 | 0.743 | 0.793 | 0.696 | 0.614 |
| Neural Network (MLP) | 0.652 ± 0.024 | 0.670 | 0.643 | 0.724 | 0.779 | 0.668 | 0.562 |

ทุกโมเดลชนะ baseline มาก และคะแนน test ไม่ต่ำกว่า CV จึงไม่มีสัญญาณ overfit ค่า CV ของ SVM ต่ำกว่า test มาก เพราะ CV คิดจาก subsample 5,000 แถว ส่วน test ใช้โมเดลที่ refit บน train ทั้งหมด MLP ได้ต่ำสุดโดยเฉพาะ `severe` เพราะ `MLPClassifier` ของ scikit-learn ไม่รองรับ class weight

### Workflow A: random split (Decision Tree & Random Forest, default hyperparameters)

| Training data | Model | Test rows | Accuracy | Macro-F1 | F1 severe |
|---|---|---:|---:|---:|---:|
| Global | Decision Tree | 3,501 | 0.749 | 0.73 | 0.65 |
| Global | Random Forest | 3,501 | 0.798 | 0.77 | 0.69 |
| SEA | Decision Tree | 785 | 0.706 | 0.56 | 0.35 |
| SEA | Random Forest | 785 | 0.767 | 0.64 | 0.49 |

ใน Workflow A ทั้ง Global และ SEA ทดสอบบน test set ของตัวเอง (คนละชุดกัน) และแบ่งแบบสุ่มตามแถว ซึ่งทำให้ site เดียวกันอยู่ได้ทั้งใน train และ test คะแนน RF Global (0.77) จึงน่าจะสูงเกินจริงเมื่อเทียบกับ Workflow B ส่วน SEA test มี `severe` เพียง 23 แถว คะแนน F1 severe ของ SEA จึงไม่แน่นอนสูง

### Global vs. Southeast Asia (Workflow B, XGBoost)

ประเมินบน SEA test sites ชุดเดียวกัน 849 แถว (none 595, mild 230, severe 24)

| Training data | Training rows | Macro-F1 on SEA test |
|---|---:|---:|
| **Global data** | 13,944 | **0.691** |
| SEA data only | 3,639 | 0.666 |

โมเดลที่ train ด้วยข้อมูล Global ทำคะแนนบน SEA test ได้สูงกว่าเล็กน้อย น่าจะเป็นเพราะข้อมูลจากภูมิภาคอื่นช่วยให้โมเดลเห็นตัวอย่าง `severe` มากขึ้น แต่ SEA test มี `severe` เพียง 24 แถว (2.8%) ผลต่าง 0.025 จึงยังสรุปไม่ได้ว่า Global model ดีกว่าในทุกกรณี

## Best Model

**XGBoost** (Workflow B) ได้ Macro-F1 **0.746**, Balanced Accuracy 0.751 และ F1 `severe` **0.705** บนแนวปะการังที่ไม่เคยเห็น โดยจับเหตุการณ์ `severe` ได้ **73%** (recall 0.73, precision 0.68) LightGBM ได้ผลแทบเท่ากัน (Macro-F1 0.744) แต่ XGBoost ทำ F1 `severe` ได้ดีกว่าเล็กน้อย ซึ่งเป็น class ที่สำคัญที่สุดสำหรับการเตือนภัย

Random Forest ใน Workflow A ได้ Macro-F1 0.77 แต่ใช้ random split, ไม่ได้ tune และรวมแหล่งข้อมูลที่มี bias จึงไม่นำมาเทียบเพื่อเลือก Best Model

## Feature Importance

ใช้ **Permutation Importance** บน XGBoost test set (macro-F1 ที่ลดลงเมื่อสลับค่า feature ทีละตัว, `n_repeats=5`) และ **Partial Dependence** (ค่าเฉลี่ยของ P(severe) ที่โมเดลทำนาย เมื่อเปลี่ยนค่า feature ทีละตัว)

| Feature | Importance (ΔMacro-F1) | สิ่งที่โมเดลพบ (P(severe)) | Environmental interpretation |
|---|---:|---|---|
| `date_year` | 0.102 | 2–3% ในปีปกติ พุ่งถึงประมาณ 9.1% ช่วง 2014–2016 | ตรงกับ Global Coral Bleaching Event ครั้งที่ 3 (2014–2017) ปีจึงแทนคลื่นความร้อนระดับโลกที่ค่าความร้อนรายจุดจับได้ไม่หมด |
| `longitude_degrees`, `latitude_degrees` | 0.035, 0.026 | ความเสี่ยงต่างกันตามภูมิภาค | แต่ละพื้นที่ทนร้อนได้ไม่เท่ากัน และอาจสะท้อนโปรแกรมสำรวจที่ต่างกัน |
| `tsa_dhw` | 0.026 | 4.5% ที่ DHW 0 เพิ่มเป็นประมาณ 8.8% ที่ DHW ~5 สัปดาห์ แล้วค่อย ๆ อิ่มตัว (10.2% ที่ DHW 9.6) | ตัวแปรสิ่งแวดล้อมที่สำคัญที่สุด ความเสี่ยงชันที่สุดช่วง DHW 0–4 ใกล้เกณฑ์ NOAA Coral Reef Watch (4 DHW = Alert Level 1) |
| `depth_m` | 0.024 | 8.1% ที่ 0.7 m ลดเหลือ 4.9% ที่ 18 m | น้ำตื้นได้รับแสงและความร้อนมากกว่า แนวที่ลึกกว่าอาจเป็นแหล่งหลบภัย |
| `cyclone_frequency` | 0.018 | พื้นที่ที่มีพายุบ่อยมีความเสี่ยงต่ำลงเล็กน้อย | การกวนผสมของน้ำอาจช่วยลดความร้อน แต่ผลมีขนาดเล็ก |
| `ssta_frequency` | 0.012 | 4.3% เพิ่มเป็นประมาณ 6.9% เมื่อมีสัปดาห์ร้อนผิดปกติ 15–18 สัปดาห์ในรอบปี | ความร้อนที่เกิดซ้ำทำให้ปะการังฟื้นตัวไม่ทัน |
| `turbidity` | 0.005 | สูงสุด 6.9% ที่น้ำใสปานกลาง ลดเหลือ 5.0% เมื่อน้ำขุ่น | น้ำขุ่นอาจช่วยบังแสง แต่ผลมีขนาดเล็ก |
| `exposure` | 0.004 | Exposed 6.5%, Sometimes 6.4%, Sheltered 6.8% | ไม่พบความแตกต่างชัดเจนในข้อมูลนี้ |

ทุกโมเดลใน Workflow B (XGBoost, LightGBM, SVM, MLP) ให้ `date_year`, พิกัด และ `tsa_dhw` อยู่กลุ่มบน ผลจึงไม่ได้ขึ้นกับโมเดลใดโมเดลหนึ่ง ใน Workflow A ใช้ impurity importance และ SHAP ของ Random Forest เพิ่มเติม (ดูใน notebook)

**Ablation Analysis** (XGBoost, best params เดิม, test set เดิม)

| Feature set | n features | Macro-F1 | F1 severe |
|---|---:|---:|---:|
| All features | 21 | 0.746 | 0.705 |
| Without `date_year` | 20 | 0.719 | 0.659 |
| Environmental only (ตัด `date_year`, `date_month`, `latitude_degrees`, `longitude_degrees`, `ocean_name`, `realm_name`) | 15 | 0.702 | 0.633 |

แม้เหลือแต่ตัวแปรความร้อนและสภาพแนวปะการัง โมเดลยังได้ Macro-F1 0.702 (baseline 0.230) แปลว่าทำนายจากสิ่งแวดล้อมได้จริง ไม่ได้อาศัยการจำพื้นที่หรือปี ข้อมูลเวลาและตำแหน่งช่วยเพิ่มอีกประมาณ 0.04

## Environmental Interpretation

- **Accumulated heat stress:** `tsa_dhw` เป็นตัวแปรสิ่งแวดล้อมที่สำคัญที่สุด ความเสี่ยงเพิ่มเร็วที่สุดช่วง DHW 0–4 จึงควรเริ่มเฝ้าระวังตั้งแต่ DHW ประมาณ 2–4 ไม่ต้องรอถึง Alert Level 2 (8 DHW)
- **Repeated heat:** `ssta_frequency` สูงขึ้น ความเสี่ยงก็สูงขึ้น ความร้อนที่เกิดซ้ำในรอบปีทำให้ปะการังมีเวลาฟื้นตัวน้อยลง
- **Depth:** จุดสำรวจน้ำตื้นเสี่ยงกว่าน้ำลึก (8.1% เทียบกับ 4.9%) แนวที่ลึกกว่าจึงน่าจะเป็นพื้นที่หลบภัยที่ดีกว่า แต่ผลนี้ยังไม่ได้ยืนยันว่าความลึกเป็นสาเหตุโดยตรง
- **Time and location:** ปี 2014–2016 และภูมิภาคมีผลมาก แสดงว่าเหตุการณ์ระดับโลกและความแตกต่างระหว่างพื้นที่มีส่วนสำคัญ นอกเหนือจากค่าความร้อนรายจุด
- **Cyclone, turbidity, exposure:** ผลมีขนาดเล็ก ยังไม่ควรสรุปว่าปัจจัยเหล่านี้ช่วยป้องกันการฟอกขาว

**การนำไปใช้**

1. **ฟื้นฟูแนวปะการัง:** เลือกพื้นที่ที่ลึกกว่าและมี DHW ต่ำในอดีตเป็นลำดับแรก
2. **การท่องเที่ยวและการเตือนภัย:** ใช้ DHW รายสัปดาห์จาก NOAA Coral Reef Watch เป็นตัวตั้งต้นเตือนภัย แล้วใช้โมเดลประเมินความเสี่ยงของแต่ละแหล่งดำน้ำ เพื่อวางแผนปิดพักแหล่งน้ำตื้นที่เสี่ยงก่อน
3. **SEA:** ข้อมูล `severe` ใน SEA มีน้อย การใช้ข้อมูล Global ช่วย train จึงเหมาะกว่าการใช้ข้อมูล SEA อย่างเดียวในตอนนี้

ผลจาก Machine Learning แสดงความสัมพันธ์ที่ช่วยในการทำนาย ไม่ใช่หลักฐานเหตุและผล จึงควรใช้เป็นข้อมูลประกอบการวางแผนติดตามแนวปะการัง ไม่ใช่ใช้ตัดสินใจเพียงอย่างเดียว

## Limitations

- `severe` มีเพียง 6% ของข้อมูลใน Workflow B และใน SEA test มีเพียง 24 แถว
- Feature importance และ partial dependence แสดงความสัมพันธ์ที่โมเดลเรียนรู้ ไม่ได้ยืนยันเหตุและผล และ partial dependence แสดงผลเฉลี่ยโดยไม่แยก interaction
- `date_year` และพิกัดอาจสะท้อนความแตกต่างของโปรแกรมสำรวจในแต่ละพื้นที่ด้วย
- Heat stress มาจากข้อมูลดาวเทียมที่เฉลี่ยเป็นพื้นที่ ไม่ได้วัดที่ตัวปะการัง
- Feature ด้านความร้อนหลายตัวมีความสัมพันธ์กันสูง (เช่น `tsa_dhw` กับ `ssta_dhw`) ทำให้ importance ถูกแบ่งกระจาย
- แหล่งข้อมูลวัดการฟอกขาวด้วยวิธีต่างกัน และบางแหล่งบันทึกเฉพาะเหตุการณ์ที่ฟอกขาว
- Workflow A ใช้ random split, ไม่ได้ tune, ไม่ได้จัดการ class imbalance และรวม `country_name` เป็น feature จึงเทียบกับ Workflow B ไม่ได้โดยตรง
- SVM ถูก tune บน subsample และ MLP ไม่มี class weight

## Future Work

- รัน Decision Tree และ Random Forest ใหม่ด้วย setup เดียวกับ Workflow B (ตัด Donner, group split ตาม site, tune ด้วย `GroupKFold`) เพื่อให้เทียบทั้ง 6 โมเดลในตารางเดียวได้
- ทดลอง KNN และ regression บน `percent_bleaching` (ใช้ `log1p` เพราะ target เบ้มาก)
- ทำ SHAP ของ XGBoost สำหรับ class `severe` และดู interaction ระหว่าง DHW กับความลึก
- ใช้ weighted loss กับ Neural Network และทดสอบการแบ่งตามเวลา (train ปีก่อน ทดสอบปีหลัง)

## How to Run

1. เปิด `Boosting_SVM_NN.ipynb` ใน Google Colab แล้วเลือก Runtime → Run all (ใช้เวลาประมาณ 20–25 นาที)
2. Notebook ใน `NOTE/` อ่านข้อมูลจาก Google Drive (`/content/drive/MyDrive/ML PROJ101/`) ให้อัปโหลดไฟล์จาก `data/` ไปที่ Drive หรือแก้ path ก่อนรัน
3. ติดตั้ง library ที่ใช้: `pandas`, `numpy`, `scikit-learn`, `matplotlib`, `scipy`, `xgboost`, `lightgbm`, `shap`

ตัวเลขทั้งหมดมาจากการรันบน Colab ด้วย `random_state=42` ถ้ารันซ้ำด้วย library คนละเวอร์ชัน ตัวเลขอาจต่างเล็กน้อย

## Repository Structure

| File / Folder | Description |
|---|---|
| `data/coral_clean_global.csv`, `data/coral_clean_sea.csv` | ข้อมูลที่ clean แล้ว (Global และ SEA) |
| `data/clean_data.py` | Script สำหรับ clean ข้อมูลดิบจาก BCO-DMO |
| `data/data_dictionary.csv` | คำอธิบาย feature, หน่วย และ % missing |
| `NOTE/DT&RF_XAI(global) (เเก้ล่าสุด).ipynb`, `NOTE/DT&RF_XAI(sea) (เเก้ล่าสุด).ipynb` | Decision Tree, Random Forest และ SHAP (target `bleach_class`) |
| `NOTE/DT&RF_XAI(global).ipynb`, `NOTE/DT&RF_XAI(sea).ipynb` | รุ่นแรก ใช้ target `bleached` (0/1) ไม่ได้ใช้ในตารางผล |
| `Boosting_SVM_NN.ipynb` | XGBoost, LightGBM, SVM, MLP, tuning และ interpretation |
| `MODEL_RESULTS.md` | รายละเอียดผลของ Workflow B |
| `FIGURES/` | รูปผลการวิเคราะห์ |

## References

van Woesik, R., & Kratochwill, C. (2022). A global coral-bleaching database, 1980–2020. *Scientific Data, 9*, 20. [https://doi.org/10.1038/s41597-021-01101-8](https://doi.org/10.1038/s41597-021-01101-8)

BCO-DMO. *A global coral-bleaching database, 1980–2020* (Dataset 773466). [https://www.bco-dmo.org/dataset/773466](https://www.bco-dmo.org/dataset/773466)

NOAA Coral Reef Watch. Degree Heating Week and bleaching alert levels. [https://coralreefwatch.noaa.gov/](https://coralreefwatch.noaa.gov/)
