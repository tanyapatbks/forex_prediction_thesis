# Forex Price Prediction System

ระบบทำนายแนวโน้มราคาในตลาด Forex โดยใช้แบบจำลองการเรียนรู้เชิงลึก (Deep Learning) และการเรียนรู้ของเครื่อง (Machine Learning) ที่มุ่งเน้นความสามารถในการทำกำไร

## ภาพรวมโปรเจค

โปรเจคนี้พัฒนาระบบทำนายแนวโน้มราคาในตลาด Forex โดยใช้เทคนิคการเรียนรู้ของเครื่องและการเรียนรู้เชิงลึก มุ่งเน้นที่การพัฒนาและเปรียบเทียบแบบจำลองสามรูปแบบ ได้แก่:

1. **CNN-LSTM Hybrid**: แบบจำลองผสมที่รวมจุดแข็งของ Convolutional Neural Network (CNN) และ Long Short-Term Memory (LSTM)
2. **Temporal Fusion Transformer (TFT)**: แบบจำลองที่ออกแบบเฉพาะสำหรับการทำนายอนุกรมเวลาหลายตัวแปร 
3. **XGBoost**: แบบจำลองการเรียนรู้ของเครื่องที่มีประสิทธิภาพสูงสำหรับการทำนายแนวโน้ม

นอกจากนี้ โปรเจคยังศึกษาประสิทธิภาพของเทคนิค Bagging ที่ใช้ข้อมูลจากหลายคู่สกุลเงิน (EURUSD, GBPUSD, USDJPY) มาเรียนรู้ร่วมกัน เปรียบเทียบกับการใช้ข้อมูลจากคู่สกุลเงินเดียว โดยมีจุดประสงค์ในการพัฒนาระบบที่มีความสามารถในการทำกำไรและสามารถนำไปประยุกต์ใช้ในการลงทุนจริง

## โครงสร้างโปรเจค

```
forex_prediction_thesis/
├── data/
│   ├── raw/                      # ข้อมูลดิบ CSV
│   ├── processed/                # ข้อมูลที่ผ่านการ preprocess
│   └── features/                 # ข้อมูลที่ผ่านการสร้างคุณลักษณะ
├── models/
│   ├── trained/                  # โมเดลที่ผ่านการฝึกฝน
│   └── results/                  # ผลลัพธ์การทำนาย
├── logs/                         # บันทึกการทำงาน
├── notebooks/                    # Jupyter notebooks
├── src/
│   ├── data/
│   │   ├── __init__.py
│   │   ├── data_loader.py        # ฟังก์ชันโหลดข้อมูล
│   │   └── data_preprocessor.py  # ฟังก์ชันเตรียมข้อมูล
│   ├── features/
│   │   ├── __init__.py
│   │   └── feature_enhancement.py # การสร้างและคัดเลือกคุณลักษณะ
│   ├── models/
│   │   ├── __init__.py
│   │   ├── cnn_lstm.py           # โมเดล CNN-LSTM
│   │   ├── tft.py                # โมเดล Temporal Fusion Transformer
│   │   ├── xgboost_model.py      # โมเดล XGBoost
│   │   └── bagging_model.py      # โมเดล Bagging
│   ├── evaluation/
│   │   ├── __init__.py
│   │   └── performance_metrics.py # ฟังก์ชันประเมินประสิทธิภาพ
│   └── utils/
│       ├── __init__.py
│       ├── config.py             # ไฟล์การตั้งค่า
│       ├── logger.py             # ฟังก์ชันบันทึกการทำงาน
│       └── visualization.py      # ฟังก์ชันการแสดงผลข้อมูล
├── config/
│   ├── default_config.json       # การตั้งค่าเริ่มต้น
│   └── hyperparameters/          # การตั้งค่า hyperparameters
├── main.py                       # จุดเริ่มต้นการทำงานหลัก
├── train.py                      # สคริปต์สำหรับการฝึกฝนโมเดล
├── evaluate.py                   # สคริปต์สำหรับการประเมินโมเดล
├── hyperparameter_tuning.py      # สคริปต์สำหรับปรับแต่ง hyperparameters
└── requirements.txt              # แพ็คเกจที่จำเป็น
```

## ขั้นตอนการพัฒนา

โปรเจคแบ่งการพัฒนาออกเป็น 4 ขั้นตอนหลัก:

1. **Data Collection & Preprocessing**: การรวบรวมและเตรียมข้อมูลสำหรับการวิเคราะห์
2. **Feature Enhancement**: การสร้างและคัดเลือกคุณลักษณะที่เหมาะสม
3. **Model Development**: การพัฒนาและฝึกฝนแบบจำลอง
4. **Model Evaluation & Performance Analysis**: การประเมินประสิทธิภาพและวิเคราะห์ผลการทำนาย

## การติดตั้ง

1. Clone โปรเจค:
```bash
git clone https://github.com/your-username/forex_prediction_thesis.git
cd forex_prediction_thesis
```

2. สร้าง virtual environment และติดตั้งแพ็คเกจที่จำเป็น:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

3. เตรียมโครงสร้างโฟลเดอร์:
```bash
mkdir -p data/raw data/processed data/features models/trained models/results logs notebooks config/hyperparameters
```

4. ย้ายไฟล์ข้อมูล CSV ไปยังโฟลเดอร์ `data/raw`:
```bash
cp path/to/EURUSD_1H.csv data/raw/
cp path/to/GBPUSD_1H.csv data/raw/
cp path/to/USDJPY_1H.csv data/raw/
```

## การใช้งาน

### การรันทั้งกระบวนการ (Pipeline)

```bash
python main.py --model all --currency all
```

### การรันเฉพาะขั้นตอนที่ต้องการ

1. เตรียมข้อมูล:
```bash
python main.py --stage 1 --currency all
```

2. สร้างและคัดเลือกคุณลักษณะ:
```bash
python main.py --stage 2 --currency all
```

3. ฝึกฝนโมเดล:
```bash
python main.py --stage 3 --model CNN-LSTM --currency EURUSD
```

4. ประเมินโมเดล:
```bash
python main.py --stage 4 --model all --currency all --visualize
```

### การฝึกฝนโมเดลแบบแยกไฟล์

```bash
python train.py --model CNN-LSTM --currency EURUSD --visualize
```

### การประเมินโมเดลแบบแยกไฟล์

```bash
python evaluate.py --model CNN-LSTM --currency all --visualize --compare --market_conditions
```

### การปรับแต่ง Hyperparameters

```bash
python hyperparameter_tuning.py --model XGBoost --currency EURUSD --trials 100 --visualize
```

## ตัวชี้วัดประสิทธิภาพ

โปรเจคประเมินประสิทธิภาพของโมเดลโดยพิจารณาตัวชี้วัดด้านการลงทุนหลายมิติ:

- **ผลตอบแทนรายปี (Annual Return)**: วัดผลตอบแทนที่ได้จากการลงทุนตามสัญญาณของแบบจำลองในช่วงเวลา 1 ปี
- **อัตราส่วนของการเทรดที่ทำกำไร (Win Rate)**: คำนวณเปอร์เซ็นต์ของการเทรดที่ให้ผลกำไร
- **ประสิทธิภาพในสภาวะตลาดที่แตกต่างกัน**: ทดสอบประสิทธิภาพของแบบจำลองในสภาวะตลาดที่แตกต่างกัน เช่น ตลาดขาขึ้น ตลาดขาลง และตลาดไซด์เวย์
- **เปรียบเทียบกับกลยุทธ์ซื้อและถือครอง (Buy & Hold)**: เปรียบเทียบผลตอบแทนจากแบบจำลองกับกลยุทธ์ซื้อและถือครอง
- **เปรียบเทียบผลตอบแทนระหว่างการใช้ข้อมูลคู่สกุลเงินเดียวกับการใช้ข้อมูลหลายคู่สกุลเงิน (Bagging)**: วิเคราะห์ว่าการใช้ข้อมูลจากหลายคู่สกุลเงินช่วยปรับปรุงประสิทธิภาพของแบบจำลองได้จริงหรือไม่

## แนวทางการพัฒนาต่อ

- เพิ่มคู่สกุลเงินและช่วงเวลาในการทดสอบ
- เพิ่มปัจจัยพื้นฐาน (Fundamental Factors) เข้าไปในโมเดล
- พัฒนาระบบการซื้อขายอัตโนมัติ (Automated Trading System) ที่เชื่อมต่อกับโบรกเกอร์
- ทดลองใช้เทคนิค Reinforcement Learning เพื่อพัฒนากลยุทธ์การลงทุน
- สร้าง Web Application สำหรับแสดงผลการทำนายและผลการดำเนินงาน

## เทคโนโลยีที่ใช้

- **Python**: ภาษาโปรแกรมมิ่งหลัก
- **TensorFlow/Keras**: สำหรับพัฒนาโมเดล CNN-LSTM
- **PyTorch/PyTorch Forecasting**: สำหรับพัฒนาโมเดล TFT
- **XGBoost**: สำหรับพัฒนาโมเดล XGBoost
- **Pandas/NumPy**: สำหรับการจัดการและวิเคราะห์ข้อมูล
- **Scikit-learn**: สำหรับการประมวลผลข้อมูลและการประเมินโมเดล
- **Matplotlib/Seaborn**: สำหรับการแสดงผลข้อมูลและผลลัพธ์
- **Optuna**: สำหรับการปรับแต่ง Hyperparameters
