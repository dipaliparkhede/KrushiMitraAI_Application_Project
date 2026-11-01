# 🌾 KrishiMitra AI (कृषिमित्र) — Intelligent Agri-Advisory Platform

An end-to-end, AI-powered agricultural intelligence platform designed to empower farmers with timely crop yield forecasts, disease detection via leaf images, real-time APMC Mandi commodity rates, personalized soil nutrient recommendations, and multilingual voice assistance.

---

## 🚀 Key Modules & Capabilities

1. **Crop Yield Prediction (Random Forest Regressor)**
   - High-precision agricultural yield estimation based on state, crop variety, cropping season, cultivated area, annual rainfall, and chemical inputs (fertilizer & pesticide).
   - Validated against 21,750 historical records with zero target leakage.

2. **Crop Pest & Disease Detection (EfficientNet-B0)**
   - Deep learning vision model trained on over 22 plant disease and pest classes (Cashew, Cassava, Maize, Tomato).
   - Achieves **89.94% overall accuracy** on unseen field leaf samples.

3. **Mandi Market Prices & Trends**
   - Real-time commodity price tracking across agricultural markets and APMC mandis.

4. **Soil Health & Fertilizer Advisory**
   - Macro/micro-nutrient optimization (N-P-K, pH, Organic Carbon, Calcium Carbonate) tailored to target crops.

5. **Multilingual Voice Assistant & Weather Advisory**
   - Voice-to-text querying powered by Whisper and multilingual localization for rural accessibility.

---

## 📊 Machine Learning Model Benchmarks

### 1. Crop Yield Prediction Model Performance & Dataset Split Alignment

The production crop yield model artifact (`yield_prediction_model.joblib`) was trained on the first **17,400 rows** and rigorously evaluated on the remaining **4,350 rows (rows 17,400 to 21,750)** of the standard dataset.

The table below confirms the exact alignment between the production metadata (`yield_model_metadata.json`) and the computed metrics across the evaluation slice:

| Metric | Production Model Metadata (`yield_model_metadata.json`) | Computed on Rows `17400:21750` | Split Alignment Status |
| :--- | :---: | :---: | :--- |
| **Test $R^2$ Score** | **0.9895** | **0.9895** | Exact Match to 4 Decimals |
| **Test MAE** | **7.8035** | **7.8035** | Exact Match to 4 Decimals |
| **Test RMSE** | **96.8552** | **96.8552** | Exact Match to 4 Decimals |
| **Train $R^2$ Score** | **0.9924** | **0.9924** | Exact Match on First 17,400 Rows |
| **Train MAE** | **5.5610** | **5.5610** | Exact Match on First 17,400 Rows |
| **Train RMSE** | **85.2803** | **85.2803** | Exact Match on First 17,400 Rows |

#### Yield Model Dataset Specifications:
- **Total Dataset Records**: 21,750
- **Training Partition**: First 17,400 rows (Rows `0:17400`, 80%)
- **Testing Partition**: 4,350 rows (Rows `17400:21750`, 20%)
- **Target Variable**: `Yield` (normalized per hectare)
- **Features Used**: `Year`, `State`, `Crop`, `Season`, `Area`, `Annual_Rainfall`, `Fertilizer`, `Pesticide`
- **Data Leakage Safeguard**: The total `Production` column was strictly excluded from features to prevent circular leakage into `Yield`.
- **Coconut Multi-Scale Handling**: Coconut yields (measured in nuts/hectare up to 21,000) are natively handled alongside field crops (tonnes/hectare) using Random Forest multi-split decision trees.

---

### 2. Crop Disease Classification Model Performance

The disease classification model utilizes transfer learning on **EfficientNet-B0** evaluated on 3,776 test images across 22 crop condition classes:

| Metric | Value |
| :--- | :---: |
| **Overall Accuracy** | **89.94%** |
| **Weighted Precision** | **90.02%** |
| **Weighted Recall** | **89.94%** |
| **Weighted F1-Score** | **89.92%** |
| **Macro F1-Score** | **89.37%** |

*(Detailed class-level reports and LaTeX research tables are available in `backend/model/research_paper_table.tex` and `backend/model/evaluation_report.txt`)*.

---

## 📁 Repository Structure

```
KrishiMitraAI_project-main/
├── backend/
│   ├── main.py                     # FastAPI server & route orchestration
│   ├── database.py                 # SQLite database engine
│   ├── ml/
│   │   ├── artifacts/
│   │   │   ├── yield_prediction_model.joblib   # Trained Random Forest pipeline
│   │   │   ├── yield_model_metadata.json       # Production metrics & parameters
│   │   │   ├── yield_evaluation_report.txt     # Formatted evaluation report
│   │   │   └── yield_research_paper_table.tex  # LaTeX table for papers
│   │   ├── evaluate_yield.py       # Yield model split alignment validation script
│   │   ├── yield_predictor.py      # Yield inference engine
│   │   ├── yield_schema.py         # Pydantic request/response schemas
│   │   ├── train.py                # Crop disease training pipeline
│   │   └── evaluate.py             # Crop disease evaluation script
│   └── model/
│       ├── best_model.pth          # PyTorch EfficientNet-B0 disease weights
│       ├── classes.json            # 22 disease classes mapping
│       ├── evaluation_report.txt   # Disease classification report
│       └── research_paper_table.tex # LaTeX classification table
├── src/                            # React 18 + Vite Frontend
│   ├── pages/
│   │   ├── YieldPrediction.jsx     # Yield prediction interface & benchmark card
│   │   ├── YieldPrediction.css     # Styling for prediction & metrics table
│   │   ├── DiseaseDetection.jsx    # Leaf image upload & analysis
│   │   ├── MandiPrices.jsx         # Live market price boards
│   │   └── SoilAdvisory.jsx        # Soil testing & nutrient advisory
│   └── services/                   # Frontend API integration
└── README.md
```

---

## 🛠️ Verification & Evaluation Commands

### Verify Yield Model Split Alignment
To verify that the model metadata and dataset split match to 4 decimal places:

```bash
# Run validation with ASCII table output
python backend/ml/evaluate_yield.py

# Export table in Markdown format
python backend/ml/evaluate_yield.py --format markdown

# Export table in LaTeX format for research publications
python backend/ml/evaluate_yield.py --format latex
```

---

## 🚀 Getting Started

### 1. Backend Setup (FastAPI)
```bash
cd backend
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -r ml/requirements.txt
uvicorn main:app --reload --port 8000
```

### 2. Frontend Setup (React + Vite)
```bash
npm install
npm run dev
```
Open [http://localhost:5173](http://localhost:5173) in your browser.
