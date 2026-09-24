# 🫀 Explainable AI (XAI) in Healthcare — Heart Disease Prediction

Final Year Project: A disease diagnostic model with SHAP/LIME explainability, deployed via Streamlit.

### 🔗 [Try the Live App](https://xai-healthcare-fyp-m9dqvwsttmatssww2gjsao.streamlit.app)

## Overview
This project predicts the presence of heart disease from patient clinical data using machine learning (Random Forest), and explains **why** each prediction was made using SHAP and LIME — making the model's decisions transparent and interpretable for clinical use cases.

## Project Structure
xai-healthcare-fyp/
├── data/ # Dataset + processed artifacts (scaler, encoders)
├── model/ # Saved trained model
├── notebook/ # Jupyter notebooks (EDA, preprocessing, training, XAI)
├── app/ # Streamlit web application
└── requirements.txt


## Key Features
- Data cleaning: identified and removed 723 duplicate rows (71% of raw data) that caused data leakage
- Compared 3 models: Logistic Regression, Random Forest, XGBoost — **Random Forest selected as final model (81.97% accuracy)**
- Global and local explanations using SHAP (TreeExplainer)
- Local explanations using LIME, cross-validated against SHAP
- Interactive Streamlit app with input validation for physiologically implausible values
- Live SHAP waterfall explanation for every prediction

## How to Run Locally
```bash
# 1. Clone the repo
git clone https://github.com/64271maaz/xai-healthcare-fyp.git
cd xai-healthcare-fyp

# 2. Create virtual environment
python -m venv venv
venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run the Streamlit app
cd app
streamlit run app.py
```

## Dataset
Heart Disease UCI Dataset (Kaggle)

## Tech Stack
Python, Scikit-learn, XGBoost, SHAP, LIME, Streamlit, Pandas, Matplotlib, Seaborn
