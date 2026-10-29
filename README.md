# 🚘 AutoVision AI: Car Price Prediction & Market Intelligence

A complete, end-to-end Machine Learning and Analytics project for used car price valuation. Built using **Scikit-Learn**, **Gradio**, and **Streamlit**.

---

## 🌟 Key Features

- **Dual Interactive Interfaces**:
  - **Gradio Web Application (`gradio_app.py`)**: Sleek cyberpunk/dark-gradient interface with instant valuation, fair negotiation bands, depreciation calculators, and 1-click test presets.
  - **Streamlit Analytics Dashboard (`app.py`)**: Full market exploratory dashboard with KPI metrics, Plotly scatter and depreciation curves, feature importance analysis, and future what-if scenario simulators.
- **Robust ML Pipeline (`model_utils.py`)**:
  - Multiple Linear Regression with R² ≈ 0.854 and RMSE ≈ 1.83.
  - Automated fallback training and model serialization (`car_price_model.pkl`).
  - Strict type sanitization and dummy-variable handling.
- **Practical Lab Notebook (`Car_Price_Prediction_Project(SI).ipynb`)**:
  - 11 guided practical experiments from exploratory statistics, hypothesis testing (Z/T-test, ANOVA), correlation analysis, regression model comparison, residual diagnostics, to model export.

---

## 📁 Project Directory Structure

```text
SI LAB PROJECT/
├── app.py                             # Streamlit Analytics & Valuation Dashboard
├── gradio_app.py                      # Standalone Gradio Web Application
├── model_utils.py                     # Unified ML training, loading & prediction engine
├── car_data.csv                       # Historical used car dataset
├── car_price_model.pkl                # Trained model bundle
├── Car_Price_Prediction_Project(SI).ipynb  # Jupyter Lab notebook (Practicals 1 to 11)
├── requirements.txt                   # Project dependencies
└── README.md                          # Project documentation
```

---

## 🚀 How to Run

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Launch Gradio Web App
```bash
python gradio_app.py
```
Open your browser at: **`http://127.0.0.1:7860`**

### 3. Launch Streamlit Analytics Dashboard
```bash
streamlit run app.py
```
Open your browser at: **`http://localhost:8501`**

### 4. Run the Jupyter Notebook
```bash
jupyter notebook "Car_Price_Prediction_Project(SI).ipynb"
```

---

## 📊 Dataset & Model Details

- **Dataset**: `car_data.csv` (301 records)
- **Features Used**:
  - `Present_Price`: Original showroom / ex-showroom price (₹ Lakhs)
  - `Kms_Driven`: Total distance traveled (km)
  - `Age`: Vehicle age in years (`2026 - Year`)
  - `Fuel_Type`: Petrol, Diesel, CNG
  - `Seller_Type`: Dealer, Individual
  - `Transmission`: Manual, Automatic
- **Target**: `Selling_Price` (₹ Lakhs)
- **Model Evaluation**:
  - **R² Score**: `0.8541` (explains >85% of price variation)
  - **RMSE**: `1.8331 Lakhs`
  - **MAE**: `1.1872 Lakhs`
