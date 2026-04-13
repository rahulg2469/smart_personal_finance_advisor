# Smart Personal Finance Advisor

An AI-powered personal finance system that provides actionable, prioritized budget recommendations using machine learning.

**CS 5100 - Foundations of Artificial Intelligence**  
Northeastern University, Khoury College of Computer Sciences

## Team Members
| Name | NUID |
|------|------|
| Rahul Gudivada | 002560822 |
| Elenta Suzan Jacob | 002530281 |
| Shawn Godfrey Thomas Sahaya Cruz | 002545355 |

## Problem Statement

While 78% of Americans live paycheck-to-paycheck, existing budgeting apps only provide reactive transaction tracking. They answer "where did my money go?" but not "what should I cut first and why will it work for me?"

Our system combines spending pattern analysis, mathematical optimization, and behavioral success prediction to generate prioritized budget recommendations.

## Project Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│  Layer 1: User Input                                            │
├─────────────────────────────────────────────────────────────────┤
│  Layer 2: ML Analysis                                           │
│  ┌──────────────┐ ┌──────────────┐ ┌──────────────┐ ┌─────────┐ │
│  │Random Forest │ │   K-Means    │ │  Isolation   │ │XGBoost  │ │
│  │Categorization│ │  Clustering  │ │   Forest     │ │Prediction│ │
│  └──────────────┘ └──────────────┘ └──────────────┘ └─────────┘ │
├─────────────────────────────────────────────────────────────────┤
│  Layer 3: Budget Optimization (SLSQP)                           │
├─────────────────────────────────────────────────────────────────┤
│  Layer 4: Neural Network (Advice Prioritization)                │
├─────────────────────────────────────────────────────────────────┤
│  Layer 5: Streamlit Dashboard                                   │
└─────────────────────────────────────────────────────────────────┘
```

## ML Components

| Component | Algorithm | Purpose | Performance |
|-----------|-----------|---------|-------------|
| Transaction Categorization | Random Forest | Auto-categorize transactions into 14 categories | 93.45% accuracy |
| User Segmentation | K-Means (K=4) | Cluster users into spending personality types | 4 distinct clusters |
| Anomaly Detection | Isolation Forest | Flag unusual spending patterns and potential fraud | 90.10% detection, AUC-ROC 0.9350 |
| Spending Prediction | XGBoost | Predict next month's total spending | R² 0.8151, MAE $1,118 |
| Budget Optimization | SLSQP | Calculate optimal budget cuts across categories | 10:2:1 priority ratio |
| Advice Prioritization | Neural Network (PyTorch) | Prioritize which budget cuts to make first | 90.50% accuracy |

## User Clusters (K-Means)

K-Means identified 4 distinct spending personalities:

| Cluster | Name | % of Users | Characteristics |
|---------|------|------------|-----------------|
| 0 | Average | 35.5% | Grocery and gas focused, typical spending patterns |
| 1 | High Spender | 38.0% | Highest total spent, balanced across categories |
| 2 | Inconsistent | 7.5% | Few transactions, high amounts, online shopping heavy |
| 3 | Budget-Conscious | 18.9% | Lower spending, family focused |

## Spending Categories (14 total)

1. Groceries (In-store)
2. Groceries (Online)
3. Entertainment
4. Shopping (In-store)
5. Shopping (Online)
6. Food & Dining
7. Gas & Transport
8. Health & Fitness
9. Home
10. Kids & Pets
11. Misc (Online)
12. Misc (In-store)
13. Personal Care
14. Travel

## SLSQP Priority Weights

| Priority | Weight | Categories |
|----------|--------|------------|
| Essential | 10.0 | Groceries, Gas & Transport, Health |
| Important | 5.0 | Home, Kids & Pets, Food & Dining |
| Flexible | 1.0 | Entertainment, Shopping, Travel, Personal Care, Misc |

## Project Structure

```
smart_personal_finance_advisor/
├── dashboard/
│   ├── app.py              # Main Streamlit application
│   ├── auth.py             # Supabase authentication
│   ├── models.py           # ML model integration
│   └── static/             # Images and icons
│       ├── darklogo.png
│       ├── bar-graph.png
│       ├── login.png
│       ├── spending-breakdown.png
│       └── target.png
├── notebooks/
│   ├── Data_Exploration.ipynb
│   ├── K_Means_Clustering.ipynb
│   ├── Random_Forest.ipynb
│   ├── Isolation_Forest.ipynb
│   ├── Gradient_Boosting.ipynb
│   ├── SLSQP_Optimization.ipynb
│   ├── Neural_Networks.ipynb
│   └── End_to_End_Testing.ipynb
├── models/                  # Download from Google Drive
├── data/                    # Download from Google Drive
├── images/
├── .streamlit/
│   └── config.toml
├── requirements.txt
├── PROGRESS.md
├── .gitignore
└── README.md
```

---

## Google Drive - Models & Data

All trained model files and datasets are stored in Google Drive because they are too large for GitHub.

**Google Drive Link:** https://drive.google.com/drive/folders/1iOBGDDfd3hPeawA-ALKRU9kMIrk5jZVx?usp=sharing

### Files to Download

After cloning the repo, download these files from Google Drive and place them in the correct folders:

**Place in `models/` folder:**
| File | Size | Description |
|------|------|-------------|
| `random_forest_model.pkl` | ~487MB | Transaction categorization model |
| `kmeans_model.pkl` | ~1KB | User clustering model |
| `kmeans_scaler.pkl` | ~1KB | Scaler for K-Means |
| `isolation_forest_model_v2.pkl` | ~5MB | Anomaly detection model |
| `isolation_forest_scaler_v2.pkl` | ~1KB | Scaler for Isolation Forest |
| `gradient_boosting_model_v3.pkl` | ~2MB | Spending prediction model |
| `gb_feature_columns_v3.pkl` | ~1KB | Feature columns for Gradient Boosting |
| `slsqp_category_config.pkl` | ~1KB | Budget optimization config |
| `neural_network_model.pth` | ~50KB | Advice prioritization model |
| `scaler.pkl` | ~1KB | Scaler for Neural Network |

**Place in `data/` folder:**
| File | Size | Description |
|------|------|-------------|
| `user_monthly_spending.csv` | ~10MB | Aggregated user spending data for demo |
| `fraudTrain.csv` | ~343MB | Original Kaggle training dataset |
| `fraudTrain_cleaned.csv` | ~300MB | Cleaned training dataset |
| `fraudTest.csv` | ~147MB | Original Kaggle test dataset |

---

## Setup Instructions

### Prerequisites
- Python 3.10 or higher
- pip (Python package manager)
- Git

### Step 1: Clone the Repository

```bash
git clone https://github.com/rahulg2469/smart_personal_finance_advisor.git
cd smart_personal_finance_advisor
```

### Step 2: Create Virtual Environment

**Windows:**
```bash
python -m venv venv
venv\Scripts\activate
```

**Mac/Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Download Models & Data from Google Drive

1. Go to the Google Drive link above
2. Download all `.pkl` and `.pth` files → place in `models/` folder
3. Download all `.csv` files → place in `data/` folder

### Step 5: Run the Application

```bash
streamlit run dashboard/app.py
```

The app will open at `http://localhost:8501`

---

## Using the Application

### Demo Mode (No Login Required)
1. Click "Try Demo" on the landing page
2. Select a sample user from the dropdown
3. View spending breakdown, cluster profile, budget recommendations
4. Adjust the savings slider to see optimized budget cuts
5. View anomaly detection results and spending forecast

### CSV Upload Format

For transaction uploads, use this format:

| Column | Required | Description |
|--------|----------|-------------|
| merchant | Yes | Merchant name |
| amt | Yes | Transaction amount |
| lat | Yes | User latitude |
| long | Yes | User longitude |
| city_pop | Yes | City population |
| merch_lat | Yes | Merchant latitude |
| merch_long | Yes | Merchant longitude |
| hour | Yes | Hour of transaction (0-23) |
| day_of_week | Yes | Day of week (0-6) |
| month | Yes | Month (1-12) |

---

## Jupyter Notebooks

All model training notebooks are in the `notebooks/` folder. To run them:

1. Open Google Colab
2. Upload the notebook
3. Mount Google Drive:
   ```python
   from google.colab import drive
   drive.mount('/content/drive')
   ```
4. Ensure dataset is in `/content/drive/MyDrive/archive/`
5. Run all cells

### Notebook Descriptions

| Notebook | Description |
|----------|-------------|
| `Data_Exploration.ipynb` | Initial data analysis and feature engineering |
| `Random_Forest.ipynb` | Trains transaction categorization model (93.45% accuracy) |
| `K_Means_Clustering.ipynb` | Creates 4 user spending clusters |
| `Isolation_Forest.ipynb` | Trains anomaly detection model (AUC-ROC 0.9350) |
| `Gradient_Boosting.ipynb` | Predicts next month spending (R² 0.8151) |
| `SLSQP_Optimization.ipynb` | Budget optimization with priority weights |
| `Neural_Networks.ipynb` | Ranks spending categories by cut difficulty (90.50% accuracy) |
| `End_to_End_Testing.ipynb` | Tests all models together |

---

## Dataset

We use the [Kaggle Credit Card Transactions Fraud Detection](https://www.kaggle.com/datasets/kartik2112/fraud-detection) dataset:
- 1.3M transactions from 983 users
- 14 spending categories
- Date range: January 2019 - June 2020

## Key Findings

- Fraudulent transactions average $531 vs $68 for normal transactions
- Fraud peaks during late night hours (22:00-23:00) with ~3% fraud rate
- User-relative features (z-scores) significantly improved anomaly detection
- Lag features (previous months' spending) boosted prediction R² from 0.52 to 0.77
- SLSQP optimizer produces consistent 10:2:1 cut ratio across flexible/important/essential categories

---

## Timeline

| Week | Tasks | Status |
|------|-------|--------|
| 1-2 | Data exploration, feature engineering | Complete |
| 3-4 | Train ML models (Random Forest, K-Means, Isolation Forest, Gradient Boosting) | Complete |
| 5 | SLSQP budget optimization | Complete |
| 6 | Neural network for advice prioritization | Complete |
| 7 | End-to-end testing pipeline | Complete |
| 8 | Streamlit dashboard | Complete |

---

## Links

- **GitHub Repository:** https://github.com/rahulg2469/smart_personal_finance_advisor
- **Google Drive (Models & Data):** https://drive.google.com/drive/folders/1iOBGDDfd3hPeawA-ALKRU9kMIrk5jZVx?usp=sharing
- **Dataset (Kaggle):** https://www.kaggle.com/datasets/kartik2112/fraud-detection
