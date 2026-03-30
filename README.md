# Smart Personal Finance Advisor

An AI-powered personal finance system that provides actionable, prioritized budget recommendations using machine learning.

**CS 5100 - Foundations of Artificial Intelligence**  
Northeastern University, Khoury College of Computer Sciences

## Team Members
- Rahul Gudivada (002560822)
- Elenta Suzan Jacob (002530281)
- Shawn Godfrey Thomas Sahaya Cruz (002545355)

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
| Anomaly Detection | Isolation Forest | Flag unusual spending patterns and potential fraud | 90.10% detection, 0.935 AUC |
| Spending Prediction | XGBoost | Predict next month's total spending | R²=0.815, MAE=$1,118 |
| Budget Optimization | SLSQP | Calculate optimal budget cuts across categories | 10:2:1 priority ratio |
| Advice Prioritization | Neural Network (PyTorch) | Prioritize which budget cuts to make first | 90.50% accuracy |

## User Clusters

K-Means identified 4 distinct spending personalities:

| Cluster | Name | % of Users | Characteristics |
|---------|------|------------|-----------------|
| 0 | Average | 35.5% | Grocery and gas focused, typical spending patterns |
| 1 | High Spender | 38.0% | Highest total spent, balanced across categories |
| 2 | Inconsistent | 7.5% | Few transactions, high amounts, online shopping heavy |
| 3 | Budget-Conscious | 18.9% | Lower spending, family focused |

## Project Structure

```
smart_personal_finance_advisor/
├── notebooks/
│   ├── Data_Exploration.ipynb
│   ├── K_Means_Clustering.ipynb
│   ├── Random_Forest.ipynb
│   ├── Isolation_Forest.ipynb
│   ├── Gradient_Boosting.ipynb
│   ├── SLSQP_Optimization.ipynb
│   ├── Neural_Networks.ipynb
│   └── End_to_End_Testing.ipynb
├── images/
├── data/
│   └── README.md            # Dataset download instructions
├── PROGRESS.md
├── README.md
└── requirements.txt
```

## Setup Instructions

### 1. Clone the repository
```bash
git clone https://github.com/rahulg2469/smart_personal_finance_advisor.git
cd smart_personal_finance_advisor
```

### 2. Create virtual environment (recommended)
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Download the dataset
See `data/README.md` for instructions to download the Kaggle dataset.


## Timeline

| Week | Tasks | Status |
|------|-------|--------|
| 1-2 | Data exploration, feature engineering | Complete |
| 3-4 | Train ML models (Random Forest, K-Means, Isolation Forest, Gradient Boosting) | Complete |
| 5 | SLSQP budget optimization | Complete |
| 6 | Neural network for advice prioritization | Complete |
| 7 | End-to-end testing pipeline | Complete |
| 8 | Streamlit dashboard | In Progress |


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
