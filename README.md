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
│  Layer 1: User Input (Onboarding Form)                         │
├─────────────────────────────────────────────────────────────────┤
│  Layer 2: ML Analysis                                           │
│  ┌──────────────┐ ┌──────────────┐ ┌──────────────┐ ┌─────────┐ │
│  │Random Forest │ │   K-Means    │ │  Isolation   │ │Gradient │ │
│  │Categorization│ │  Clustering  │ │   Forest     │ │Boosting │ │
│  └──────────────┘ └──────────────┘ └──────────────┘ └─────────┘ │
├─────────────────────────────────────────────────────────────────┤
│  Layer 3: Budget Optimization (SLSQP)                           │
├─────────────────────────────────────────────────────────────────┤
│  Layer 4: Neural Network (Advice Prioritization)                │
└─────────────────────────────────────────────────────────────────┘
```

## ML Components

| Component | Algorithm | Purpose |
|-----------|-----------|---------|
| Transaction Categorization | Random Forest | Auto-categorize transactions (Dining, Groceries, etc.) |
| User Segmentation | K-Means | Cluster users into spending personality types |
| Anomaly Detection | Isolation Forest | Flag unusual spending patterns |
| Spending Prediction | Gradient Boosting | Predict next month's spending |
| Budget Optimization | SLSQP | Calculate optimal budget allocations |
| Advice Prioritization | Neural Network | Prioritize which budget cuts to make first |

## Project Structure

```
smart_personal_finance_advisor/
├── data/                    # Dataset (not tracked in git)
│   └── README.md            # Download instructions
├── models/                  # Saved trained models
├── notebooks/               # Jupyter notebooks for exploration
├── src/
│   ├── categorization/      # Random Forest module
│   ├── clustering/          # K-Means module
│   ├── anomaly/             # Isolation Forest module
│   ├── prediction/          # Gradient Boosting module
│   ├── optimization/        # SLSQP budget optimizer
│   ├── neural_network/      # Advice prioritization NN
│   └── utils/               # Shared helper functions
├── dashboard/               # Streamlit app
├── docs/                    # Reports and documentation
├── tests/                   # Unit tests
├── requirements.txt         # Python dependencies
└── README.md
```

## Setup Instructions

### 1. Clone the repository
```bash
git clone https://github.com/YOUR_USERNAME/smart_personal_finance_advisor.git
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

| Week | Tasks |
|------|-------|
| 1-2 | Data exploration, feature engineering, literature review |
| 3-4 | Train ML models (Random Forest, K-Means, Isolation Forest, Gradient Boosting) |
| 5 | Optimization implementation (SLSQP) |
| 6 | Neural network for advice prioritization |
| 7 | Streamlit dashboard and user study |
| 8-10 | Paper writing and final polish |

## Tech Stack

- **Language:** Python 3.10+
- **ML:** scikit-learn, PyTorch
- **Optimization:** SciPy
- **Dashboard:** Streamlit
- **Data:** pandas, numpy
- **Visualization:** matplotlib, seaborn, plotly

## Dataset

We use the [Kaggle Credit Card Transactions Fraud Detection](https://www.kaggle.com/datasets/kartik2112/fraud-detection) dataset:
- 1.3M transactions from 1,000 users over 2 years
- Used for training all ML models

## License

This project is for educational purposes as part of CS 5100 at Northeastern University.
