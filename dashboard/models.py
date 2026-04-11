import numpy as np
import pandas as pd
from scipy.optimize import minimize
import joblib
import torch
import torch.nn as nn
import os

# Categories
CATEGORIES = [
    'grocery_pos', 'grocery_net', 'entertainment', 'shopping_pos', 
    'shopping_net', 'food_dining', 'gas_transport', 'health_fitness',
    'home', 'kids_pets', 'misc_net', 'misc_pos', 'personal_care', 'travel'
]

# Get the project root directory
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_PATH = os.path.join(PROJECT_ROOT, 'models')

# Load SLSQP category config from pkl file
try:
    CATEGORY_CONFIG = joblib.load(os.path.join(MODELS_PATH, 'slsqp_category_config.pkl'))
    print("Loaded SLSQP config from pkl file")
except:
    # Fallback to hardcoded config if file not found
    CATEGORY_CONFIG = {
        'grocery_pos': {'min_pct': 0.70, 'priority': 'essential'},
        'grocery_net': {'min_pct': 0.70, 'priority': 'essential'},
        'gas_transport': {'min_pct': 0.60, 'priority': 'essential'},
        'health_fitness': {'min_pct': 0.50, 'priority': 'important'},
        'home': {'min_pct': 0.80, 'priority': 'essential'},
        'kids_pets': {'min_pct': 0.60, 'priority': 'important'},
        'personal_care': {'min_pct': 0.50, 'priority': 'flexible'},
        'food_dining': {'min_pct': 0.30, 'priority': 'flexible'},
        'entertainment': {'min_pct': 0.20, 'priority': 'flexible'},
        'shopping_net': {'min_pct': 0.20, 'priority': 'flexible'},
        'shopping_pos': {'min_pct': 0.20, 'priority': 'flexible'},
        'travel': {'min_pct': 0.10, 'priority': 'flexible'},
        'misc_net': {'min_pct': 0.30, 'priority': 'flexible'},
        'misc_pos': {'min_pct': 0.30, 'priority': 'flexible'}
    }
    print("Using fallback SLSQP config")

# Load K-Means model
try:
    kmeans_model = joblib.load(os.path.join(MODELS_PATH, 'kmeans_model.pkl'))
    kmeans_scaler = joblib.load(os.path.join(MODELS_PATH, 'kmeans_scaler.pkl'))
    KMEANS_LOADED = True
    print("Loaded K-Means model")
except:
    KMEANS_LOADED = False
    print("K-Means model not found, using fallback")

# Load Isolation Forest model
try:
    iso_model = joblib.load(os.path.join(MODELS_PATH, 'isolation_forest_model_v2.pkl'))
    iso_scaler = joblib.load(os.path.join(MODELS_PATH, 'isolation_forest_scaler_v2.pkl'))
    ISO_LOADED = True
    print("Loaded Isolation Forest model")
except:
    ISO_LOADED = False
    print("Isolation Forest model not found, using fallback")

# Load Gradient Boosting model
try:
    gb_model = joblib.load(os.path.join(MODELS_PATH, 'gradient_boosting_model_v3.pkl'))
    gb_features = joblib.load(os.path.join(MODELS_PATH, 'gb_feature_columns_v3.pkl'))
    GB_LOADED = True
    print("Loaded Gradient Boosting model")
except:
    GB_LOADED = False
    print("Gradient Boosting model not found, using fallback")

# Neural Network architecture (must match training)
class PrioritizationNetwork(nn.Module):
    def __init__(self, input_size=25, output_size=14):
        super(PrioritizationNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 32)
        self.fc4 = nn.Linear(32, output_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.relu(self.fc3(x))
        x = self.fc4(x)
        return x

# Load Neural Network model
try:
    nn_checkpoint = torch.load(os.path.join(MODELS_PATH, 'neural_network_model.pth'), map_location='cpu')
    nn_model = PrioritizationNetwork(nn_checkpoint['input_size'], nn_checkpoint['output_size'])
    nn_model.load_state_dict(nn_checkpoint['model_state_dict'])
    nn_model.eval()
    nn_scaler = joblib.load(os.path.join(MODELS_PATH, 'scaler.pkl'))
    NN_LOADED = True
    print("Loaded Neural Network model")
except:
    NN_LOADED = False
    print("Neural Network model not found, using fallback")

# Load Random Forest model
try:
    rf_model = joblib.load(os.path.join(MODELS_PATH, 'random_forest_model.pkl'))
    RF_LOADED = True
    print("Loaded Random Forest model")
except:
    RF_LOADED = False
    print("Random Forest model not found")


def categorize_transactions(transactions_df: pd.DataFrame) -> pd.DataFrame:
    """
    Use Random Forest to categorize transactions.
    Input: DataFrame with columns ['merchant', 'amt', 'hour', 'day_of_week', etc.]
    Output: DataFrame with added 'predicted_category' column
    """
    if not RF_LOADED:
        raise Exception("Random Forest model not loaded")
    
    df = transactions_df.copy()
    
    # Feature engineering (same as Random_Forest.ipynb)
    # Merchant encoding
    df['merchant_length'] = df['merchant'].str.len()
    df['merchant_word_count'] = df['merchant'].str.split().str.len()
    
    # If hour/day not provided, use defaults
    if 'hour' not in df.columns:
        df['hour'] = 12
    if 'day_of_week' not in df.columns:
        df['day_of_week'] = 3
    if 'is_weekend' not in df.columns:
        df['is_weekend'] = 0
    
    # Encode merchant (simple hash for now)
    df['merchant_encoded'] = df['merchant'].apply(lambda x: hash(x.lower()) % 10000)
    
    # Select features (must match training)
    feature_cols = ['amt', 'hour', 'day_of_week', 'is_weekend', 
                    'merchant_encoded', 'merchant_length', 'merchant_word_count']
    
    # Add missing columns with defaults
    for col in feature_cols:
        if col not in df.columns:
            df[col] = 0
    
    X = df[feature_cols]
    
    # Predict
    df['predicted_category'] = rf_model.predict(X)
    
    return df


def summarize_by_category(transactions_df: pd.DataFrame) -> dict:
    """
    Sum transaction amounts by category.
    Returns dict of {category: total_amount}
    """
    if 'predicted_category' not in transactions_df.columns:
        transactions_df = categorize_transactions(transactions_df)
    
    summary = transactions_df.groupby('predicted_category')['amt'].sum().to_dict()
    
    # Ensure all categories exist
    result = {cat: 0.0 for cat in CATEGORIES}
    for cat, amt in summary.items():
        if cat in result:
            result[cat] = float(amt)
    
    return result

# Cluster names mapping
CLUSTER_NAMES = {
    0: 'Average',
    1: 'High Spender', 
    2: 'Inconsistent',
    3: 'Budget-Conscious'
}

def get_user_cluster(spending: dict) -> tuple:
    """
    Determine user's cluster based on spending pattern.
    Returns (cluster_id, cluster_name)
    """
    total_spent = sum(spending.values())
    
    if total_spent == 0:
        return 0, 'Average'
    
    if KMEANS_LOADED:
        # Use actual K-Means model
        # K-Means was trained on category spending PERCENTAGES (14 features)
        features = np.array([[
            (spending[cat] / total_spent * 100) for cat in CATEGORIES
        ]])
        
        # Scale and predict
        features_scaled = kmeans_scaler.transform(features)
        cluster = kmeans_model.predict(features_scaled)[0]
        
        return cluster, CLUSTER_NAMES.get(cluster, 'Unknown')
    else:
        # Fallback: simple heuristic
        pct = {cat: spending[cat] / total_spent * 100 for cat in CATEGORIES}
        max_category_pct = max(pct.values())
        
        if max_category_pct > 40:
            return 2, 'Inconsistent'
        elif total_spent > 8000:
            return 1, 'High Spender'
        elif total_spent < 4000:
            return 3, 'Budget-Conscious'
        else:
            return 0, 'Average'

def get_budget_recommendations(spending: dict, savings_goal: float) -> list:
    """
    Use SLSQP optimization to calculate budget cuts.
    Returns list of recommendations.
    """
    categories = list(spending.keys())
    current_values = np.array([spending[cat] for cat in categories])
    current_total = current_values.sum()
    target_total = current_total - savings_goal
    
    if target_total <= 0:
        return []
    
    if target_total >= current_total:
        return []
    
    # Priority weights
    priority_weights = {'essential': 10.0, 'important': 5.0, 'flexible': 1.0}
    weights = np.array([priority_weights[CATEGORY_CONFIG[cat]['priority']] for cat in categories])
    normalized_weights = weights / (current_values + 1)
    
    def objective(x):
        return np.sum(normalized_weights * (x - current_values) ** 2)
    
    def constraint_total(x):
        return np.sum(x) - target_total
    
    bounds = [
        (spending[cat] * CATEGORY_CONFIG[cat]['min_pct'], spending[cat]) 
        for cat in categories
    ]
    
    reduction_ratio = target_total / current_total
    x0 = current_values * reduction_ratio
    
    result = minimize(
        objective, x0, method='SLSQP',
        bounds=bounds,
        constraints={'type': 'eq', 'fun': constraint_total},
        options={'maxiter': 1000}
    )
    
    # Build recommendations
    recommendations = []
    for i, cat in enumerate(categories):
        cut = spending[cat] - result.x[i]
        if cut > 1:  # Only show meaningful cuts
            recommendations.append({
                'category': cat,
                'current': spending[cat],
                'recommended': result.x[i],
                'savings': cut,
                'pct_cut': (cut / spending[cat] * 100) if spending[cat] > 0 else 0
            })
    
    # Sort by savings amount
    recommendations.sort(key=lambda x: -x['savings'])
    
    return recommendations

def get_priority_advice(spending: dict) -> list:
    """
    Determine priority order for budget cuts based on spending patterns.
    Returns list of (category, difficulty) sorted by easiest first.
    """
    total = sum(spending.values())
    if total == 0:
        return [(cat, 'N/A') for cat in CATEGORIES]
    
    if NN_LOADED:
        # Use actual Neural Network model
        # Prepare features (same as in Neural_Networks.ipynb)
        num_transactions = 100  # Estimate
        
        user_features = {
            'total_transactions': num_transactions,
            'total_spending': total,
            'avg_transaction': total / num_transactions,
            'std_transaction': (total / num_transactions) * 0.5,
            'min_transaction': min([s for s in spending.values() if s > 0], default=0),
            'max_transaction': max(spending.values()),
            'weekend_pct': 0.3,  # Estimate
            'avg_hour': 14,  # Estimate
            'avg_day_of_week': 3,  # Estimate
            'spending_range': max(spending.values()) - min([s for s in spending.values() if s > 0], default=0),
            'coefficient_of_variation': 0.5  # Estimate
        }
        
        # Add category percentages
        for cat in CATEGORIES:
            user_features[cat] = (spending[cat] / total * 100) if total > 0 else 0
        
        # Get feature names from checkpoint and build array
        feature_names = nn_checkpoint.get('feature_names', list(user_features.keys()))
        feature_array = np.array([user_features.get(f, 0) for f in feature_names]).reshape(1, -1)
        
        # Scale and predict
        feature_scaled = nn_scaler.transform(feature_array)
        feature_tensor = torch.FloatTensor(feature_scaled)
        
        with torch.no_grad():
            priorities = nn_model(feature_tensor).numpy()[0]
        
        # Map scores to categories
        category_names = nn_checkpoint.get('category_names', CATEGORIES)
        priority_list = list(zip(category_names, priorities))
        priority_list.sort(key=lambda x: -x[1])  # Higher score = easier to cut
        
        # Convert to difficulty labels
        result = []
        for cat, score in priority_list:
            if score > 7:
                difficulty = 'Easy'
            elif score > 4:
                difficulty = 'Medium'
            else:
                difficulty = 'Hard'
            result.append((cat, difficulty))
        
        return result
    else:
        # Fallback: simple rules
        priorities = []
        
        for cat in CATEGORIES:
            pct = spending[cat] / total * 100
            config = CATEGORY_CONFIG[cat]
            
            if config['priority'] == 'flexible':
                base_score = 3
            elif config['priority'] == 'important':
                base_score = 6
            else:
                base_score = 9
            
            if pct > 20:
                base_score += 1
            elif pct < 5:
                base_score -= 1
            
            if base_score <= 3:
                difficulty = 'Easy'
            elif base_score <= 6:
                difficulty = 'Medium'
            else:
                difficulty = 'Hard'
            
            priorities.append((cat, difficulty, base_score))
        
        priorities.sort(key=lambda x: x[2])
        return [(cat, diff) for cat, diff, _ in priorities]

def detect_anomalies(spending: dict) -> list:
    """
    Detect unusual spending patterns.
    Returns list of (category, reason) for anomalies.
    """
    total = sum(spending.values())
    if total == 0:
        return []
    
    anomalies = []
    
    # Typical spending percentages (from dataset analysis)
    typical_ranges = {
        'grocery_pos': (10, 25),
        'grocery_net': (2, 10),
        'entertainment': (5, 15),
        'shopping_pos': (5, 20),
        'shopping_net': (3, 15),
        'food_dining': (5, 15),
        'gas_transport': (8, 20),
        'health_fitness': (3, 12),
        'home': (5, 20),
        'kids_pets': (3, 15),
        'misc_net': (2, 10),
        'misc_pos': (2, 10),
        'personal_care': (3, 10),
        'travel': (0, 15)
    }
    
    for cat in CATEGORIES:
        pct = spending[cat] / total * 100
        low, high = typical_ranges.get(cat, (0, 100))
        
        if pct > high * 1.5:  # More than 50% above typical max
            anomalies.append((cat, f"Unusually high ({pct:.1f}% of budget, typical max is {high}%)"))
        elif spending[cat] > 1000 and pct > high:  # Large absolute amount and above typical
            anomalies.append((cat, f"High spending (${spending[cat]:.2f}, {pct:.1f}% of budget)"))
    
    return anomalies

def predict_next_month(spending_history: list) -> float:
    """
    Predict next month's spending based on history.
    Requires at least 2 months of data.
    """
    if len(spending_history) < 2:
        return None
    
    # Simple prediction: weighted average of recent months
    totals = [sum(month.values()) for month in spending_history]
    
    # More recent months have higher weight
    weights = list(range(1, len(totals) + 1))
    weighted_avg = sum(t * w for t, w in zip(totals, weights)) / sum(weights)
    
    # Add slight trend adjustment
    if len(totals) >= 3:
        recent_trend = (totals[-1] - totals[-3]) / 2
        weighted_avg += recent_trend * 0.5
    
    return max(0, weighted_avg)