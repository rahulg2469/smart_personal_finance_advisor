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
    rf_path = os.path.join(MODELS_PATH, 'random_forest_model.pkl')
    print(f"Looking for RF model at: {rf_path}")
    print(f"File exists: {os.path.exists(rf_path)}")
    rf_model = joblib.load(rf_path)
    RF_LOADED = True
    print("Loaded Random Forest model")
except Exception as e:
    RF_LOADED = False
    print(f"Random Forest model not found: {e}")


def categorize_transactions(transactions_df: pd.DataFrame) -> pd.DataFrame:
    """
    Use Random Forest to categorize transactions.
    Input: DataFrame with columns ['merchant', 'amt', 'lat', 'long', 'city_pop', 'merch_lat', 'merch_long', 'hour', 'day_of_week', 'month']
    Output: DataFrame with added 'predicted_category' column
    """
    if not RF_LOADED:
        raise Exception("Random Forest model not loaded")
    
    df = transactions_df.copy()
    
    # Feature engineering
    df['merchant_length'] = df['merchant'].str.len()
    df['merchant_word_count'] = df['merchant'].str.split().str.len()
    
    # Merchant encoding using hash (since we don't have the original LabelEncoder)
    # This approximates the encoding - may have some variance from training
    df['merchant_encoded'] = df['merchant'].apply(lambda x: hash(x.lower()) % 10000)
    
    # Set defaults for missing columns
    if 'hour' not in df.columns:
        df['hour'] = 12
    if 'day_of_week' not in df.columns:
        df['day_of_week'] = 3
    if 'month' not in df.columns:
        df['month'] = 4  # April
    if 'lat' not in df.columns:
        df['lat'] = 42.36  # Default: Boston area
    if 'long' not in df.columns:
        df['long'] = -71.06
    if 'city_pop' not in df.columns:
        df['city_pop'] = 700000  # Default city population
    if 'merch_lat' not in df.columns:
        df['merch_lat'] = df['lat']  # Use user lat as default
    if 'merch_long' not in df.columns:
        df['merch_long'] = df['long']  # Use user long as default
    
    # Feature columns in exact order used during training
    feature_cols = [
        'amt', 'lat', 'long', 'city_pop', 'merch_lat', 'merch_long',
        'hour', 'day_of_week', 'month', 'merchant_encoded',
        'merchant_length', 'merchant_word_count'
    ]
    
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
        # IMPORTANT: Order must match training order (alphabetical)
        training_order = [
            'entertainment', 'food_dining', 'gas_transport', 'grocery_net', 
            'grocery_pos', 'health_fitness', 'home', 'kids_pets', 
            'misc_net', 'misc_pos', 'personal_care', 'shopping_net', 
            'shopping_pos', 'travel'
        ]
        
        features = np.array([[
            (spending[cat] / total_spent * 100) for cat in training_order
        ]])
        
        # Scale and predict
        features_scaled = kmeans_scaler.transform(features)
        cluster = kmeans_model.predict(features_scaled)[0]
        
        print(f"DEBUG: Total spent=${total_spent:.2f}, Cluster={cluster}, Name={CLUSTER_NAMES.get(cluster, 'Unknown')}")
        
        return cluster, CLUSTER_NAMES.get(cluster, 'Unknown')
    else:
        # Fallback: simple heuristic
        pct = {cat: spending[cat] / total_spent * 100 for cat in CATEGORIES}
        max_category_pct = max(pct.values())
        
        if max_category_pct > 35:
            return 2, 'Inconsistent'
        elif total_spent > 6000:
            return 1, 'High Spender'
        elif total_spent < 3000:
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
    Returns list of (category, difficulty) using Neural Network model.
    """
    total = sum(spending.values())
    if total == 0:
        return [(cat, 'Medium') for cat in CATEGORIES]
    
    if NN_LOADED:
        try:
            # Use actual Neural Network model
            num_transactions = 100  # Estimate
            
            user_features = {
                'total_transactions': num_transactions,
                'total_spending': total,
                'avg_transaction': total / num_transactions,
                'std_transaction': (total / num_transactions) * 0.5,
                'min_transaction': min([s for s in spending.values() if s > 0], default=0),
                'max_transaction': max(spending.values()),
                'weekend_pct': 0.3,
                'avg_hour': 14,
                'avg_day_of_week': 3,
                'spending_range': max(spending.values()) - min([s for s in spending.values() if s > 0], default=0),
                'coefficient_of_variation': 0.5
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
            
            # Assign difficulty based on position in sorted list
            # First 5 = Easy, Next 5 = Medium, Last 4 = Hard
            result = []
            for i, (cat, score) in enumerate(priority_list):
                if i < 5:
                    difficulty = 'Easy'
                elif i < 10:
                    difficulty = 'Medium'
                else:
                    difficulty = 'Hard'
                result.append((cat, difficulty))
            
            return result
        except Exception as e:
            print(f"NN Error: {e}")
            # Fallback on error
            return [(cat, 'Medium') for cat in CATEGORIES]
    else:
        # Fallback if NN not loaded
        return [(cat, 'Medium') for cat in CATEGORIES]

def detect_anomalies(spending: dict) -> list:
    """
    Detect unusual spending patterns using Isolation Forest model.
    Returns list of (category, reason) for anomalies.
    """
    total = sum(spending.values())
    if total == 0:
        return []
    
    anomalies = []
    
    if ISO_LOADED:
        try:
            # Use Isolation Forest model
            # Prepare features - spending percentages
            features = np.array([[
                (spending[cat] / total * 100) for cat in CATEGORIES
            ]])
            
            # Scale features
            features_scaled = iso_scaler.transform(features)
            
            # Predict (-1 = anomaly, 1 = normal)
            prediction = iso_model.predict(features_scaled)[0]
            anomaly_score = iso_model.decision_function(features_scaled)[0]
            
            if prediction == -1:
                # Find which categories are unusual
                for cat in CATEGORIES:
                    pct = spending[cat] / total * 100
                    if pct > 25:  # High percentage
                        anomalies.append((cat, f"Unusually high ({pct:.1f}% of budget)"))
                    elif spending[cat] > 1000 and pct > 15:
                        anomalies.append((cat, f"High spending (${spending[cat]:.2f}, {pct:.1f}% of budget)"))
            
            return anomalies
        except Exception as e:
            print(f"Isolation Forest Error: {e}")
    
    # Fallback: rule-based detection
    typical_ranges = {
        'grocery_pos': (10, 25), 'grocery_net': (2, 10), 'entertainment': (5, 15),
        'shopping_pos': (5, 20), 'shopping_net': (3, 15), 'food_dining': (5, 15),
        'gas_transport': (8, 20), 'health_fitness': (3, 12), 'home': (5, 20),
        'kids_pets': (3, 15), 'misc_net': (2, 10), 'misc_pos': (2, 10),
        'personal_care': (3, 10), 'travel': (0, 15)
    }
    
    for cat in CATEGORIES:
        pct = spending[cat] / total * 100
        low, high = typical_ranges.get(cat, (0, 100))
        if pct > high * 1.5:
            anomalies.append((cat, f"Unusually high ({pct:.1f}% of budget, typical max is {high}%)"))
    
    return anomalies

def predict_next_month(spending: dict) -> float:
    """
    Predict next month's spending using Gradient Boosting model.
    """
    total = sum(spending.values())
    if total == 0:
        return None
    
    if GB_LOADED:
        try:
            # Prepare features for Gradient Boosting
            # Build feature dict matching training
            features_dict = {}
            for cat in CATEGORIES:
                features_dict[cat] = spending.get(cat, 0)
                features_dict[f'{cat}_pct'] = (spending.get(cat, 0) / total * 100) if total > 0 else 0
            
            features_dict['total_spending'] = total
            features_dict['num_categories'] = sum(1 for v in spending.values() if v > 0)
            features_dict['avg_category_spend'] = total / 14
            features_dict['max_category_spend'] = max(spending.values())
            features_dict['min_category_spend'] = min([v for v in spending.values() if v > 0], default=0)
            
            # Build feature array in correct order
            feature_array = np.array([[features_dict.get(f, 0) for f in gb_features]])
            
            # Predict
            prediction = gb_model.predict(feature_array)[0]
            return max(0, prediction)
        except Exception as e:
            print(f"Gradient Boosting Error: {e}")
    
    # Fallback: return current total with small adjustment
    return total * 1.02  # Assume 2% increase