"""
train_model.py — trains a Linear Regression model on
House_Price_Prediction_Dataset.csv: preprocessing, feature engineering,
Linear Regression, and model evaluation (RMSE / MAE / R2). Saves the
trained model, along with everything needed to reproduce its preprocessing
at prediction time, into model.pkl.

Run this whenever you want to (re)train:
    python train_model.py
"""

import numpy as np
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

RANDOM_STATE = 42
CURRENT_YEAR = 2024
CONDITION_MAP = {'Poor': 0, 'Fair': 1, 'Good': 2, 'Excellent': 3}

print('Loading dataset...')
df = pd.read_csv('House_Price_Prediction_Dataset.csv').drop(columns=['Id'])

locations = sorted(df['Location'].unique().tolist())
conditions = list(CONDITION_MAP.keys())

# ---------------- preprocessing ----------------
df_clean = df.copy()
df_clean['Condition'] = df_clean['Condition'].map(CONDITION_MAP)
df_clean['Garage'] = (df_clean['Garage'] == 'Yes').astype(int)
df_clean = pd.get_dummies(df_clean, columns=['Location'], drop_first=True)

# ---------------- feature engineering ----------------
df_fe = df_clean.copy()
df_fe['house_age'] = CURRENT_YEAR - df_fe['YearBuilt']
df_fe['total_rooms'] = df_fe['Bedrooms'] + df_fe['Bathrooms']
df_fe['area_per_room'] = df_fe['Area'] / df_fe['total_rooms'].replace(0, 1)
df_fe['is_new'] = (df_fe['house_age'] <= 10).astype(int)
df_fe['quality_score'] = df_fe['Condition'] * df_fe['Floors']
df_fe.drop(columns=['YearBuilt'], inplace=True)

feature_cols = [c for c in df_fe.columns if c != 'Price']
X = df_fe[feature_cols]
y = df_fe['Price']

# ---------------- train/test split + scaling ----------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE
)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ---------------- Linear Regression ----------------
print('\nTraining Linear Regression...')
model = LinearRegression()
model.fit(X_train_scaled, y_train)

# ---------------- model evaluation ----------------
pred = model.predict(X_test_scaled)
rmse = np.sqrt(mean_squared_error(y_test, pred))
mae = mean_absolute_error(y_test, pred)
r2 = r2_score(y_test, pred)

print(f'  RMSE = {rmse:,.2f}')
print(f'  MAE  = {mae:,.2f}')
print(f'  R2   = {r2:.4f}')

# ---------------- save everything predict-time needs ----------------
bundle = {
    'model': model,
    'model_name': 'Linear Regression',
    'scaler': scaler,
    'feature_cols': feature_cols,
    'condition_map': CONDITION_MAP,
    'locations': locations,
    'current_year': CURRENT_YEAR,
    'test_r2': float(r2),
    'test_rmse': float(rmse),
    'test_mae': float(mae),
    'n_train': len(X_train),
    'n_test': len(X_test),
}

joblib.dump(bundle, 'model.pkl')
print('\nSaved model.pkl')
