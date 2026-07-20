"""
app.py — Flask web app for the House Price Prediction project.

Loads the trained model bundle (model.pkl, produced by train_model.py) and
serves:
  GET  /         -> input form (templates/index.html)
  POST /predict  -> runs the form input through the same preprocessing /
                     feature engineering the model was trained on, and
                     shows the prediction (templates/result.html)

Run:
    python train_model.py   # once, to create model.pkl
    python app.py
"""

import os
import pandas as pd
import joblib
from flask import Flask, render_template, request, redirect, url_for

app = Flask(__name__)

MODEL_PATH = 'model.pkl'
bundle = None


def load_bundle():
    global bundle
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"{MODEL_PATH} not found — run 'python train_model.py' first."
        )
    bundle = joblib.load(MODEL_PATH)


def build_feature_row(form):
    """Turn raw form input into the exact feature row the model expects."""
    condition_map = bundle['condition_map']
    current_year = bundle['current_year']
    feature_cols = bundle['feature_cols']

    row = pd.DataFrame([{
        'Area': float(form['area']),
        'Bedrooms': int(form['bedrooms']),
        'Bathrooms': int(form['bathrooms']),
        'Floors': int(form['floors']),
        'YearBuilt': int(form['yearBuilt']),
        'Location': form['location'],
        'Condition': form['condition'],
        'Garage': form['garage'],
    }])

    row['Condition'] = row['Condition'].map(condition_map)
    row['Garage'] = (row['Garage'] == 'Yes').astype(int)
    row = pd.get_dummies(row, columns=['Location'])

    row['house_age'] = current_year - row['YearBuilt']
    row['total_rooms'] = row['Bedrooms'] + row['Bathrooms']
    row['area_per_room'] = row['Area'] / row['total_rooms'].replace(0, 1)
    row['is_new'] = (row['house_age'] <= 10).astype(int)
    row['quality_score'] = row['Condition'] * row['Floors']
    row.drop(columns=['YearBuilt'], inplace=True)

    # align to the exact training columns (adds any missing one-hot
    # location columns as 0, drops nothing extra)
    row = row.reindex(columns=feature_cols, fill_value=0)
    return row


@app.route('/')
def index():
    return render_template(
        'index.html',
        locations=bundle['locations'],
        conditions=list(bundle['condition_map'].keys()),
        model_name=bundle['model_name'],
        test_r2=bundle['test_r2'],
        test_rmse=bundle['test_rmse'],
        test_mae=bundle['test_mae'],
        n_train=bundle['n_train'],
        n_test=bundle['n_test'],
        low_confidence=bundle['test_r2'] < 0.2,
    )


@app.route('/predict', methods=['POST'])
def predict():
    form = request.form
    row = build_feature_row(form)
    row_scaled = bundle['scaler'].transform(row)
    price = bundle['model'].predict(row_scaled)[0]

    summary = {
        'area': form['area'],
        'bedrooms': form['bedrooms'],
        'bathrooms': form['bathrooms'],
        'floors': form['floors'],
        'yearBuilt': form['yearBuilt'],
        'location': form['location'],
        'condition': form['condition'],
        'garage': form['garage'],
    }

    return render_template(
        'result.html',
        price=round(price),
        summary=summary,
        model_name=bundle['model_name'],
        test_r2=bundle['test_r2'],
        low_confidence=bundle['test_r2'] < 0.2,
    )


load_bundle()

if __name__ == '__main__':
    app.run(debug=True)
