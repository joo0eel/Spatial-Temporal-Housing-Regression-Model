# """
# streamlit_app.py — House Price Prediction (Streamlit + Linear Regression).

# Same pipeline as app.py (the Flask version): load data, preprocess,
# engineer features, train Linear Regression, evaluate, then let the user
# enter property details and get a live prediction — just built with
# Streamlit's widgets instead of Flask + HTML templates, so it can be
# deployed for free on Streamlit Community Cloud.

# Run locally:
#     streamlit run streamlit_app.py

# Deploy: push this repo to GitHub, then deploy at https://share.streamlit.io
# (see the "How to deploy" notes at the bottom of this file).
# """

# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# import streamlit as st

# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import StandardScaler
# from sklearn.linear_model import LinearRegression
# from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# # ============================================================
# # CONFIG
# # ============================================================
# RANDOM_STATE = 42
# CURRENT_YEAR = 2024
# CONDITION_MAP = {'Poor': 0, 'Fair': 1, 'Good': 2, 'Excellent': 3}
# DATA_PATH = 'House_Price_Prediction_Dataset.csv'

# st.set_page_config(page_title='House Price Appraisal', page_icon='🏠', layout='wide')


# # ============================================================
# # TRAIN THE MODEL ONCE (cached — reused across every user interaction,
# # not retrained on every click/rerun)
# # ============================================================
# @st.cache_resource
# def train():
#     df = pd.read_csv(DATA_PATH).drop(columns=['Id'])
#     locations = sorted(df['Location'].unique().tolist())

#     # --- preprocessing ---
#     df_clean = df.copy()
#     df_clean['Condition'] = df_clean['Condition'].map(CONDITION_MAP)
#     df_clean['Garage'] = (df_clean['Garage'] == 'Yes').astype(int)
#     df_clean = pd.get_dummies(df_clean, columns=['Location'], drop_first=True)

#     # --- feature engineering ---
#     df_fe = df_clean.copy()
#     df_fe['house_age'] = CURRENT_YEAR - df_fe['YearBuilt']
#     df_fe['total_rooms'] = df_fe['Bedrooms'] + df_fe['Bathrooms']
#     df_fe['area_per_room'] = df_fe['Area'] / df_fe['total_rooms'].replace(0, 1)
#     df_fe['is_new'] = (df_fe['house_age'] <= 10).astype(int)
#     df_fe['quality_score'] = df_fe['Condition'] * df_fe['Floors']
#     df_fe.drop(columns=['YearBuilt'], inplace=True)

#     feature_cols = [c for c in df_fe.columns if c != 'Price']
#     X = df_fe[feature_cols]
#     y = df_fe['Price']

#     # --- split + scale ---
#     X_train, X_test, y_train, y_test = train_test_split(
#         X, y, test_size=0.2, random_state=RANDOM_STATE
#     )
#     scaler = StandardScaler()
#     X_train_scaled = scaler.fit_transform(X_train)
#     X_test_scaled = scaler.transform(X_test)

#     # --- train Linear Regression ---
#     model = LinearRegression()
#     model.fit(X_train_scaled, y_train)

#     # --- evaluate ---
#     pred = model.predict(X_test_scaled)
#     metrics = {
#         'rmse': float(np.sqrt(mean_squared_error(y_test, pred))),
#         'mae': float(mean_absolute_error(y_test, pred)),
#         'r2': float(r2_score(y_test, pred)),
#         'n_train': len(X_train),
#         'n_test': len(X_test),
#     }

#     return {
#         'model': model,
#         'scaler': scaler,
#         'feature_cols': feature_cols,
#         'locations': locations,
#         'metrics': metrics,
#         'df': df,
#         'y_test': y_test,
#         'pred': pred,
#     }


# def build_feature_row(state, area, bedrooms, bathrooms, floors, year_built, location, condition, garage):
#     row = pd.DataFrame([{
#         'Area': area, 'Bedrooms': bedrooms, 'Bathrooms': bathrooms, 'Floors': floors,
#         'YearBuilt': year_built, 'Location': location, 'Condition': condition, 'Garage': garage,
#     }])
#     row['Condition'] = row['Condition'].map(CONDITION_MAP)
#     row['Garage'] = (row['Garage'] == 'Yes').astype(int)
#     row = pd.get_dummies(row, columns=['Location'])

#     row['house_age'] = CURRENT_YEAR - row['YearBuilt']
#     row['total_rooms'] = row['Bedrooms'] + row['Bathrooms']
#     row['area_per_room'] = row['Area'] / row['total_rooms'].replace(0, 1)
#     row['is_new'] = (row['house_age'] <= 10).astype(int)
#     row['quality_score'] = row['Condition'] * row['Floors']
#     row.drop(columns=['YearBuilt'], inplace=True)

#     return row.reindex(columns=state['feature_cols'], fill_value=0)


# state = train()

# # ============================================================
# # UI
# # ============================================================
# st.title('🏠 House Price Appraisal')
# st.caption('Linear Regression model trained on `House_Price_Prediction_Dataset.csv`')

# col_form, col_spec = st.columns([1.3, 1])

# with col_form:
#     st.subheader('Property Specification')
#     with st.form('predict_form'):
#         area = st.number_input('Area (sqft)', min_value=200, max_value=20000, value=2200)
#         c1, c2 = st.columns(2)
#         bedrooms = c1.number_input('Bedrooms', min_value=0, max_value=20, value=3)
#         bathrooms = c2.number_input('Bathrooms', min_value=0, max_value=20, value=2)
#         c3, c4 = st.columns(2)
#         floors = c3.number_input('Floors', min_value=1, max_value=10, value=2)
#         year_built = c4.number_input('Year built', min_value=1800, max_value=2026, value=2005)
#         location = st.selectbox('Location', state['locations'])
#         condition = st.selectbox('Condition', list(CONDITION_MAP.keys()), index=2)
#         garage = st.selectbox('Garage', ['Yes', 'No'])
#         submitted = st.form_submit_button('Get Appraisal')

#     if submitted:
#         row = build_feature_row(state, area, bedrooms, bathrooms, floors, year_built, location, condition, garage)
#         row_scaled = state['scaler'].transform(row)
#         price = state['model'].predict(row_scaled)[0]

#         st.success(f'### Predicted Price: ${price:,.0f}')
#         st.caption(
#             f'{area} sqft · {bedrooms}bd/{bathrooms}ba · {floors} floor(s) · '
#             f'built {year_built} · {location} · {condition} condition · garage: {garage}'
#         )
#         if state['metrics']['r2'] < 0.2:
#             st.warning(
#                 'LOW CONFIDENCE — this model\'s test R² is near zero, meaning these '
#                 'features barely explain price in this dataset. Treat this as a '
#                 'demonstration of the pipeline, not a reliable appraisal.'
#             )

# with col_spec:
#     st.subheader('Model Spec')
#     m = state['metrics']
#     st.metric('Test R²', f"{m['r2']:.4f}")
#     st.metric('Test RMSE', f"${m['rmse']:,.0f}")
#     st.metric('Test MAE', f"${m['mae']:,.0f}")
#     st.caption(f"Trained on {m['n_train']} rows · tested on {m['n_test']} rows")

#     st.subheader('Price Distribution & Fit')
#     fig, axes = plt.subplots(1, 2, figsize=(9, 3.6))
#     axes[0].hist(state['df']['Price'], bins=40, color='#16436b', edgecolor='#dcecf9')
#     axes[0].set_title('Price Distribution', fontsize=10)
#     axes[0].set_xlabel('Price', fontsize=8)
#     axes[0].set_ylabel('Count', fontsize=8)

#     axes[1].scatter(state['y_test'], state['pred'], alpha=0.4, s=10, color='#c9973a')
#     lims = [min(state['y_test'].min(), state['pred'].min()), max(state['y_test'].max(), state['pred'].max())]
#     axes[1].plot(lims, lims, 'r--', linewidth=1, label='Perfect prediction')
#     axes[1].set_title('Actual vs. Predicted', fontsize=10)
#     axes[1].set_xlabel('Actual price', fontsize=8)
#     axes[1].set_ylabel('Predicted price', fontsize=8)
#     axes[1].legend(fontsize=7)
#     plt.tight_layout()
#     st.pyplot(fig)

# # ============================================================
# # HOW TO DEPLOY (Streamlit Community Cloud — free)
# # ============================================================
# # 1. Push this whole project folder to a public (or private) GitHub repo.
# #    Make sure these are included: streamlit_app.py, requirements.txt,
# #    House_Price_Prediction_Dataset.csv
# # 2. Go to https://share.streamlit.io and sign in with GitHub.
# # 3. Click "New app" -> pick your repo/branch -> set "Main file path"
# #    to streamlit_app.py -> Deploy.
# # 4. You'll get a public URL like https://<your-app-name>.streamlit.app
# #    that anyone can open from anywhere, no install needed.




"""
streamlit_app.py — House Price Prediction (Streamlit + Linear Regression)

Two-page flow, themed to match the blueprint/appraisal-ledger design used
in the Flask version:
  - Page 1 ("form"):   property specification form + model spec panel + charts
  - Page 2 ("result"): predicted price, styled like a stamped appraisal ticket

Run locally:
    streamlit run streamlit_app.py

Deploy: push this repo to GitHub, then deploy at https://share.streamlit.io
(Main file path: streamlit_app.py)
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# ============================================================
# CONFIG
# ============================================================
RANDOM_STATE = 42
CURRENT_YEAR = 2024
CONDITION_MAP = {'Poor': 0, 'Fair': 1, 'Good': 2, 'Excellent': 3}
DATA_PATH = 'House_Price_Prediction_Dataset.csv'
IMAGE_PATH = 'image.jpg'

st.set_page_config(page_title='House Price Appraisal', page_icon='🏠', layout='wide')

# ============================================================
# THEME — custom CSS to approximate the blueprint/appraisal-ledger look
# ============================================================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;700&family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500;600&display=swap');

:root {
  --blueprint-bg: #10314f;
  --blueprint-bg-2: #16436b;
  --panel-bg: #f3f6f2;
  --line-white: #dcecf9;
  --ink: #0d2036;
  --ink-soft: #3c5670;
  --brass: #c9973a;
  --brass-bright: #e3b458;
  --danger: #b5502f;
}

.stApp {
  background:
    linear-gradient(rgba(220,236,249,0.07) 1px, transparent 1px) 0 0/20px 20px,
    linear-gradient(90deg, rgba(220,236,249,0.07) 1px, transparent 1px) 0 0/20px 20px,
    linear-gradient(180deg, var(--blueprint-bg-2), var(--blueprint-bg));
  font-family: 'IBM Plex Sans', sans-serif;
}

/* headings */
h1, h2, h3 { font-family: 'Space Grotesk', sans-serif !important; color: var(--line-white); }
.block-container h1 { font-weight: 700 !important; }

/* eyebrow / mono labels */
.eyebrow {
  font-family: 'IBM Plex Mono', monospace;
  font-size: 0.75rem;
  letter-spacing: 0.14em;
  text-transform: uppercase;
  color: var(--brass-bright);
  margin-bottom: 6px;
}
.mono { font-family: 'IBM Plex Mono', monospace; color: var(--brass-bright); }

/* panel containers (st.container(border=True)) */
div[data-testid="stVerticalBlockBorderWrapper"] {
  background: var(--panel-bg) !important;
  border-radius: 4px !important;
  border: 1px solid rgba(13,32,54,0.15) !important;
  padding: 6px;
}
div[data-testid="stVerticalBlockBorderWrapper"] * {
  color: var(--ink);
}
div[data-testid="stVerticalBlockBorderWrapper"] h2,
div[data-testid="stVerticalBlockBorderWrapper"] h3 {
  color: var(--ink) !important;
  font-weight: 600 !important;
  border-bottom: 2px solid var(--ink);
  padding-bottom: 8px;
}

/* labels */
label, .stMarkdown p { font-family: 'IBM Plex Sans', sans-serif; }

/* buttons */
.stButton > button, .stFormSubmitButton > button {
  background: var(--brass) !important;
  color: var(--blueprint-bg) !important;
  font-family: 'Space Grotesk', sans-serif !important;
  font-weight: 600 !important;
  border: none !important;
  border-radius: 4px !important;
  width: 100%;
}
.stButton > button:hover, .stFormSubmitButton > button:hover {
  background: var(--brass-bright) !important;
}

/* metrics */
div[data-testid="stMetric"] {
  background: rgba(13,32,54,0.04);
  border-radius: 4px;
  padding: 8px 10px;
}
div[data-testid="stMetricLabel"] { font-family: 'IBM Plex Mono', monospace; color: var(--ink-soft) !important; }
div[data-testid="stMetricValue"] { color: var(--ink) !important; font-family: 'Space Grotesk', sans-serif; }

/* stamp + result value (used on the result page) */
.stamp {
  display: inline-block;
  font-family: 'IBM Plex Mono', monospace;
  font-size: 0.75rem;
  letter-spacing: 0.12em;
  text-transform: uppercase;
  color: var(--danger);
  border: 1.5px solid var(--danger);
  border-radius: 999px;
  padding: 4px 14px;
  transform: rotate(-2deg);
  margin-bottom: 10px;
}
.result-value {
  font-family: 'Space Grotesk', sans-serif;
  font-weight: 700;
  font-size: 3rem;
  color: var(--ink);
}
.result-note {
  font-family: 'IBM Plex Mono', monospace;
  font-size: 0.82rem;
  color: var(--ink-soft);
  line-height: 1.6;
}
.spec-note {
  margin-top: 14px;
  padding: 12px 14px;
  background: #eaeee4;
  border-left: 3px solid var(--danger);
  font-family: 'IBM Plex Mono', monospace;
  font-size: 0.8rem;
  line-height: 1.55;
  color: var(--ink);
  border-radius: 0 4px 4px 0;
}

.site-footer {
  font-family: 'IBM Plex Mono', monospace;
  font-size: 0.75rem;
  color: rgba(220,236,249,0.55);
  margin-top: 24px;
}
</style>
""", unsafe_allow_html=True)


# ============================================================
# TRAIN THE MODEL ONCE (cached)
# ============================================================
@st.cache_resource
def train():
    df = pd.read_csv(DATA_PATH).drop(columns=['Id'])
    locations = sorted(df['Location'].unique().tolist())

    df_clean = df.copy()
    df_clean['Condition'] = df_clean['Condition'].map(CONDITION_MAP)
    df_clean['Garage'] = (df_clean['Garage'] == 'Yes').astype(int)
    df_clean = pd.get_dummies(df_clean, columns=['Location'], drop_first=True)

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

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE
    )
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = LinearRegression()
    model.fit(X_train_scaled, y_train)

    pred = model.predict(X_test_scaled)
    metrics = {
        'rmse': float(np.sqrt(mean_squared_error(y_test, pred))),
        'mae': float(mean_absolute_error(y_test, pred)),
        'r2': float(r2_score(y_test, pred)),
        'n_train': len(X_train),
        'n_test': len(X_test),
    }

    return {
        'model': model, 'scaler': scaler, 'feature_cols': feature_cols,
        'locations': locations, 'metrics': metrics, 'df': df,
        'y_test': y_test, 'pred': pred,
    }


def build_feature_row(state, area, bedrooms, bathrooms, floors, year_built, location, condition, garage):
    row = pd.DataFrame([{
        'Area': area, 'Bedrooms': bedrooms, 'Bathrooms': bathrooms, 'Floors': floors,
        'YearBuilt': year_built, 'Location': location, 'Condition': condition, 'Garage': garage,
    }])
    row['Condition'] = row['Condition'].map(CONDITION_MAP)
    row['Garage'] = (row['Garage'] == 'Yes').astype(int)
    row = pd.get_dummies(row, columns=['Location'])

    row['house_age'] = CURRENT_YEAR - row['YearBuilt']
    row['total_rooms'] = row['Bedrooms'] + row['Bathrooms']
    row['area_per_room'] = row['Area'] / row['total_rooms'].replace(0, 1)
    row['is_new'] = (row['house_age'] <= 10).astype(int)
    row['quality_score'] = row['Condition'] * row['Floors']
    row.drop(columns=['YearBuilt'], inplace=True)

    return row.reindex(columns=state['feature_cols'], fill_value=0)


state = train()

if 'page' not in st.session_state:
    st.session_state.page = 'form'
if 'last_prediction' not in st.session_state:
    st.session_state.last_prediction = None


# ============================================================
# PAGE 1 — FORM
# ============================================================
def render_form_page():
    hero_left, hero_right = st.columns([1.3, 1])
    with hero_left:
        st.markdown('<p class="eyebrow">Regression-based estimate &middot; Streamlit + scikit-learn</p>', unsafe_allow_html=True)
        st.markdown('# House Price\nAppraisal')
        st.markdown(
            f'Enter a property\'s specs and get an estimate from a '
            f'<span class="mono">Linear Regression</span> model trained on '
            f'<span class="mono">House_Price_Prediction_Dataset.csv</span>.',
            unsafe_allow_html=True,
        )
    with hero_right:
        if os.path.exists(IMAGE_PATH):
            st.image(IMAGE_PATH, use_container_width=True)

    st.write('')
    col_form, col_spec = st.columns([1.3, 1])

    with col_form:
        with st.container(border=True):
            st.markdown('<p class="eyebrow">FORM A</p>', unsafe_allow_html=True)
            st.markdown('### Property Specification')
            with st.form('predict_form'):
                area = st.number_input('Area (sqft)', min_value=200, max_value=20000, value=2200)
                c1, c2 = st.columns(2)
                bedrooms = c1.number_input('Bedrooms', min_value=0, max_value=20, value=3)
                bathrooms = c2.number_input('Bathrooms', min_value=0, max_value=20, value=2)
                c3, c4 = st.columns(2)
                floors = c3.number_input('Floors', min_value=1, max_value=10, value=2)
                year_built = c4.number_input('Year built', min_value=1800, max_value=2026, value=2005)
                location = st.selectbox('Location', state['locations'])
                condition = st.selectbox('Condition', list(CONDITION_MAP.keys()), index=2)
                garage = st.selectbox('Garage', ['Yes', 'No'])
                submitted = st.form_submit_button('Get Appraisal')

            if submitted:
                row = build_feature_row(state, area, bedrooms, bathrooms, floors, year_built, location, condition, garage)
                row_scaled = state['scaler'].transform(row)
                price = state['model'].predict(row_scaled)[0]

                st.session_state.last_prediction = {
                    'price': price,
                    'area': area, 'bedrooms': bedrooms, 'bathrooms': bathrooms,
                    'floors': floors, 'year_built': year_built,
                    'location': location, 'condition': condition, 'garage': garage,
                }
                st.session_state.page = 'result'
                st.rerun()

    with col_spec:
        with st.container(border=True):
            st.markdown('<p class="eyebrow">TITLE BLOCK</p>', unsafe_allow_html=True)
            st.markdown('### Model Spec')
            m = state['metrics']
            st.metric('Test R²', f"{m['r2']:.4f}")
            st.metric('Test RMSE', f"${m['rmse']:,.0f}")
            st.metric('Test MAE', f"${m['mae']:,.0f}")
            st.caption(f"Trained on {m['n_train']} rows &middot; tested on {m['n_test']} rows")

            if m['r2'] < 0.2:
                st.markdown(
                    '<div class="spec-note">LOW CONFIDENCE &mdash; test R&sup2; is near zero, '
                    'meaning these features barely explain price in this dataset. Treat estimates '
                    'as illustrative of the pipeline, not reliable appraisals.</div>',
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    f'<div class="spec-note">Model explains roughly '
                    f'{m["r2"] * 100:.0f}% of price variance on held-out data.</div>',
                    unsafe_allow_html=True,
                )

            st.markdown('#### Price Distribution &amp; Fit', unsafe_allow_html=True)
            fig, axes = plt.subplots(1, 2, figsize=(9, 3.6))
            axes[0].hist(state['df']['Price'], bins=40, color='#16436b', edgecolor='#dcecf9')
            axes[0].set_title('Price Distribution', fontsize=10)
            axes[0].set_xlabel('Price', fontsize=8)
            axes[0].set_ylabel('Count', fontsize=8)

            axes[1].scatter(state['y_test'], state['pred'], alpha=0.4, s=10, color='#c9973a')
            lims = [min(state['y_test'].min(), state['pred'].min()), max(state['y_test'].max(), state['pred'].max())]
            axes[1].plot(lims, lims, 'r--', linewidth=1, label='Perfect prediction')
            axes[1].set_title('Actual vs. Predicted', fontsize=10)
            axes[1].set_xlabel('Actual price', fontsize=8)
            axes[1].set_ylabel('Predicted price', fontsize=8)
            axes[1].legend(fontsize=7)
            plt.tight_layout()
            st.pyplot(fig)

    st.markdown(
        '<p class="site-footer">Trained with scikit-learn, served with Streamlit.</p>',
        unsafe_allow_html=True,
    )


# ============================================================
# PAGE 2 — RESULT
# ============================================================
def render_result_page():
    st.markdown('<p class="eyebrow">Appraisal Result</p>', unsafe_allow_html=True)
    st.markdown('# Estimated Price')
    st.write('')

    data = st.session_state.last_prediction
    with st.container(border=True):
        st.markdown('<p class="eyebrow">TICKET</p>', unsafe_allow_html=True)
        st.markdown('### Property Appraisal')

        st.markdown('<div class="stamp">Preliminary Estimate</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="result-value">${data["price"]:,.0f}</div>', unsafe_allow_html=True)
        st.markdown(
            f'<p class="result-note">{data["area"]} sqft &middot; '
            f'{data["bedrooms"]}bd/{data["bathrooms"]}ba &middot; '
            f'{data["floors"]} floor(s) &middot; built {data["year_built"]} &middot; '
            f'{data["location"]} &middot; {data["condition"]} condition &middot; '
            f'garage: {data["garage"]}</p>',
            unsafe_allow_html=True,
        )

        if state['metrics']['r2'] < 0.2:
            st.markdown(
                f'<div class="spec-note">LOW CONFIDENCE &mdash; the underlying Linear Regression '
                f'model has a test R&sup2; of {state["metrics"]["r2"]:.4f}, close to zero. Treat this '
                f'figure as a demonstration of the pipeline rather than a trustworthy estimate.</div>',
                unsafe_allow_html=True,
            )

        st.write('')
        if st.button('← New appraisal'):
            st.session_state.page = 'form'
            st.rerun()

    st.markdown(
        '<p class="site-footer">Trained with scikit-learn, served with Streamlit.</p>',
        unsafe_allow_html=True,
    )


# ============================================================
# ROUTER
# ============================================================
if st.session_state.page == 'result' and st.session_state.last_prediction:
    render_result_page()
else:
    render_form_page()

# ============================================================
# HOW TO DEPLOY (Streamlit Community Cloud — free)
# ============================================================
# 1. Push this whole project folder to a GitHub repo, including:
#    streamlit_app.py, requirements.txt, House_Price_Prediction_Dataset.csv,
#    and image.jpg (optional — used in the hero image).
# 2. Go to https://share.streamlit.io and sign in with GitHub.
# 3. Click "New app" -> pick your repo/branch -> set "Main file path"
#    to streamlit_app.py -> Deploy.
# 4. You'll get a public URL like https://<your-app-name>.streamlit.app
#    that anyone can open from anywhere, no install needed.
