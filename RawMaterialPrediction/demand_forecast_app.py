# demand_forecast_app.py

import pandas as pd
import numpy as np
import joblib
import streamlit as st
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt

st.set_page_config(page_title="Raw Material Demand Forecast", page_icon="📦", layout="centered")

st.title("📦 Raw Material Demand Forecasting App (with Trend Visualization)")
st.write("Forecasts that inform procurement schedules, inventory levels, and production planning — reducing waste and improving service levels.")

# ------------------------------------------------
# Step 1️⃣: Load Dataset
# ------------------------------------------------
@st.cache_data
def load_data():
    data = pd.read_csv("https://www.kaggle.com/datasets/arashnic/food-demand/train.csv")
    return data

data = load_data()
st.subheader("📊 Dataset Overview")
st.write(data.head())

# ------------------------------------------------
# Step 2️⃣: Load or Train Model (Optimized)
# ------------------------------------------------
MODEL_PATH = "demand_model.pkl"

@st.cache_resource
def load_or_train_model():
    if os.path.exists(MODEL_PATH):
        try:
            model = joblib.load(MODEL_PATH)
            st.success("✅ Model loaded from saved file!")
            return model
        except MemoryError:
            st.warning("⚠️ Existing model too large to load. Training lightweight version...")
            os.remove(MODEL_PATH)

    # Sample smaller portion of dataset to reduce training time
    sample_data = data.sample(frac=0.15, random_state=42)

    X = sample_data[['week', 'center_id', 'meal_id', 'checkout_price', 'base_price',
                     'emailer_for_promotion', 'homepage_featured']]
    y = sample_data['num_orders']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    try:
        # Lightweight Random Forest
        model = RandomForestRegressor(
            n_estimators=25,
            max_depth=10,
            n_jobs=-1,
            random_state=42
        )
        model.fit(X_train, y_train)
        st.info("🌳 Trained lightweight Random Forest model successfully!")
    except MemoryError:
        # Fall back to linear regression if memory is too low
        st.warning("⚠️ Memory low! Switching to Linear Regression model.")
        model = LinearRegression()
        model.fit(X_train, y_train)

    # Evaluate performance
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    st.write(f"📊 Model Evaluation — Mean Absolute Error (MAE): `{mae:.2f}`")

    # Save model
    joblib.dump(model, MODEL_PATH)
    st.success("💾 Model saved as 'demand_model.pkl' for future fast loading!")
    return model

model = load_or_train_model()

# ------------------------------------------------
# Step 3️⃣: Single Demand Prediction
# ------------------------------------------------
st.header("🔮 Predict Material Demand for a Specific Input")

week = st.number_input("📅 Week Number", min_value=1, max_value=145, value=50)
center_id = st.number_input("🏭 Center ID", min_value=1, max_value=100, value=13)
meal_id = st.number_input("🍴 Material / Meal ID", min_value=1, max_value=2000, value=1248)
checkout_price = st.number_input("💰 Checkout Price", min_value=50.0, max_value=800.0, value=250.0)
base_price = st.number_input("💸 Base Price", min_value=50.0, max_value=800.0, value=280.0)
emailer_for_promotion = st.selectbox("📧 Emailer Promotion?", [0, 1])
homepage_featured = st.selectbox("🏠 Homepage Featured?", [0, 1])

if st.button("🚀 Predict Demand"):
    features = np.array([[week, center_id, meal_id, checkout_price, base_price,
                          emailer_for_promotion, homepage_featured]])
    prediction = model.predict(features)
    st.success(f"📈 Predicted Material Demand: **{prediction[0]:.0f} units**")

# ------------------------------------------------
# Step 4️⃣: Demand Trend Visualization
# ------------------------------------------------
st.markdown("---")
st.header("📊 Forecast Demand Trend (Week Range)")

start_week = st.number_input("Start Week", min_value=1, max_value=145, value=1)
end_week = st.number_input("End Week", min_value=1, max_value=145, value=20)

if st.button("📅 Show Demand Trend"):
    if start_week >= end_week:
        st.error("⚠️ End Week must be greater than Start Week.")
    else:
        weeks = np.arange(start_week, end_week + 1)
        inputs = pd.DataFrame({
            'week': weeks,
            'center_id': [center_id] * len(weeks),
            'meal_id': [meal_id] * len(weeks),
            'checkout_price': [checkout_price] * len(weeks),
            'base_price': [base_price] * len(weeks),
            'emailer_for_promotion': [emailer_for_promotion] * len(weeks),
            'homepage_featured': [homepage_featured] * len(weeks)
        })

        preds = model.predict(inputs)

        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(weeks, preds, marker='o', linestyle='-', linewidth=2)
        ax.set_xlabel("Week Number")
        ax.set_ylabel("Predicted Demand (Units)")
        ax.set_title("📈 Predicted Material Demand Trend Over Time")
        ax.grid(True)
        st.pyplot(fig)

        st.info("💡 This graph shows how predicted demand is expected to change week by week based on current input parameters.")

# ------------------------------------------------
# Step 5️⃣: Footer
# ------------------------------------------------
st.markdown("---")




