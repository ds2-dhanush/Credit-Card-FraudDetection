import streamlit as st
import pandas as pd
import joblib

# Load trained model and features
model = joblib.load("fraud_detector_model.pkl")
features = joblib.load("model_features.pkl")

# --- Encoding dictionaries with Indian context ---
merchant_dict = {
    "Amazon": 101,
    "Flipkart": 102,
    "Zomato": 103,
    "Swiggy": 104,
    "Myntra": 105,
    "Paytm": 106,
    "PhonePe": 107,
    "Other": -1
}

category_dict = {
    "shopping": 1,
    "food_delivery": 2,
    "entertainment": 3,
    "travel": 4,
    "recharge": 5,
    "utility": 6,
    "Other": -1
}

street_dict = {
    "MG Road": 11,
    "Brigade Road": 12,
    "Linking Road": 13,
    "BTM Layout": 14,
    "Sector 18": 15,
    "Other": -1
}

city_dict = {
    "Mumbai": 21,
    "Delhi": 22,
    "Bangalore": 23,
    "Hyderabad": 24,
    "Chennai": 25,
    "Kolkata": 26,
    "Pune": 27,
    "Other": -1
}

state_dict = {
    "Maharashtra": 31,
    "Karnataka": 32,
    "Delhi NCR": 33,
    "Telangana": 34,
    "Tamil Nadu": 35,
    "West Bengal": 36,
    "Other": -1
}

# --- Streamlit UI ---
st.set_page_config(page_title="Credit Card Fraud Detector", layout="wide")
st.title("💳 Credit Card Fraud Detection App")

# About section
with st.expander("📘 About This App"):
    st.markdown("""
    This Streamlit app uses a machine learning model trained to detect fraudulent credit card transactions.
    - Upload a CSV file or manually enter transaction details.
    - The model returns predictions along with confidence scores.
    - Built using Python, XGBoost, SMOTE, and Streamlit.
    """)

# --- CSV Upload ---
st.header("📁 Upload CSV File")
uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

if uploaded_file:
    try:
        df = pd.read_csv(uploaded_file)
        input_data = df[features]
        df['Prediction'] = model.predict(input_data)
        df['Confidence (%)'] = model.predict_proba(input_data)[:, 1] * 100
        st.success("✅ Prediction complete")
        st.dataframe(df)

        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button("Download Results", data=csv, file_name="predictions.csv", mime='text/csv')

    except Exception as e:
        st.error(f"CSV error: {e}")

# --- Manual Form ---
st.markdown("---")
st.subheader("✍️ Manually Enter Transaction")

with st.form("manual_form"):
    col1, col2 = st.columns(2)

    with col1:
        merchant_choice = st.selectbox("Merchant", list(merchant_dict.keys()))
        if merchant_choice == "Other":
            custom_merchant = st.text_input("Enter custom merchant name")
            merchant = abs(hash(custom_merchant)) % 1000 if custom_merchant else 0
        else:
            merchant = merchant_dict[merchant_choice]

        category_choice = st.selectbox("Category", list(category_dict.keys()))
        if category_choice == "Other":
            custom_category = st.text_input("Enter custom category name")
            category = abs(hash(custom_category)) % 100 if custom_category else 0
        else:
            category = category_dict[category_choice]

        amt = st.number_input("Amount ($)", value=0.45)

        street_choice = st.selectbox("Street", list(street_dict.keys()))
        if street_choice == "Other":
            custom_street = st.text_input("Enter custom street name")
            street = abs(hash(custom_street)) % 100 if custom_street else 0
        else:
            street = street_dict[street_choice]

        city_choice = st.selectbox("City", list(city_dict.keys()))
        if city_choice == "Other":
            custom_city = st.text_input("Enter custom city name")
            city = abs(hash(custom_city)) % 100 if custom_city else 0
        else:
            city = city_dict[city_choice]

        state_choice = st.selectbox("State", list(state_dict.keys()))
        if state_choice == "Other":
            custom_state = st.text_input("Enter custom state name")
            state = abs(hash(custom_state)) % 100 if custom_state else 0
        else:
            state = state_dict[state_choice]

        lat = st.number_input("Latitude", value=19.076)
        long = st.number_input("Longitude", value=72.8777)

    with col2:
        city_pip = st.number_input("City Pip", value=1001)
        merch_lat = st.number_input("Merchant Latitude", value=19.075)
        merch_log = st.number_input("Merchant Longitude", value=72.8776)
        distance = st.number_input("Distance", value=0.12)
        hour = st.number_input("Hour", value=13)
        day = st.number_input("Day", value=14)
        weekday = st.number_input("Weekday", value=2)
        month = st.number_input("Month", value=6)

    submitted = st.form_submit_button("🔍 Predict Fraud")

    if submitted:
        try:
            input_row = pd.DataFrame([[
                merchant, category, amt, street, city, state, lat, long,
                city_pip, merch_lat, merch_log, distance, hour, day, weekday, month
            ]], columns=features)

            prediction = model.predict(input_row)[0]
            proba = model.predict_proba(input_row)[0][1] * 100  # Probability of fraud
            confidence = round(proba, 2)

            if prediction == 1:
                st.error(f"❌ Fraud Detected – Confidence: {confidence}%")
            else:
                st.success(f"✅ Transaction is Safe – Confidence: {100 - confidence}%")

        except Exception as e:
            st.error(f"Something went wrong: {e}")