import streamlit as st
import joblib
import numpy as np

st.title("Cellphone Price Prediction Application")
Product_id = st.text_input("Enter Product ID:")
weight = st.slider("Select Weight (in grams):", min_value=500, max_value=3000)
resoloution = st.text_input("Enter Resolution (e.g., 1080x2400):")
ppi = st.slider("Select PPI (Pixels Per Inch):", min_value=1000, max_value=3000)
cpu_core = st.radio("Select Number of CPU Cores:", options=[4,5,6,7,8,9,10])
cpu_freq = st.slider("Select CPU Frequency (in GHz):", min_value=1.5, max_value=4.6)
internal_mem = st.slider("Select Internal Memory (in GB):", min_value=4, max_value=24)
ram = st.slider("Select RAM (in GB):", min_value=256, max_value=4000)
RearCam = st.number_input("Enter Rear Camera (in MP):")
Front_Cam = st.number_input("Enter Front Camera (in MP):")
battery = st.slider("Select Battery Capacity (in mAh):", min_value=500, max_value=2000)
thickness = st.number_input("Enter Thickness (in mm):", min_value=4.2, max_value=10.0)
if st.button("Predict Price"):
    # Load model & scaler
    model = joblib.load("mobile_price_prediction_model.pkl")
    scaler = joblib.load("scaler.pkl")

    # Prepare input (REMOVE Product_id)
    input_data = np.array([[ 
        Product_id,
        weight,
        resoloution,
        ppi,
        cpu_core,
        cpu_freq,
        internal_mem,
        ram,
        RearCam,
        Front_Cam,
        battery,
        thickness
    ]])

    # Scale
    input_data_scaled = scaler.transform(input_data)

    # Predict
    predicted_price = model.predict(input_data_scaled)

    st.success(f"The predicted price of the cellphone is: ${predicted_price[0]:.2f}")
    

    


