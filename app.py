import streamlit as st
import pandas as pd
import pickle

st.title("⚡ Energy Consumption Forecasting System")

# Load model
model = pickle.load(open('energy_model.pkl', 'rb'))

st.write("Upload your dataset (CSV format)")

uploaded_file = st.file_uploader("Choose a file")

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    st.write("Input Data:")
    st.write(df.head())

    try:
        prediction = model.predict(df)

        st.write("Predictions:")
        st.write(prediction)

    except Exception as e:
        st.error("Error in prediction. Check input format.")
        st.write(e)