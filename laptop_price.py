import streamlit as st
import pickle
import numpy as np
#import model
pipe = pickle.load(open("pipe.pkl",'rb'))
df = pickle.load(open('df.pkl','rb'))
st.title("Laptop predictor")
#BRAND
Company = st.selectbox('Brand',df['Brand'].unique())
RAM = st.selectbox('RAM',df["RAM (GB)"].unique())
Memory = st.selectbox('Memory',df["Storage (GB)"].unique())
GPU = st.selectbox('GPU',df["GPU"].unique())
SCREEN_S = st.selectbox("Screen Size (inches)",df["Screen Size (inches)"].unique())
Battery = st.selectbox("Battery Life (hours)",df["Battery Life (hours)"].unique())
CPU = st.selectbox("CPU CHOICE",df["cpu brand"].unique())
if st.button("Predict Price"):
    query = np.array([Company, RAM, Memory, GPU, SCREEN_S, Battery, CPU]).reshape(1, -1)
    predicted_price = round(np.exp(pipe.predict(query)[0]), 2)  # Rounds to 2 decimal places
    st.title(f"Predicted Price: ₹{predicted_price}")
