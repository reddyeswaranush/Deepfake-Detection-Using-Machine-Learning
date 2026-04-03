import streamlit as st
import pickle
import numpy as np

# Load model
model = pickle.load(open("model.pkl", "rb"))

st.title("Deepfake Detection App")

st.write("Paste power spectrum values (comma separated)")

input_data = st.text_area("Input")

if st.button("Predict"):
    try:
        data = np.array(list(map(float, input_data.split(",")))).reshape(1, -1)
        prediction = model.predict(data)

        if prediction[0] == 1:
            st.error("Fake Image")
        else:
            st.success("Real Image")
    except:
        st.warning("Invalid input")
