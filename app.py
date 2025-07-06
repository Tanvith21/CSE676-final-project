import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np
import pandas as pd

st.markdown(
    """
    <style>
    .stApp {
        background: linear-gradient(135deg, #0f2027, #203a43, #2c5364);
        color: white;
    }
    h1 {
        text-align: center;
        color: cyan;
        text-shadow: 0 0 10px cyan, 0 0 20px cyan;
    }
    .block-container {
        padding-top: 2rem;
    }
    </style>
    """,
    unsafe_allow_html=True
)

model = load_model('mobilenetv2_best_model/mobilenetv2_best_model.keras')

class_names = ['Drowsy', 'Not Drowsy']

st.set_page_config(
    page_title="Driver Drowsiness Detection",
    page_icon="🚦",
    layout="centered"
)

st.sidebar.title("About")
st.sidebar.markdown("""
This app detects **driver drowsiness** using a trained MobileNetV2 model.

- Upload an image of the driver  
- The model will classify whether the driver is **Drowsy** or **Not Drowsy**
""")
st.sidebar.markdown("---")
st.sidebar.write("Made by **Sai Tanvith Gulla**, **Harshith Reddy Pediredy**, **Akash Kumar Reddy Pallerla** for CSE676 Final Project.")

st.markdown("<h1>🚦 Driver Drowsiness Detection App</h1>", unsafe_allow_html=True)

st.subheader("👋 Welcome!")
st.markdown("Upload a **clear driver image** to check their drowsiness status.")

st.markdown("""
### Instructions:
1. Click **'Browse files'** to upload an image (`.jpg`, `.png`, `.jpeg`).
2. Click **'Predict'**.
3. The app will show whether the driver is **Drowsy** or **Not Drowsy** with a confidence score.
---
""")

uploaded_file = st.file_uploader("📤 Upload an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption="🖼️ Uploaded Image", use_column_width=True)

    img = image.resize((224, 224))
    img_array = np.array(img) / 255.0

    if img_array.shape[-1] == 4:
        img_array = img_array[..., :3]

    img_array = np.expand_dims(img_array, axis=0)

    if st.button("🚀 Predict"):
        with st.spinner('Predicting... ⏳'):
            predictions = model.predict(img_array)
            score = tf.nn.softmax(predictions[0])
            predicted_class = class_names[np.argmax(score)]
            confidence = 100 * np.max(score)

        if predicted_class == 'Drowsy':
            st.error(f"🚨 **Prediction:** {predicted_class}")
        else:
            st.success(f"✅ **Prediction:** {predicted_class}")

        st.info(f"🔍 **Confidence:** {confidence:.2f}%")

        st.progress(int(confidence))

        st.subheader("📊 Prediction Confidence")
        probabilities = {class_names[i]: float(score[i]*100) for i in range(len(class_names))}
        df = pd.DataFrame(list(probabilities.items()), columns=['Class', 'Confidence (%)'])
        st.bar_chart(df.set_index('Class'))

st.markdown("---")
st.caption("🚀 Made by **Sai Tanvith Gulla**, **Harshith Reddy Pediredy**, **Akash Kumar Reddy Pallerla** | CSE676 Final Project")
