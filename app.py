import os
# Reduce TensorFlow/absl noise before TF imports
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"        # hide INFO/WARNING logs
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")  # optional: disable oneDNN info line

try:
    import absl.logging
    absl.logging.set_verbosity(absl.logging.ERROR)
except Exception:
    pass

import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import pandas as pd
from pathlib import Path

# Use Streamlit's cache decorator to load the model only once
# This is crucial for performance and scalability!
@st.cache_resource
def load_model():
    # Use a path relative to this app.py so Streamlit can be run from any CWD
    model_path = (Path(__file__).parent / "practical" / "mnist_cnn_model.h5").resolve()
    try:
        model = tf.keras.models.load_model(str(model_path))
        # Ensure compiled metrics exist (avoids "compiled metrics have yet to be built" warning)
        try:
            model.compile(optimizer='adam',
                          loss='sparse_categorical_crossentropy',
                          metrics=['accuracy'])
        except Exception:
            # If recompile fails (custom objects etc.), ignore — model still usable for predict()
            pass
    except Exception as e:
        st.error(f"Could not load model at {model_path}: {e}")
        st.stop()
    return model

# Function to preprocess the uploaded image
def preprocess_image(image):
    # The MNIST model expects a 28x28 grayscale image.
    img = image.resize((28, 28))
    img = img.convert('L') # Convert to grayscale (single channel)
    
    # Convert image to numpy array and normalize
    img_array = np.array(img).astype('float32') / 255.0
    
    # Invert colors: MNIST data is white digit on black background (values 0-1)
    # Uploaded images are usually black digit on white background
    img_array = 1 - img_array
    
    # Reshape for the CNN: (1, 28, 28, 1) - (batch size, height, width, channels)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = np.expand_dims(img_array, axis=-1)
    return img_array

# --- Main Streamlit App ---

# Load model globally
model = load_model()

st.title("🔢 MNIST CNN Digit Classifier")
st.markdown("Upload a handwritten digit image (0-9) to get a real-time prediction from the Convolutional Neural Network (Test Accuracy: >95%).")

# File uploader widget
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # 1. Display the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True, width=150)
    st.write("")
    
    # 2. Preprocess and Predict
    st.subheader("Analysis Results:")
    
    # Use st.spinner for a professional look while processing
    with st.spinner('Classifying image...'):
        processed_image = preprocess_image(image)
        
        # Get raw prediction probabilities
        predictions = model.predict(processed_image)
        
        # Get the class with the highest probability
        predicted_class = np.argmax(predictions[0])
        confidence = predictions[0][predicted_class] * 100

    # 3. Display Prediction
    st.success(f"**Predicted Digit: {predicted_class}**")
    st.info(f"Confidence: {confidence:.2f}%")
    
    # Optionally display the probability distribution
    st.subheader("Probability Distribution")
    # Create a simple bar chart of the probabilities
    prob_df = pd.DataFrame(predictions[0], columns=['Probability'])
    prob_df['Digit'] = range(10)
    prob_df = prob_df.sort_values('Probability', ascending=False)
    
    st.bar_chart(prob_df.set_index('Digit'))

# Add a note on the model source
st.sidebar.markdown("---")
st.sidebar.markdown("Model: **TensorFlow CNN** trained on the MNIST dataset.")