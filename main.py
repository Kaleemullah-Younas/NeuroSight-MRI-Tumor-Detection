import streamlit as st
from PIL import Image
import numpy as np
import os
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from tensorflow.keras.models import load_model

from util import set_background


st.set_page_config(page_title='NeuroSight: MRI Tumor Detection', page_icon='🧠')
set_background('./bg.jpg')

# Center align title and content
st.markdown("""
<style>
.block-container {
    max-width: 800px;
    padding-top: 2rem;
    padding-bottom: 2rem;
    margin: 0 auto;
}
h1, h2, h3 {
    text-align: center;
}
div.stButton > button {
    display: block;
    margin: 0 auto;
}
</style>
""", unsafe_allow_html=True)

st.title('NeuroSight: MRI Tumor Detection')
st.header('Upload an axial brain MRI image')

# Simple centered file uploader
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    file = st.file_uploader('', type=['png', 'jpg', 'jpeg'])

# Load the Brain Tumor CNN Model
MODEL_PATH = './model/braintumor_binary.h5'
braintumor_model = None

if os.path.exists(MODEL_PATH):
    braintumor_model = load_model(MODEL_PATH)
else:
    st.error('Model not found at ./model/braintumor_binary.h5')

if file and braintumor_model is not None:
    # Display the uploaded image
    image = Image.open(file)
    st.image(image, caption='Uploaded MRI', use_column_width=True)
    
    # Convert to temp file for load_img compatibility
    import tempfile
    temp_file = tempfile.NamedTemporaryFile(delete=False)
    temp_file_path = temp_file.name
    image.save(temp_file_path)
    
    try:
        # Load and preprocess the image
        img = load_img(temp_file_path, target_size=(128, 128))
        img_array = img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0) / 255.0
        
        # Make prediction
        pred = braintumor_model.predict(img_array)
        prediction = pred[0][0]
        confidence = prediction if prediction > 0.5 else 1 - prediction
        predicted_class = 'Tumor Detected' if prediction > 0.5 else 'No Tumor Detected'
        
        # Display result in a nice format
        st.markdown(f"<h2 style='text-align: center;'>{predicted_class}</h2>", unsafe_allow_html=True)
        st.markdown(f"<h3 style='text-align: center;'>Confidence: {confidence:.2f}</h3>", unsafe_allow_html=True)
    finally:
        # Clean up temp file
        os.unlink(temp_file_path)

