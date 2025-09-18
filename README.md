# NeuroSight: MRI Tumor Detection

<p align="center">
  <img src="https://img.shields.io/badge/Streamlit-App-red?logo=streamlit" />
  <img src="https://img.shields.io/badge/TensorFlow-Keras-orange?logo=tensorflow" />
  <img src="https://img.shields.io/badge/Python-3.9%2B-green?logo=python" />
  <img src="https://img.shields.io/badge/License-MIT-black" />
</p>

NeuroSight is a simple Streamlit app that loads a TensorFlow/Keras `.h5` binary classifier to predict whether an axial brain MRI image contains a tumor.

> Disclaimer: For research/education only. Not a medical device.

## Features

- Upload `.png`/`.jpg` images and get a binary prediction
- Adjustable classification threshold in the sidebar
- Clean UI with background image

## Quickstart

### 1) Setup

```bash
git clone "<this-repo>"
cd "Object Detection (MRI Detection)"
python -m venv .venv
.venv\Scripts\activate  # Windows
pip install -r requirements.txt
```

### 2) Add model file

Place your trained Keras model at:

```
./models/braintumor_binary.h5
```

### 3) Run the app

```bash
streamlit run main.py
```

Open the URL shown in the terminal (usually `http://localhost:8501`).

## Usage

1. Set the classification threshold in the sidebar.
2. Upload an axial brain MRI image (`.png`/`.jpg`).
3. Read the predicted label and confidence displayed above the image.

## Project Structure

```
main.py                     # Streamlit app (Keras classifier)
util.py                     # Background styling helper
requirements.txt            # Python dependencies
bg.png                      # Background image
models/braintumor_binary.h5  # Your model (not included)
```

## Acknowledgements

- Built with [Streamlit](https://streamlit.io/) and [TensorFlow/Keras](https://www.tensorflow.org/)
