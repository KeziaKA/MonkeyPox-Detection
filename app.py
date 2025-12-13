import streamlit as st
import cv2
import numpy as np
import joblib
import os
from PIL import Image
from skimage.feature import local_binary_pattern

# Load model
IMG_SIZE = (224, 224)

@st.cache_resource
def load_models():
    model_path = 'final_monkeypox_model.pkl' 
    
    if not os.path.exists(model_path):
        st.error(f"Model path not found...")
        return None

    # Load dictionary
    bundle = joblib.load(model_path)
    return bundle

# Load sistem
bundle = load_models()

if bundle:
    kmeans = bundle['kmeans']
    scaler = bundle['scaler']
    model = bundle['model']
    classes = bundle['classes']
    NUM_CLUSTERS = bundle['num_clusters']
else:
    st.stop()

# Preprocessing
def preprocess_image(image_arr):
    img = cv2.resize(image_arr, IMG_SIZE)
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    lab = cv2.merge((l, a, b))
    img = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)

    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    return gray, img

# Feature extraction
def extract_color_features(image):
    # RGB
    (R, G, B) = cv2.split(image)
    
    # HSV
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    (H, S, V) = cv2.split(hsv)

    features = [
        np.mean(R)/255.0, np.mean(G)/255.0, np.mean(B)/255.0,
        np.std(R)/255.0, np.std(G)/255.0, np.std(B)/255.0,
        np.mean(H)/180.0, np.mean(S)/255.0, np.mean(V)/255.0,
        np.std(H)/180.0, np.std(S)/255.0, np.std(V)/255.0,
        np.median(R)/255.0, np.median(G)/255.0, np.median(B)/255.0,
    ]
    return np.array(features)

def extract_texture_features(gray_image):
    radius = 3
    n_points = 8 * radius
    lbp = local_binary_pattern(gray_image, n_points, radius, method='uniform')

    n_bins = int(lbp.max() + 1)
    hist, _ = np.histogram(lbp.ravel(), bins=n_bins, range=(0, n_bins), density=True)
    return hist

def extract_sift_features(gray_image, kmeans):
    sift = cv2.SIFT_create(nfeatures=500)
    kp, des = sift.detectAndCompute(gray_image, None)

    if des is not None and len(des) > 0:
        words = kmeans.predict(des)
        hist, _ = np.histogram(words, bins=np.arange(NUM_CLUSTERS + 1))
        hist = hist.astype(np.float32)
        hist = hist / (np.linalg.norm(hist) + 1e-7)
    else:
        hist = np.zeros(NUM_CLUSTERS, dtype=np.float32)

    return hist

# Concat all features
def get_final_vector(img_array):
    gray, img_color = preprocess_image(img_array)
    sift_hist = extract_sift_features(gray, kmeans)
    color_feats = extract_color_features(img_color)
    texture_feats = extract_texture_features(gray)

    final_vector = np.concatenate([sift_hist, color_feats, texture_feats])
    return final_vector

# User interface
st.set_page_config(page_title="Mpox Detector", page_icon="🐒")

st.markdown("""
    <h1 style='text-align: center;'>
        🐒 Monkeypox Detection
    </h1>
""", unsafe_allow_html=True)
st.markdown("""
Monkeypox Skin Detection
* **Shape:** SIFT + Bag of Visual Words
* **Texture:** Local Binary Pattern (LBP)
* **Color:** HSV & RGB
""")

uploaded_file = st.file_uploader("Upload skin image (JPG/PNG)", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Tampilkan Gambar
    image = Image.open(uploaded_file)
    
    st.image(image, caption='Original Image')
        
    if st.button("Predict"):
        with st.spinner('Preprocessing image...'):
            # Convert ke Array
            img_array = np.array(image.convert('RGB'))
            
            # Ekstrak Fitur
            features = get_final_vector(img_array)
            
            # Normalisasi (Scaler)
            features_scaled = scaler.transform(features.reshape(1, -1))
            
            # Prediksi SVM
            pred_idx = model.predict(features_scaled)[0]
            probs = model.predict_proba(features_scaled)[0]
            
            result_class = classes[pred_idx]
            confidence = probs[pred_idx] * 100
            
        # Show result
        if result_class == "Monkeypox":
            st.error(f"### Detection result: {result_class}")
        elif result_class == "Measles" or result_class == "Chickenpox":
            st.warning(f"### Detection result: {result_class}")
        else:
            st.success(f"### Detection result: {result_class}")
                
        st.metric("Model confidence score", f"{confidence:.2f}%")
            
        # Grafik Probabilitas
        st.write("---")
        st.write("Class probability:")
        for i, cls in enumerate(classes):
            st.progress(int(probs[i]*100))
            st.caption(f"{cls}: {probs[i]*100:.2f}%")