import streamlit as st
import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLOWorld

st.set_page_config(page_title="Madagascar Lemur Detector")
st.title("🇲🇬 Lemur Species Detector (Zero-Shot)")

# 1. Load ny Model (V8 Small dia ampy tsara amin'ny sary)
@st.cache_resource
def load_model():
    model = YOLOWorld('yolov8s-world.pt')
    # Faritana eto ny species rehetra tianao ho hita
    model.set_classes(["Indri lemur", "Ring-tailed lemur", "Sifaka", "Bamboo lemur"])
    return model

model = load_model()

# 2. Upload Image
uploaded_file = st.file_uploader("Mampidira sarina gidro...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Avadika ho format azon'ny OpenCV ampiasaina ny sary
    image = Image.open(uploaded_file)
    img_array = np.array(image)
    
    # 3. Detection
    results = model.predict(img_array, conf=0.25) # Azonao ovaina ny conf raha tsy hitany
    
    # 4. Asehoy ny valiny
    res_plotted = results[0].plot() # Ity no manisy bounding boxes
    st.image(res_plotted, caption="Vokatry ny detection", use_container_width=True)
    
    # Manisa firy ny hita
    count = len(results[0].boxes)
    st.write(f"**Biby {count} no hita teo amin'ny sary.**")
