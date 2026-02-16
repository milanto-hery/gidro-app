import streamlit as st
import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLOWorld

st.set_page_config(page_title="Madagascar Lemur Expert AI", layout="wide")
st.title("🇲🇬 Lemur Species Identification (Zero-Shot)")
st.write("Mampiasà sary mba hamantarana ireo karazana gidro misy eto Madagasikara.")

# 1. Lisitra feno an'ireo species (Anarana mahazatra + Siantifika)
LEMUR_SPECIES = [
    "Ring-tailed Lemur (Lemur catta)", "Indri lemur (Babakoto)", 
    "Aye-aye (Daubentonia madagascariensis)", "Black-and-white Ruffed Lemur (Varecia variegata)",
    "Red Ruffed Lemur (Varecia rubra)", "Coquerel's Sifaka (Propithecus coquereli)",
    "Diademed Sifaka (Propithecus diadema)", "Verreaux's Sifaka (Propithecus verreauxi)",
    "Gray Mouse Lemur (Microcebus murinus)", "Goodman's Mouse Lemur (Microcebus lehilahytsara)",
    "Common Brown Lemur (Eulemur fulvus)", "Blue-eyed Black Lemur (Eulemur flavifrons)",
    "Red-bellied Lemur (Eulemur rubriventer)", "Crowned Lemur (Eulemur coronatus)",
    "Collared Brown Lemur (Eulemur collaris)", "Greater Bamboo Lemur (Prolemur simus)",
    "Golden Bamboo Lemur (Hapalemur aureus)", "Eastern Lesser Bamboo Lemur (Hapalemur griseus)",
    "Fat-tailed Dwarf Lemur (Cheirogaleus medius)", "Weasel Sportive Lemur (Lepilemur mustelinus)",
    "Pale Fork-marked Lemur (Phaner pallescens)"
]

# 2. Load ny Model
@st.cache_resource
def load_model():
    # Mampiasa 's' (small) ho an'ny Streamlit Cloud mba ho maivana
    model = YOLOWorld('yolov8s-world.pt')
    model.set_classes(LEMUR_SPECIES)
    return model

model = load_model()

# 3. Sidebar ho an'ny fanitsiana (Settings)
st.sidebar.header("Fikirakirana")
conf_threshold = st.sidebar.slider("Confidence Threshold (Fahatokisana)", 0.0, 1.0, 0.25)

# 4. Upload Image
uploaded_file = st.file_uploader("Misafidiana sary...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    img_array = np.array(image)
    
    with st.spinner('Eo am-pamakiana ny sary...'):
        # Inference
        results = model.predict(img_array, conf=conf_threshold)
        
        # Plotting
        res_plotted = results[0].plot()
        
    # Asehoy ny valiny
    col1, col2 = st.columns(2)
    
    with col1:
        st.image(image, caption="Sary nampidirinao", use_container_width=True)
    
    with col2:
        st.image(res_plotted, caption="Vokatry ny AI", use_container_width=True)

    # Lisitry ny species hita teo amin'ny sary
    detected_boxes = results[0].boxes
    if len(detected_boxes) > 0:
        st.success(f"Biby {len(detected_boxes)} no hita!")
        found_species = []
        for box in detected_boxes:
            cls_id = int(box.cls[0])
            found_species.append(LEMUR_SPECIES[cls_id])
        
        st.write("**Species hita:** " + ", ".join(list(set(found_species))))
    else:
        st.warning("Tsy nisy gidro hita tamin'io sary io. Andramo ampidinina ny Confidence.")
