import streamlit as st
from PIL import Image
from ultralytics import YOLO
import numpy as np
import cv2

# Page configuration
st.set_page_config(
    page_title="Space Station Safety Detector",
    page_icon="🛸",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Minimal, clean CSS
st.markdown("""
    <style>
    /* Base styling */
    .main {
        background: #0f0f1e;
    }
    .stApp {
        background: linear-gradient(180deg, #0f0f1e 0%, #1a1a2e 100%);
    }
    
    /* Hide default Streamlit elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* Typography */
    h1 {
        color: #ffffff;
        font-weight: 700;
        font-size: 2.5rem;
        margin-bottom: 0.25rem;
        letter-spacing: -0.02em;
    }
    h3 {
        color: #e0e0e0;
        font-weight: 600;
        font-size: 1.25rem;
        margin-top: 0;
    }
    
    /* Header section */
    .header-container {
        text-align: center;
        padding: 2rem 0 3rem 0;
        border-bottom: 1px solid rgba(255, 255, 255, 0.1);
        margin-bottom: 2rem;
    }
    .subtitle {
        color: #a0a0b0;
        font-size: 1rem;
        margin-top: 0.5rem;
        font-weight: 400;
    }
    .badge {
        display: inline-block;
        background: rgba(99, 102, 241, 0.15);
        border: 1px solid rgba(99, 102, 241, 0.3);
        padding: 0.4rem 1rem;
        border-radius: 2rem;
        color: #818cf8;
        font-size: 0.85rem;
        margin-top: 1rem;
        font-weight: 500;
    }
    
    /* Cards */
    .card {
        background: rgba(26, 26, 46, 0.6);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 1rem;
        padding: 1.5rem;
        backdrop-filter: blur(10px);
        transition: all 0.3s ease;
    }
    .card:hover {
        border-color: rgba(99, 102, 241, 0.3);
        transform: translateY(-2px);
        box-shadow: 0 8px 16px rgba(0, 0, 0, 0.3);
    }
    
    /* Detection items */
    .detection-item {
        background: rgba(30, 30, 50, 0.8);
        border: 1px solid rgba(255, 255, 255, 0.06);
        border-radius: 0.75rem;
        padding: 1rem;
        margin-bottom: 0.75rem;
        transition: all 0.2s ease;
    }
    .detection-item:hover {
        background: rgba(40, 40, 60, 0.8);
        border-color: rgba(255, 255, 255, 0.12);
    }
    
    /* Class label */
    .class-label {
        color: #ffffff;
        font-weight: 600;
        font-size: 1rem;
        margin: 0;
    }
    .detected-badge {
        background: rgba(16, 185, 129, 0.2);
        color: #6ee7b7;
        padding: 0.25rem 0.75rem;
        border-radius: 1rem;
        font-size: 0.75rem;
        font-weight: 600;
        letter-spacing: 0.05em;
    }
    
    /* Progress bar */
    .confidence-bar {
        width: 100%;
        height: 6px;
        background: rgba(255, 255, 255, 0.08);
        border-radius: 3px;
        overflow: hidden;
        margin: 0.75rem 0 0.5rem 0;
    }
    .confidence-fill {
        height: 100%;
        border-radius: 3px;
        transition: width 0.5s ease;
    }
    .confidence-text {
        color: #9ca3af;
        font-size: 0.8rem;
        text-align: right;
        margin: 0;
    }
    
    /* Legend */
    .legend-item {
        display: flex;
        align-items: center;
        gap: 0.5rem;
        padding: 0.4rem 0;
        color: #d1d5db;
        font-size: 0.9rem;
        transition: color 0.2s ease;
    }
    .legend-item:hover {
        color: #ffffff;
    }
    .legend-dot {
        width: 10px;
        height: 10px;
        border-radius: 50%;
        flex-shrink: 0;
    }
    
    /* Empty state */
    .empty-state {
        text-align: center;
        padding: 4rem 2rem;
        color: #6b7280;
    }
    .empty-icon {
        font-size: 4rem;
        opacity: 0.5;
        margin-bottom: 1rem;
    }
    
    /* Alert boxes */
    .alert {
        border-radius: 0.75rem;
        padding: 1rem 1.25rem;
        margin-top: 1.5rem;
        border-left: 4px solid;
    }
    .alert-info {
        background: rgba(59, 130, 246, 0.1);
        border-color: #3b82f6;
        color: #93c5fd;
    }
    .alert-warning {
        background: rgba(245, 158, 11, 0.1);
        border-color: #f59e0b;
        color: #fbbf24;
    }
    
    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background: rgba(15, 15, 30, 0.95);
        border-right: 1px solid rgba(255, 255, 255, 0.1);
    }
    
    /* Button enhancements */
    .stButton button {
        background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%);
        color: white;
        font-weight: 600;
        border: none;
        transition: all 0.3s ease;
    }
    .stButton button:hover {
        background: linear-gradient(135deg, #4f46e5 0%, #7c3aed 100%);
        transform: translateY(-1px);
        box-shadow: 0 6px 20px rgba(99, 102, 241, 0.4);
    }
    
    /* Divider */
    .divider {
        height: 1px;
        background: rgba(255, 255, 255, 0.1);
        margin: 1.5rem 0;
    }
    </style>
""", unsafe_allow_html=True)

# Define classes and colors
CLASSES = [
    'OxygenTank',
    'NitrogenTank',
    'FirstAidBox',
    'FireAlarm',
    'SafetySwitchPanel',
    'EmergencyPhone',
    'FireExtinguisher'
]

CLASS_COLORS = {
    'OxygenTank': '#3b82f6',
    'NitrogenTank': '#8b5cf6',
    'FirstAidBox': '#ef4444',
    'FireAlarm': '#f59e0b',
    'SafetySwitchPanel': '#10b981',
    'EmergencyPhone': '#06b6d4',
    'FireExtinguisher': '#ec4899'
}

CLASS_COLORS_RGB = {
    'OxygenTank': (59, 130, 246),
    'NitrogenTank': (139, 92, 246),
    'FirstAidBox': (239, 68, 68),
    'FireAlarm': (245, 158, 11),
    'SafetySwitchPanel': (16, 185, 129),
    'EmergencyPhone': (6, 182, 212),
    'FireExtinguisher': (236, 72, 153)
}

@st.cache_resource
def load_model(model_path):
    """Load the YOLO model"""
    try:
        model = YOLO(model_path)
        return model, None
    except Exception as e:
        return None, str(e)

def detect_objects(model, image, conf_threshold=0.25):
    """Run detection on the image"""
    try:
        img_array = np.array(image)
        results = model(img_array, conf=conf_threshold, verbose=False)
        
        detections = []
        annotated_image = img_array.copy()
        
        for r in results:
            boxes = r.boxes
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                conf = float(box.conf[0])
                cls = int(box.cls[0])
                class_name = r.names[cls]
                
                color = CLASS_COLORS_RGB.get(class_name, (255, 255, 255))
                cv2.rectangle(annotated_image, 
                            (int(x1), int(y1)), 
                            (int(x2), int(y2)), 
                            color, 3)
                
                label = f"{class_name}"
                label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
                cv2.rectangle(annotated_image,
                            (int(x1), int(y1) - label_size[1] - 10),
                            (int(x1) + label_size[0] + 8, int(y1)),
                            color, -1)
                cv2.putText(annotated_image, label,
                          (int(x1) + 4, int(y1) - 5),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                
                detections.append({
                    'class': class_name,
                    'confidence': conf,
                    'color': CLASS_COLORS[class_name],
                    'bbox': [int(x1), int(y1), int(x2), int(y2)]
                })
        
        return detections, Image.fromarray(annotated_image)
        
    except Exception as e:
        st.error(f"Detection error: {str(e)}")
        return [], image

# Initialize session state
if 'detections' not in st.session_state:
    st.session_state.detections = []
if 'annotated_image' not in st.session_state:
    st.session_state.annotated_image = None
if 'image_processed' not in st.session_state:
    st.session_state.image_processed = False

def main():
    # Header
    st.markdown("""
        <div class='header-container'>
            <h1>🛸 Space Station Safety Detector</h1>
            <p class='subtitle'>AI-powered detection of critical safety equipment in space environments</p>
            <span class='badge'>YOLOv8 • mAP@0.5: 80.4%</span>
        </div>
    """, unsafe_allow_html=True)
    
    # Sidebar configuration
    with st.sidebar:
        st.markdown("### ⚙️ Settings")
        
        model_path = st.text_input(
            "Model Path",
            value="runs/detect/train2/weights/best.pt",
            help="Path to your trained YOLOv8 model weights"
        )
        
        conf_threshold = st.slider(
            "Confidence Threshold",
            min_value=0.01,
            max_value=1.0,
            value=0.25,
            step=0.01,
            help="Minimum confidence for detections"
        )
        
        st.markdown("<div class='divider'></div>", unsafe_allow_html=True)
        
        # Model info
        st.markdown("### 📊 Model Performance")
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Precision", "89.7%")
            st.metric("mAP@0.5", "80.4%")
        with col2:
            st.metric("Recall", "66.3%")
            st.metric("Classes", "7")
        
        st.markdown("<div class='divider'></div>", unsafe_allow_html=True)
        
        # Legend in sidebar
        st.markdown("### 🎨 Detection Classes")
        for cls in CLASSES:
            st.markdown(f"""
                <div class='legend-item'>
                    <span class='legend-dot' style='background-color: {CLASS_COLORS[cls]};'></span>
                    <span>{cls}</span>
                </div>
            """, unsafe_allow_html=True)
    
    # Load model
    model, error = load_model(model_path)
    
    if error:
        st.markdown(f"""
            <div class='alert alert-warning'>
                <strong>⚠️ Model Loading Error</strong><br>
                Could not load model from: <code>{model_path}</code><br>
                Please check the path in the sidebar settings.
            </div>
        """, unsafe_allow_html=True)
        return
    
    # Main content area
    col1, col2 = st.columns([1, 1], gap="large")
    
    with col1:
        st.markdown("<h3>📤 Upload & Analyze</h3>", unsafe_allow_html=True)
        
        uploaded_file = st.file_uploader(
            "Choose an image",
            type=['jpg', 'jpeg', 'png'],
            help="Upload a space station image to detect safety equipment",
            label_visibility="collapsed"
        )
        
        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            
            # Show image with rounded corners
            st.markdown("<div class='card'>", unsafe_allow_html=True)
            if st.session_state.annotated_image is not None and st.session_state.image_processed:
                st.image(st.session_state.annotated_image, use_container_width=True)
            else:
                st.image(image, use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)
            
            # Detect button
            st.markdown("<br>", unsafe_allow_html=True)
            if st.button("🔍 Detect Safety Equipment", type="primary", use_container_width=True):
                with st.spinner("🔄 Analyzing image..."):
                    detections, annotated_img = detect_objects(model, image, conf_threshold)
                    st.session_state.detections = detections
                    st.session_state.annotated_image = annotated_img
                    st.session_state.image_processed = True
                    st.rerun()
        else:
            st.markdown("""
                <div class='card'>
                    <div class='empty-state'>
                        <div class='empty-icon'>📷</div>
                        <p>Upload an image to begin detection</p>
                        <p style='font-size: 0.85rem; margin-top: 0.5rem;'>Supported formats: JPG, PNG</p>
                    </div>
                </div>
            """, unsafe_allow_html=True)
            st.session_state.detections = []
            st.session_state.annotated_image = None
            st.session_state.image_processed = False
    
    with col2:
        st.markdown("<h3>✅ Detection Results</h3>", unsafe_allow_html=True)
        
        if st.session_state.detections and st.session_state.image_processed:
            # Summary
            st.markdown(f"""
                <div class='card' style='background: rgba(16, 185, 129, 0.1); border-color: rgba(16, 185, 129, 0.3); margin-bottom: 1rem;'>
                    <div style='text-align: center;'>
                        <span style='font-size: 2rem; font-weight: 700; color: #6ee7b7;'>{len(st.session_state.detections)}</span>
                        <p style='color: #d1fae5; margin: 0.25rem 0 0 0;'>Objects Detected</p>
                    </div>
                </div>
            """, unsafe_allow_html=True)
            
            # Detections list
            for detection in st.session_state.detections:
                st.markdown(f"""
                    <div class='detection-item'>
                        <div style='display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.5rem;'>
                            <div style='display: flex; align-items: center; gap: 0.75rem;'>
                                <span class='legend-dot' style='background-color: {detection["color"]};'></span>
                                <span class='class-label'>{detection["class"]}</span>
                            </div>
                            <span class='detected-badge'>DETECTED</span>
                        </div>
                        <div class='confidence-bar'>
                            <div class='confidence-fill' style='width: {detection["confidence"]*100}%; background: linear-gradient(90deg, {detection["color"]}, {detection["color"]}dd);'></div>
                        </div>
                        <p class='confidence-text'>Confidence: {detection["confidence"]*100:.1f}%</p>
                    </div>
                """, unsafe_allow_html=True)
        else:
            st.markdown("""
                <div class='card'>
                    <div class='empty-state'>
                        <div class='empty-icon'>🔍</div>
                        <p>No detections yet</p>
                        <p style='font-size: 0.85rem; margin-top: 0.5rem;'>Upload an image and click detect to see results</p>
                    </div>
                </div>
            """, unsafe_allow_html=True)
    
    # Footer info
    st.markdown("""
        <div class='alert alert-info'>
            <strong>ℹ️ How It Works</strong><br>
            This application uses a trained YOLOv8 model to identify critical safety equipment in space station environments. 
            Upload an image, adjust the confidence threshold if needed, and click detect to analyze.
        </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()