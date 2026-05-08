import streamlit as st
import gdown
import os
import tempfile
import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO

# ─────────────────────────────────────────────
# CONFIG 
# ─────────────────────────────────────────────
GDRIVE_FILE_ID = st.secrets["GDRIVE_FILE_ID"]
MODEL_PATH = "best.pt"

# Disease info dictionary for extra context
DISEASE_INFO = {
    "Apple Scab": "Fungal disease causing dark, scabby lesions on leaves and fruit.",
    "Apple Rust": "Rust fungus that creates orange/yellow spots on apple leaves.",
    "Corn Leaf Blight": "Fungal disease causing long, tan lesions on corn leaves.",
    "Corn Gray Spot": "Causes rectangular gray/brown lesions running parallel to leaf veins.",
    "Potato Early Blight": "Dark brown spots with concentric rings, usually on older leaves.",
    "Potato Late Blight": "Water-soaked lesions that turn brown/black — highly destructive.",
    "Tomato Bacterial Spot": "Small, water-soaked spots that turn brown with yellow halos.",
    "Tomato Leaf Mold": "Pale green/yellow patches on upper leaf surface, mold on underside.",
    "Tomato Mosaic Virus": "Mottled light/dark green pattern on leaves, stunted growth.",
    "Healthy Leaf": "No disease detected. The plant appears healthy!",
}


# ─────────────────────────────────────────────
# LOAD MODEL (cached so it only downloads once)
# ─────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_model():
    if not os.path.exists(MODEL_PATH):
        with st.spinner("Downloading model weights from Google Drive..."):
            url = f"https://drive.google.com/uc?id={GDRIVE_FILE_ID}"
            gdown.download(url, MODEL_PATH, quiet=False)
    return YOLO(MODEL_PATH)


# ─────────────────────────────────────────────
# RUN DETECTION ON A PIL IMAGE
# ─────────────────────────────────────────────
def run_detection(model, image: Image.Image, conf_threshold: float):
    img_array = np.array(image)
    results = model.predict(img_array, conf=conf_threshold, verbose=False)
    result = results[0]

    # Draw bounding boxes on the image
    annotated = result.plot()
    annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)

    # Collect detections
    detections = []
    for box in result.boxes:
        class_id = int(box.cls[0])
        class_name = model.names[class_id]
        confidence = float(box.conf[0])
        detections.append({"class": class_name, "confidence": confidence})

    return Image.fromarray(annotated_rgb), detections


# ─────────────────────────────────────────────
# RUN DETECTION ON A VIDEO FILE
# ─────────────────────────────────────────────
def run_video_detection(model, video_path: str, conf_threshold: float):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    out_path = video_path.replace(".mp4", "_detected.mp4")
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(out_path, fourcc, fps, (width, height))

    all_detections = []
    frame_count = 0
    progress = st.progress(0, text="Processing video frames...")
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        results = model.predict(frame, conf=conf_threshold, verbose=False)
        annotated = results[0].plot()
        out.write(annotated)

        for box in results[0].boxes:
            class_id = int(box.cls[0])
            class_name = model.names[class_id]
            confidence = float(box.conf[0])
            all_detections.append({"class": class_name, "confidence": confidence})

        frame_count += 1
        if total_frames > 0:
            progress.progress(
                min(frame_count / total_frames, 1.0),
                text=f"Processing frame {frame_count}/{total_frames}...",
            )

    cap.release()
    out.release()
    progress.empty()
    return out_path, all_detections


# ─────────────────────────────────────────────
# UI
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Plant Disease Detection",
    page_icon="🌿",
    layout="wide",
)

st.title("🌿 Plant Disease Detection")
st.markdown(
    "Upload a leaf image or video to detect diseases using a YOLOv8 model "
    "trained on 10+ plant disease classes."
)

# Sidebar
with st.sidebar:
    st.header("Settings")
    conf_threshold = st.slider(
        "Confidence Threshold",
        min_value=0.1,
        max_value=1.0,
        value=0.5,
        step=0.05,
        help="Detections below this confidence score will be ignored.",
    )
    st.divider()
    st.header("Supported Classes")
    for disease in DISEASE_INFO:
        st.markdown(f"- {disease}")

# Load model
try:
    model = load_model()
    st.success("Model loaded successfully.", icon="✅")
except Exception as e:
    st.error(f"Failed to load model: {e}")
    st.stop()

# Input mode tabs
tab_image, tab_video = st.tabs(["📷 Image Detection", "🎥 Video Detection"])

# ── IMAGE TAB ──
with tab_image:
    uploaded_image = st.file_uploader(
        "Upload a leaf image", type=["jpg", "jpeg", "png"], key="img_upload"
    )

    if uploaded_image:
        image = Image.open(uploaded_image).convert("RGB")
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Original Image")
            st.image(image, use_container_width=True)

        with st.spinner("Running detection..."):
            annotated_image, detections = run_detection(model, image, conf_threshold)

        with col2:
            st.subheader("Detection Results")
            st.image(annotated_image, use_container_width=True)

        if detections:
            st.subheader(f"Found {len(detections)} detection(s)")
            for det in detections:
                conf_pct = det["confidence"] * 100
                disease = det["class"]
                info = DISEASE_INFO.get(disease, "")
                color = "green" if disease == "Healthy Leaf" else "red"

                with st.expander(
                    f":{color}[{disease}] — {conf_pct:.1f}% confidence"
                ):
                    st.write(info)
        else:
            st.info(
                "No detections found. Try lowering the confidence threshold in the sidebar."
            )

# ── VIDEO TAB ──
with tab_video:
    uploaded_video = st.file_uploader(
        "Upload a plant video", type=["mp4", "mov", "avi"], key="vid_upload"
    )

    if uploaded_video:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
            tmp.write(uploaded_video.read())
            tmp_path = tmp.name

        st.video(tmp_path)

        if st.button("Run Detection on Video", type="primary"):
            with st.spinner("Analyzing video..."):
                out_path, detections = run_video_detection(
                    model, tmp_path, conf_threshold
                )

            st.success("Video processing complete!")
            st.subheader("Annotated Video")
            st.video(out_path)

            if detections:
                # Summarize detections
                from collections import Counter
                counts = Counter(d["class"] for d in detections)
                avg_conf = {}
                for d in detections:
                    avg_conf.setdefault(d["class"], []).append(d["confidence"])

                st.subheader("Detection Summary")
                for cls, count in counts.most_common():
                    avg = sum(avg_conf[cls]) / len(avg_conf[cls]) * 100
                    color = "green" if cls == "Healthy Leaf" else "red"
                    st.markdown(
                        f"**:{color}[{cls}]** — {count} detection(s), avg confidence {avg:.1f}%"
                    )
            else:
                st.info("No diseases detected in the video.")

            # Cleanup temp files
            os.unlink(tmp_path)
