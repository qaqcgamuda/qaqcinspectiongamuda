import io
import json
import base64
import cv2
from datetime import datetime
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
import requests
import streamlit as st
from PIL import Image
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input


APP_TITLE = "AI-Based QAQC Inspection System"
MODEL_PATH = "fine_tuned_mobilenetv2_defect_model.h5"
ALLOWED_DOMAIN = "gamuda.com.my"   
GOOGLE_SHEETS_WEBHOOK_URL = "https://script.google.com/macros/s/AKfycbxoMvixenlh6PpOtDw0LgwB6lVRwAwI6kj6wdZyQ3Kit3icsUjWGh7W3AcA0BypuGby0Q/exec"  

CLASS_NAMES = [
    "exposed_surface",
    "good_surface",
    "honeycomb",
    "loose_grout",
    "segregation",
]

PASS_CLASSES = {"good_surface"}
IMAGE_SIZE = (224, 224)

LOCAL_LOG_DIR = Path("inspection_logs")
LOCAL_IMAGE_DIR = Path("inspection_images")
LOCAL_LOG_DIR.mkdir(exist_ok=True)
LOCAL_IMAGE_DIR.mkdir(exist_ok=True)


st.set_page_config(
    page_title=APP_TITLE,
    page_icon="🧱",
    layout="centered",
)


@st.cache_resource
def get_model():
    if not Path(MODEL_PATH).exists():
        raise FileNotFoundError(
            f"Model file not found: {MODEL_PATH}. Put the real .h5 file in the same folder as this app."
        )
    return load_model(MODEL_PATH)


def normalize_user_sheet_name(email: str) -> str:
    username = email.split("@")[0].replace(".", "_")
    return f"QAQC_{username}_Inspection_Log"


def email_is_allowed(email: str) -> bool:
    return email.strip().lower().endswith("@gamuda.com.my")


def pil_to_model_array(img: Image.Image) -> np.ndarray:
    img = img.convert("RGB")
    img = img.resize(IMAGE_SIZE)
    arr = np.array(img, dtype=np.float32)
    arr = np.expand_dims(arr, axis=0)
    arr = preprocess_input(arr)
    return arr


def predict_defect(model, img: Image.Image) -> Tuple[str, float, np.ndarray]:
    arr = pil_to_model_array(img)
    preds = model.predict(arr, verbose=0)[0]
    idx = int(np.argmax(preds))
    defect = CLASS_NAMES[idx]
    confidence = float(preds[idx])
    return defect, confidence, preds


def detect_regions(image: Image.Image, model, grid_size=3, threshold=0.5):
    width, height = image.size
    results = []

    patch_w = width // grid_size
    patch_h = height // grid_size

    for i in range(grid_size):
        for j in range(grid_size):
            left = j * patch_w
            top = i * patch_h
            right = left + patch_w
            bottom = top + patch_h

            patch = image.crop((left, top, right, bottom))

            defect, confidence, _ = predict_defect(model, patch)

            if confidence > threshold and defect != "good_surface":
                results.append({
                    "box": (left, top, right, bottom),
                    "label": defect,
                    "confidence": confidence
                })

    return results


def draw_boxes(image: Image.Image, detections):
    img = np.array(image)

    color_map = {
        "honeycomb": (255, 0, 0),      # Blue
        "segregation": (0, 255, 255),  # Yellow
        "loose_grout": (0, 0, 255),    # Red
        "exposed_surface": (255, 255, 0),
    }

    for det in detections:
        x1, y1, x2, y2 = det["box"]
        label = det["label"]
        conf = det["confidence"]

        color = color_map.get(label, (255, 255, 255))

        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        cv2.putText(
            img,
            f"{label} ({conf:.2f})",
            (x1, y1 - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            2
        )

    return img

def qaqc_decision(defect: str) -> str:
    return "PASS" if defect in PASS_CLASSES else "FAIL"


def timestamp_strings():
    now = datetime.now()
    return now.strftime("%Y-%m-%d"), now.strftime("%H:%M:%S"), now.strftime("%Y%m%d_%H%M%S")


def save_image_locally(img: Image.Image, email: str, filename_hint: str) -> Path:
    _, _, compact_ts = timestamp_strings()
    username = email.split("@")[0].replace(".", "_")
    ext = Path(filename_hint).suffix.lower() or ".jpg"
    safe_name = f"{username}_{compact_ts}{ext}"
    save_path = LOCAL_IMAGE_DIR / safe_name
    img.convert("RGB").save(save_path)
    return save_path


def append_local_record(record: dict) -> Path:
    sheet_name = normalize_user_sheet_name(record["user_email"])
    csv_path = LOCAL_LOG_DIR / f"{sheet_name}.csv"

    row_df = pd.DataFrame([record])
    if csv_path.exists():
        existing = pd.read_csv(csv_path)
        combined = pd.concat([existing, row_df], ignore_index=True)
    else:
        combined = row_df

    combined.to_csv(csv_path, index=False)
    return csv_path


def image_to_base64(img: Image.Image) -> str:
    buffer = io.BytesIO()
    img.convert("RGB").save(buffer, format="JPEG", quality=90)
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def send_to_google_sheets_webhook(record: dict, img: Image.Image):
    if not GOOGLE_SHEETS_WEBHOOK_URL:
        return False, "Webhook not configured. Saved locally only."

    payload = {
    	"timestamp": record["timestamp"],
    	"date": record["date"],
    	"time": record["time"],
    	"user_email": record["user_email"],
    	"sheet_name": record["sheet_name"],
    	"file_name": record["file_name"],
    	"defect_type": record["defect_type"],
    	"confidence": record["confidence"],
    	"qaqc_result": record["qaqc_result"],
    	"remarks": record["remarks"],
    	"local_image_path": "",
    	"image_base64": image_to_base64(img),
	}
	

    try:
        response = requests.post(
            GOOGLE_SHEETS_WEBHOOK_URL,
            headers={"Content-Type": "application/json"},
            data=json.dumps(payload),
            timeout=30,
        )
        if response.ok:
            return True, response.text
        return False, f"Webhook error {response.status_code}: {response.text}"
    except Exception as exc:
        return False, f"Webhook request failed: {exc}"


def defect_label(defect: str) -> str:
    return defect.replace("_", " ").title()


def reset_prediction_state():
    st.session_state["prediction_ready"] = False
    st.session_state["predicted_defect"] = None
    st.session_state["confidence"] = None
    st.session_state["probs"] = None
    st.session_state["qaqc_result"] = None


for key, value in {
    "authenticated": False,
    "user_email": "",
    "prediction_ready": False,
    "predicted_defect": None,
    "confidence": None,
    "probs": None,
    "qaqc_result": None,
}.items():
    st.session_state.setdefault(key, value)


st.title(APP_TITLE)
st.caption("Online prototype for defect detection, QAQC decision support, and inspection record logging.")


with st.container(border=True):
    st.subheader("1. Inspector Access")
    st.write("Enter your domain email to continue. This prototype uses one user = one inspection log.")

    email_input = st.text_input(
        "Domain Email",
        value=st.session_state["user_email"],
        placeholder=f"name@{ALLOWED_DOMAIN}",
    )

    col1, col2 = st.columns(2)

    with col1:
        if st.button("Start Inspection Session", use_container_width=True):
            if not email_input.strip():
                st.error("Please enter your email.")
            elif not email_is_allowed(email_input):
                st.error(f"Only @{ALLOWED_DOMAIN} accounts are allowed.")
            else:
                st.session_state["authenticated"] = True
                st.session_state["user_email"] = email_input.strip().lower()
                st.success("Inspection session started.")

    with col2:
        if st.button("Reset Session", use_container_width=True):
            st.session_state["authenticated"] = False
            st.session_state["user_email"] = ""
            reset_prediction_state()
            st.rerun()


if not st.session_state["authenticated"]:
    st.info("Sign in to continue.")
    st.stop()


try:
    model = get_model()
except Exception as exc:
    st.error(str(exc))
    st.stop()


st.success(f"Inspector: {st.session_state['user_email']}")
st.write(f"Inspection log: **{normalize_user_sheet_name(st.session_state['user_email'])}**")


with st.container(border=True):
    st.subheader("2. Capture or Upload Inspection Image")

    source = st.radio(
        "Image Source",
        options=["Camera", "Upload"],
        horizontal=True,
    )

    selected_file = None

    if source == "Camera":
        selected_file = st.camera_input("Capture site image")
    else:
        selected_file = st.file_uploader(
            "Upload site image",
            type=["jpg", "jpeg", "png"],
        )

    if selected_file is not None:
        pil_img = Image.open(selected_file)
        st.image(pil_img, caption="Image Preview", use_container_width=True)
        st.write(f"**File Name:** {selected_file.name}")
    else:
        pil_img = None


with st.container(border=True):
    st.subheader("3. AI QAQC Inspection")

    col1, col2 = st.columns([2, 1])

    with col1:
        inspect_clicked = st.button("Run QAQC Inspection", use_container_width=True, type="primary")
    with col2:
        clear_clicked = st.button("Clear Result", use_container_width=True)

    if clear_clicked:
        reset_prediction_state()
        st.rerun()

    if inspect_clicked:
		if pil_img is None:
        st.error("Please capture or upload an image first.")
    else:
        with st.spinner("Analyzing defect image..."):

            # ORIGINAL prediction
            defect, confidence, probs = predict_defect(model, pil_img)
            decision = qaqc_decision(defect)

            # 🔥 NEW: PATCH DETECTION
            detections = detect_regions(pil_img, model, grid_size=3, threshold=0.5)
            boxed_img = draw_boxes(pil_img, detections)

            # SAVE RESULT
            st.session_state["prediction_ready"] = True
            st.session_state["predicted_defect"] = defect
            st.session_state["confidence"] = confidence
            st.session_state["probs"] = probs
            st.session_state["qaqc_result"] = decision
            st.session_state["boxed_image"] = boxed_img

	
    if st.session_state["prediction_ready"]:
    if "boxed_image" in st.session_state:
    st.image(
        st.session_state["boxed_image"],
        caption="Detected Defect Areas",
        use_container_width=True
    )
        defect = st.session_state["predicted_defect"]
        confidence = st.session_state["confidence"]
        decision = st.session_state["qaqc_result"]

        st.markdown("### Inspection Result")
        st.write(f"**Detected Defect:** {defect_label(defect)}")
        st.write(f"**Confidence:** {confidence * 100:.2f}%")

        if decision == "PASS":
            st.success("QAQC Decision: PASS")
        else:
            st.error("QAQC Decision: FAIL")

        probs_df = pd.DataFrame({
            "Class": [defect_label(name) for name in CLASS_NAMES],
            "Probability": [float(x) for x in st.session_state["probs"]],
        }).sort_values("Probability", ascending=False)

        st.dataframe(probs_df, use_container_width=True, hide_index=True)


with st.container(border=True):
    st.subheader("4. Save Inspection Record")
    st.caption("This saves a local backup and can also send the record to Google Sheets later.")

    remarks = st.text_area("Remarks (optional)", placeholder="Add inspector notes if needed.")
    save_clicked = st.button("Save Inspection Record", use_container_width=True)

    if save_clicked:
        if pil_img is None:
            st.error("No image available to save.")
        elif not st.session_state["prediction_ready"]:
            st.error("Run QAQC Inspection first.")
        else:
            date_str, time_str, _ = timestamp_strings()
            defect = st.session_state["predicted_defect"]
            confidence = st.session_state["confidence"]
            decision = st.session_state["qaqc_result"]
            email = st.session_state["user_email"]
            file_name = selected_file.name if selected_file is not None else "captured_image.jpg"
            sheet_name = normalize_user_sheet_name(email)

            record = {
                "timestamp": f"{date_str} {time_str}",
                "date": date_str,
                "time": time_str,
                "user_email": email,
                "sheet_name": sheet_name,
                "file_name": file_name,
                "defect_type": defect,
                "confidence": round(confidence * 100, 2),
                "qaqc_result": decision,
                "remarks": remarks.strip()
            }

            webhook_ok, webhook_msg = send_to_google_sheets_webhook(record, pil_img) 
		

            if webhook_ok:
                st.success("Inspection saved to Google Sheets.")
            else:
                st.error(webhook_msg)

            st.markdown("### Saved Record Summary")
            st.image(pil_img, caption="Inspection Image", use_column_width=True)
            st.write(f"**Timestamp:** {record['timestamp']}")
            st.write(f"**File Name:** {record['file_name']}")
            st.write(f"**Defect Type:** {defect_label(record['defect_type'])}")
            st.write(f"**QAQC Result:** {record['qaqc_result']}")
            st.write(f"**User Sheet:** {record['sheet_name']}")
