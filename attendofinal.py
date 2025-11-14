#pip install deepface streamlit opencv-python pandas openpyxl numpy

import streamlit as st
import cv2
import pandas as pd
import numpy as np
import os
from datetime import datetime
import pickle
import time

# --- DeepFace Imports ---
from deepface import DeepFace

# --- Configuration and Constants ---
REGISTER_DIR = "registered_faces"
KNOWN_FACES_FILE = "known_faces.csv"
ATTENDANCE_DIR = "attendance_sheets"

THRESHOLD = 0.4
MODEL_NAME = "Facenet"
DETECTOR_BACKEND = "mtcnn"

SAVE_INTERVAL_SECONDS = 5 

# Ensure directories exist
os.makedirs(REGISTER_DIR, exist_ok=True)
os.makedirs(ATTENDANCE_DIR, exist_ok=True)

# Load model once
@st.cache_resource
def load_deepface_model():
    st.sidebar.info(f"Loading DeepFace model: {MODEL_NAME}...")
    try:
        model = DeepFace.build_model(MODEL_NAME)
        st.sidebar.success("Model loaded!")
        return model
    except Exception as e:
        st.sidebar.error(f"Model Error: {e}")
        return None

face_model = load_deepface_model()


# ---------------------- Helper Functions ----------------------

@st.cache_data
def load_known_faces():
    encodings, names, rolls = [], [], []

    if os.path.exists(KNOWN_FACES_FILE):
        try:
            df = pd.read_csv(KNOWN_FACES_FILE)
            for _, row in df.iterrows():
                if os.path.exists(row["encoding_path"]):
                    with open(row["encoding_path"], "rb") as f:
                        enc = pickle.load(f)
                        encodings.append(enc)
                        names.append(row["name"])
                        rolls.append(row["roll_no"])
            st.sidebar.success(f"Loaded {len(names)} faces.")
        except:
            st.sidebar.error("Error loading known faces.")
    else:
        st.sidebar.info("No registered faces found.")

    return encodings, names, rolls


def save_face_encoding(name, roll_no, encoding):
    if os.path.exists(KNOWN_FACES_FILE):
        df = pd.read_csv(KNOWN_FACES_FILE)
    else:
        df = pd.DataFrame(columns=["name", "roll_no", "encoding_path"])

    # duplicate check
    if ((df["name"] == name) & (df["roll_no"] == roll_no)).any():
        return False

    file_path = os.path.join(REGISTER_DIR, f"{name}_{roll_no}.pkl")
    with open(file_path, "wb") as f:
        pickle.dump(encoding, f)

    new_row = pd.DataFrame([{
        "name": name,
        "roll_no": roll_no,
        "encoding_path": file_path
    }])

    df = pd.concat([df, new_row], ignore_index=True)
    df.to_csv(KNOWN_FACES_FILE, index=False)

    load_known_faces.clear()
    return True


def extract_and_encode_face_deepface(img_array):
    if face_model is None:
        return []

    try:
        faces = DeepFace.extract_faces(
            img_path=img_array,
            detector_backend=DETECTOR_BACKEND,
            enforce_detection=False
        )
    except:
        return []

    results = []

    for f in faces:
        if f["confidence"] > 0.9:
            x, y, w, h = f["facial_area"]["x"], f["facial_area"]["y"], f["facial_area"]["w"], f["facial_area"]["h"]
            face_loc = (y, x+w, y+h, x)
            face_crop = f["face"]

            emb = DeepFace.represent(
                img_path=face_crop,
                model_name=MODEL_NAME,
                enforce_detection=False,
                detector_backend="skip"
            )

            if emb:
                results.append((face_loc, np.array(emb[0]["embedding"])))

    return results


def get_deepface_distance(e1, e2):
    return np.linalg.norm(e1 - e2)


# ------------------------ Streamlit UI ------------------------

st.title("👨‍🎓 DeepFace Attendance System")

st.sidebar.header("Registered Students")
known_encs, known_names, known_rolls = load_known_faces()

if known_names:
    st.sidebar.dataframe(pd.DataFrame({"Roll No": known_rolls, "Name": known_names}))
else:
    st.sidebar.info("No registered faces yet.")

st.markdown("---")

# ------------------------ 1. REGISTER STUDENT ------------------------

st.header("Step 1: Register Student")

name = st.text_input("Student Name")
roll = st.text_input("Roll Number")
photo = st.camera_input("Take Student Photo")

if st.button("Register Face"):
    if not (name and roll and photo):
        st.warning("Enter name, roll number and take a picture!")
    else:
        file_bytes = np.asarray(bytearray(photo.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        with st.spinner("Processing face..."):
            data = extract_and_encode_face_deepface(img)

        if len(data) == 0:
            st.error("No face detected!")
        elif len(data) > 1:
            st.warning("Multiple faces detected. Only one allowed.")
        else:
            encoding = data[0][1]
            ok = save_face_encoding(name, roll, encoding)
            if ok:
                st.success(f"{name} registered successfully!")
                st.rerun()
            else:
                st.error("Student already registered!")

st.markdown("---")

# ------------------------ 2. LIVE ATTENDANCE ------------------------

st.header("Step 2: Start Live Attendance")

camera_choice = st.radio("Select Camera:", ("Front Camera", "Back Camera"), horizontal=True)

if "running" not in st.session_state:
    st.session_state.running = False

if st.button("Start / Stop Attendance"):
    st.session_state.running = not st.session_state.running

placeholder = st.empty()

if st.session_state.running:

    if not known_encs:
        st.error("No registered faces available.")
        st.session_state.running = False
        st.stop()

    st.warning("Attendance Running... Click again to stop.")

    today = datetime.now().strftime("%Y-%m-%d")
    file_path = os.path.join(ATTENDANCE_DIR, f"attendance_{today}.xlsx")

    if os.path.exists(file_path):
        df_att = pd.read_excel(file_path)
    else:
        df_att = pd.DataFrame({
            "Name": known_names,
            "Roll No": known_rolls,
            "Status": ["Absent"] * len(known_names),
            "Time": ["" for _ in known_names]
        })

    # ------------------ FIXED CAMERA INIT ------------------

    cam_index = 0
    if camera_choice == "Back Camera":
        cam_index = 1

    cap = cv2.VideoCapture(cam_index)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    time.sleep(1)  # camera warmup

    frame_box = st.empty()

    while st.session_state.running:
        ret, frame = cap.read()

        if not ret:
            st.error("Cannot read camera frame!")
            break

        faces = extract_and_encode_face_deepface(frame)

        updated = False

        for loc, enc in faces:
            name = "Unknown"
            roll_num = ""

            dists = [get_deepface_distance(k, enc) for k in known_encs]

            if dists:
                idx = np.argmin(dists)
                if dists[idx] < THRESHOLD:
                    name = known_names[idx]
                    roll_num = known_rolls[idx]

            # mark attendance
            if name != "Unknown":
                mask = (df_att["Name"] == name) & (df_att["Roll No"] == roll_num)
                if df_att.loc[mask, "Status"].iloc[0] != "Present":
                    df_att.loc[mask, "Status"] = "Present"
                    df_att.loc[mask, "Time"] = datetime.now().strftime("%H:%M:%S")
                    updated = True

            # draw face box
            top, right, bottom, left = loc
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
            cv2.putText(
                frame, f"{name} {roll_num}",
                (left, bottom - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9, (255, 255, 255), 2
            )

        frame_box.image(frame, channels="BGR")

        if updated:
            df_att.to_excel(file_path, index=False)

        placeholder.dataframe(df_att, use_container_width=True)

        time.sleep(0.05)

    cap.release()
    df_att.to_excel(file_path, index=False)
