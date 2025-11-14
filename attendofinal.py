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

# --- Configuration and Constants (OPTIMIZED FOR SPEED) ---
REGISTER_DIR = "registered_faces"
KNOWN_FACES_FILE = "known_faces.csv"
ATTENDANCE_DIR = "attendance_sheets"

# --- FASTER SETTINGS ---
THRESHOLD = 0.4  # Adjusted for Facenet (typical threshold is 0.4 for Euclidean distance)
MODEL_NAME = "Facenet" 
DETECTOR_BACKEND = "mtcnn" 

SAVE_INTERVAL_SECONDS = 5 

# Ensure directories exist
os.makedirs(REGISTER_DIR, exist_ok=True)
os.makedirs(ATTENDANCE_DIR, exist_ok=True)

# Initialize DeepFace model (cached)
@st.cache_resource
def load_deepface_model():
    """Load the DeepFace model only once."""
    st.sidebar.info(f"Loading DeepFace model: {MODEL_NAME}...")
    try:
        model = DeepFace.build_model(MODEL_NAME)
        st.sidebar.success(f"DeepFace Model '{MODEL_NAME}' loaded successfully.")
        return model
    except Exception as e:
        st.sidebar.error(f"Error loading DeepFace model: {e}")
        return None

# Load the model once
face_model = load_deepface_model()

# --- Helper Functions ---

@st.cache_data
def load_known_faces():
    """
    Loads known face encodings, names, and roll numbers from disk.
    Returns a tuple: (encodings, names, roll_nos)
    """
    known_face_encodings = []
    known_face_names = []
    known_roll_nos = []
    if os.path.exists(KNOWN_FACES_FILE):
        try:
            df = pd.read_csv(KNOWN_FACES_FILE)
            for _, row in df.iterrows():
                name = row['name']
                roll_no = row['roll_no'] 
                encoding_path = row['encoding_path']
                if os.path.exists(encoding_path):
                    with open(encoding_path, 'rb') as f:
                        encoding = pickle.load(f)
                        known_face_encodings.append(encoding)
                        known_face_names.append(name)
                        known_roll_nos.append(roll_no)
            st.sidebar.success(f"Loaded {len(known_face_names)} known faces.")
        except Exception as e:
            st.sidebar.error(f"Error loading known faces: {e}")
    else:
        st.sidebar.warning("No known faces found. Please register faces first.")
    return known_face_encodings, known_face_names, known_roll_nos

def save_face_encoding(name, roll_no, encoding):
    """
    Saves a face encoding to a pickle file and updates the known faces CSV.
    
    Returns: True if saved, False if duplicate.
    """
    # Load existing data
    if os.path.exists(KNOWN_FACES_FILE):
        df = pd.read_csv(KNOWN_FACES_FILE)
    else:
        df = pd.DataFrame(columns=['name', 'roll_no', 'encoding_path'])

    # --- DUPLICATE CHECK LOGIC ---
    if ((df['name'] == name) & (df['roll_no'] == roll_no)).any():
        return False # Indicate failure due to duplicate

    # --- SAVE LOGIC (if not duplicate) ---
    filename = f"{name.replace(' ', '_')}_{roll_no}.pkl" 
    filepath = os.path.join(REGISTER_DIR, filename)

    with open(filepath, 'wb') as f:
        pickle.dump(encoding, f)

    # Add new entry
    new_row = pd.DataFrame([{'name': name, 'roll_no': roll_no, 'encoding_path': filepath}])
    df = pd.concat([df, new_row], ignore_index=True)

    df.to_csv(KNOWN_FACES_FILE, index=False)
    load_known_faces.clear() 
    st.success(f"Successfully registered {name} (Roll No: {roll_no}).")
    return True # Indicate success

def extract_and_encode_face_deepface(img_array):
    """
    Uses DeepFace to detect faces and generate embeddings.
    Returns a list of (face_location, face_encoding) tuples.
    """
    if face_model is None:
        return []

    try:
        # Step 1: Face Extraction (Detection and Alignment)
        face_objs = DeepFace.extract_faces(
            img_path=img_array, 
            detector_backend=DETECTOR_BACKEND,
            enforce_detection=False
        )
    except Exception:
        return []

    results = []

    for face_obj in face_objs:
        if face_obj['confidence'] > 0.9: 
            x, y, w, h = face_obj['facial_area']['x'], face_obj['facial_area']['y'], face_obj['facial_area']['w'], face_obj['facial_area']['h']

            face_location = (y, x + w, y + h, x)
            face_img = face_obj['face']

            # Step 2: Face Representation (Encoding)
            embeddings = DeepFace.represent(
                img_path=face_img,
                model_name=MODEL_NAME, 
                enforce_detection=False,
                detector_backend='skip'
            )
            
            if embeddings:
                embedding = embeddings[0]['embedding']
                results.append((face_location, np.array(embedding)))

    return results

def get_deepface_distance(encoding1, encoding2):
    """Calculates the Euclidean distance between two DeepFace embeddings."""
    return np.linalg.norm(encoding1 - encoding2)

# --- Streamlit UI ---

st.title("👨‍🎓 Facial Recognition Attendance System (DeepFace)")

st.sidebar.header("Registered Students")
known_face_encodings, known_face_names, known_roll_nos = load_known_faces()
if known_face_names:
    st.sidebar.dataframe(pd.DataFrame({
        "Roll No.": known_roll_nos, 
        "Student Name": known_face_names
    }))
else:
    st.sidebar.info("No students registered yet.")

st.markdown("---")

## 1. Register Faces

st.header("Step 1: Register a New Student")
st.write("Ensure only one person is in the frame. Enter their details and take a picture, then click **Register Face**.")

person_name = st.text_input("Enter Student's Name:", key="name_input")
roll_no = st.text_input("Enter Student's Roll No.:", key="roll_no_input") 
uploaded_image = st.camera_input("Take a photo of the student", key="camera_reg")

# --- Robust Check for Inputs ---
is_name_valid = len(person_name.strip()) > 0
is_roll_no_valid = len(roll_no.strip()) > 0
is_photo_taken = uploaded_image is not None
is_ready_to_register = is_name_valid and is_roll_no_valid and is_photo_taken

register_button = st.button(
    "Register Face", 
    key="register_button",
    disabled=not is_ready_to_register 
)

# --- User Feedback ---
if not is_ready_to_register and register_button: 
    if not is_name_valid:
        st.info("Please enter the student's **Name**.")
    elif not is_roll_no_valid:
        st.info("Please enter the student's **Roll No.**.")
    elif not is_photo_taken:
        st.info("Please **take a photo** of the student.")


if register_button and is_ready_to_register:
    
    # Read the image from the camera input
    file_bytes = np.asarray(bytearray(uploaded_image.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    # --- SPEED-UP SECTION (using faster Facenet/mtcnn) ---
    with st.spinner('Processing face, please wait...'):
        face_data = extract_and_encode_face_deepface(image)
        face_encodings = [data[1] for data in face_data]
    
    if face_encodings:
        if len(face_encodings) == 1:
            # Attempt to save and check for success/duplicate
            saved_successfully = save_face_encoding(person_name, roll_no, face_encodings[0]) 
            
            if saved_successfully:
                 # Force rerun to clear camera input and reload data
                st.rerun()
            else:
                st.error(f"Registration Failed: Student '{person_name}' with Roll No. '{roll_no}' is already registered.")

        elif len(face_encodings) > 1:
            st.warning("Multiple faces detected! Please ensure only one person is in the frame.")
        else:
            st.warning("No face detected! Please try again with a clear photo.")
    else:
        st.warning("No face detected! Please try again with a clear photo.")

st.markdown("---")

## 2. Mark Attendance

st.header("Step 2: Start Live Attendance Marking")
st.write("Press the button to start the camera and mark attendance in real-time.")

# Camera selection UI
camera_choice = st.radio(
    "Select Camera:",
    ("Front Camera", "Back Camera"),
    horizontal=True,
    help="Select the camera to use."
)

start_button = st.button("Start Live Attendance", key="start_button")

if 'running' not in st.session_state:
    st.session_state['running'] = False
if 'last_save_time' not in st.session_state: 
    st.session_state['last_save_time'] = time.time()

if start_button:
    st.session_state['running'] = not st.session_state['running']

attendance_placeholder = st.empty()


if st.session_state['running']:
    st.warning("Live attendance is running. To stop, click 'Start Live Attendance' again.")
    st.info("Recognized students will be marked as 'Present'.")

    if not known_face_encodings:
        st.error("Cannot start attendance: No faces are registered!")
        st.session_state['running'] = False
        st.rerun() 

    today_date = datetime.now().strftime("%Y-%m-%d")
    attendance_file = os.path.join(ATTENDANCE_DIR, f"attendance_{today_date}.xlsx")

    if os.path.exists(attendance_file):
        attendance_df = pd.read_excel(attendance_file, engine='openpyxl')
    else:
        attendance_df = pd.DataFrame(columns=['Name', 'Roll No.', 'Status', 'Time']) 
        for i, name in enumerate(known_face_names):
            roll_no = known_roll_nos[i]
            if not ((attendance_df['Name'] == name) & (attendance_df['Roll No.'] == roll_no)).any():
                attendance_df.loc[len(attendance_df)] = [name, roll_no, 'Absent', '']

    frame_placeholder = st.empty()

    camera_index = 0
    if camera_choice == "Back Camera":
        try:
            # Check for a second camera index
            cap_test = cv2.VideoCapture(1)
            if cap_test.isOpened():
                camera_index = 1
            cap_test.release()
        except:
            camera_index = 0

    # --- FIX: Force OpenCV to use the stable Windows driver (DirectShow) ---
    cap = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)

    if not cap.isOpened():
        st.error("Error: Could not open webcam. Ensure the camera is not in use by another application.")
        st.session_state['running'] = False
        st.stop()
        
    # --- Camera Settings Reverted to Default ---
    # Running in default mode with forced driver is the most stable state.
    # ------------------------------------------

    try:
        while st.session_state['running']:
            ret, frame = cap.read()
            if not ret:
                # If frame reading fails, stop the loop.
                st.error("Failed to grab frame. Camera stream stopped.")
                st.session_state['running'] = False
                break
                
            # Frame processing (face detection/recognition) starts here
            face_data = extract_and_encode_face_deepface(frame)

            attendance_updated = False 

            for face_location, face_encoding in face_data:
                name = "Unknown"
                recognized_roll_no = ""

                face_distances = [
                    get_deepface_distance(known_encoding, face_encoding)
                    for known_encoding in known_face_encodings
                ]

                if face_distances:
                    best_match_index = np.argmin(face_distances)
                    min_distance = face_distances[best_match_index]

                    if min_distance < THRESHOLD:
                        name = known_face_names[best_match_index]
                        recognized_roll_no = known_roll_nos[best_match_index] 

                    if (name != "Unknown" and 
                        (attendance_df['Name'] == name) & (attendance_df['Roll No.'] == recognized_roll_no)).any():

                        current_status = attendance_df.loc[(attendance_df['Name'] == name) & (attendance_df['Roll No.'] == recognized_roll_no), 'Status'].iloc[0]
                        
                        if current_status != 'Present':
                            attendance_df.loc[(attendance_df['Name'] == name) & (attendance_df['Roll No.'] == recognized_roll_no), 'Status'] = 'Present'
                            attendance_df.loc[(attendance_df['Name'] == name) & (attendance_df['Roll No.'] == recognized_roll_no), 'Time'] = datetime.now().strftime("%H:%M:%S")
                            attendance_updated = True

                top, right, bottom, left = face_location

                display_text = f"{name} ({recognized_roll_no})" if recognized_roll_no else name
                
                cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
                cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
                cv2.putText(frame, display_text, (left + 6, bottom - 6), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 1)

            # Use use_container_width to remove deprecation warnings
            frame_placeholder.image(frame, channels="BGR", use_container_width=True)

            current_time = time.time()
            if attendance_updated or (current_time - st.session_state['last_save_time'] > SAVE_INTERVAL_SECONDS):
                attendance_df.to_excel(attendance_file, index=False, engine='openpyxl')
                st.session_state['last_save_time'] = current_time

            with attendance_placeholder.container():
                st.subheader(f"Attendance for {today_date}")
                st.dataframe(attendance_df, use_container_width=True)

            time.sleep(0.05) 

    finally:
        cap.release()
        attendance_df.to_excel(attendance_file, index=False, engine='openpyxl')
        st.session_state['running'] = False
