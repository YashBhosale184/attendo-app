import streamlit as st
import os
import pandas as pd
import numpy as np
from deepface import DeepFace
from datetime import datetime
from PIL import Image
import cv2 # DeepFace dependency

# --- Configuration ---
DATABASE_PATH = "face_database"
LOG_FILE = "attendance_log.csv"
MODEL = "VGG-Face" # Using VGG-Face for stability
DETECTOR = "opencv" # Simple and reliable detector
st.set_page_config(layout="wide", page_title="DeepFace Attendance System")
# --- End Configuration ---


# --- Utility Functions ---

def ensure_directories_and_state():
    """Initializes session state, database folder, and log file."""
    if 'roll_map' not in st.session_state:
        # Load existing roll map or initialize empty
        st.session_state.roll_map = {}
        st.session_state.log_df = pd.DataFrame(columns=["Date", "Time", "Name", "Roll_No", "Status"])

    # 1. Database Directory
    os.makedirs(DATABASE_PATH, exist_ok=True)
    
    # 2. Attendance Log
    if not os.path.exists(LOG_FILE):
        st.session_state.log_df.to_csv(LOG_FILE, index=False)
    else:
        # Load existing log
        st.session_state.log_df = pd.read_csv(LOG_FILE)
        
    # 3. Load Roll Map from existing folders
    for person_name in os.listdir(DATABASE_PATH):
        person_dir = os.path.join(DATABASE_PATH, person_name)
        if os.path.isdir(person_dir):
            if person_name not in st.session_state.roll_map:
                 # If only folder exists, assign a temporary N/A roll number
                st.session_state.roll_map[person_name] = "UNKNOWN" 

def log_attendance(name, roll_no, status="PRESENT"):
    """Appends attendance record to the log file and updates session state."""
    now = datetime.now()
    date_str = now.strftime("%Y-%m-%d")
    time_str = now.strftime("%H:%M:%S")

    # Check if person already logged attendance today
    df_current = st.session_state.log_df
    already_present = ((df_current['Date'] == date_str) & (df_current['Name'] == name)).any()
    
    if already_present:
        st.warning(f"⚠️ {name} already marked present today. Skipping log update.")
        return False
            
    new_entry = pd.DataFrame([{
        "Date": date_str,
        "Time": time_str,
        "Name": name,
        "Roll_No": roll_no,
        "Status": status
    }])
    
    # Update state and CSV
    st.session_state.log_df = pd.concat([df_current, new_entry], ignore_index=True)
    st.session_state.log_df.to_csv(LOG_FILE, index=False)
    return True

# --- Registration Tab Functions ---

def register_face(uploaded_file, name, roll_no):
    """Saves the uploaded face image to the database and updates the roll map."""
    
    # Check inputs
    if not name or not roll_no:
        st.error("Name and Roll Number cannot be empty.")
        return
    if uploaded_file is None:
        st.error("Please upload an image for registration.")
        return

    try:
        # 1. Save Image
        person_dir = os.path.join(DATABASE_PATH, name.strip())
        os.makedirs(person_dir, exist_ok=True)
        image_path = os.path.join(person_dir, uploaded_file.name)
        
        # Save the uploaded file content
        with open(image_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        # 2. Check for face quality (optional but recommended)
        img = cv2.imread(image_path)
        if img is None:
            st.error("Could not load the saved image file.")
            os.remove(image_path)
            return

        detected_faces = DeepFace.extract_faces(img, detector_backend=DETECTOR, enforce_detection=False)
        if not detected_faces:
             st.warning(f"Face not clearly detected in {name}'s image. Registration recorded, but recognition might fail.")
        else:
             st.success(f"Face detected and registered for {name}.")

        # 3. Update Roll Map
        st.session_state.roll_map[name.strip()] = roll_no.strip()

        st.success(f"✅ Successfully registered {name} (Roll: {roll_no})!")

    except Exception as e:
        st.error(f"An error occurred during registration: {e}")

# --- Attendance Tab Functions ---

def check_attendance(captured_image):
    """Processes a captured image to check for a match in the database."""
    if captured_image is None:
        st.error("Please capture or upload an image to check attendance.")
        return

    # Convert Streamlit uploaded file/camera image to a format DeepFace can use
    try:
        # Save image temporarily
        img_np = np.array(Image.open(captured_image).convert('RGB'))
        # Convert RGB to BGR for OpenCV/DeepFace
        img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
        temp_path = "temp_capture.jpg"
        cv2.imwrite(temp_path, img_bgr)
        
        # 1. Run DeepFace Identification
        st.info("🔎 Analyzing face for match...")
        
        # Ensure the database is not empty before finding
        if not os.listdir(DATABASE_PATH) or not any(os.listdir(os.path.join(DATABASE_PATH, d)) for d in os.listdir(DATABASE_PATH) if os.path.isdir(os.path.join(DATABASE_PATH, d))):
             st.error("❌ Face database is empty. Please register faces first in the 'Register Faces' tab.")
             os.remove(temp_path)
             return

        # DeepFace.find() will run the identification
        results = DeepFace.find(
            img_path=temp_path,
            db_path=DATABASE_PATH,
            model_name=MODEL,
            detector_backend=DETECTOR,
            enforce_detection=True # We require a face for attendance
        )
        
        # Clean up temporary file
        os.remove(temp_path)

        # 2. Process Results
        
        # DeepFace returns a list of dataframes, one for each face found in the input.
        # We assume only one main face is in the attendance photo.
        results_df = results[0] 

        if results_df.empty:
            st.warning("❌ No matching face found in the database. Please try again.")
            return

        # Get the closest match (first row)
        closest_match_path = results_df.iloc[0]['identity']
        
        # Extract the person's name (the sub-folder name)
        person_name = os.path.basename(os.path.dirname(closest_match_path))
        
        # Get the corresponding roll number from the map
        roll_no = st.session_state.roll_map.get(person_name, "N/A - Check Map")

        # 3. Log and Display
        if log_attendance(person_name, roll_no):
            st.balloons()
            st.success(f"🎉 ATTENDANCE MARKED! Welcome, {person_name} (Roll: {roll_no})!")
        
    except ValueError as e:
        # This often happens if no face is detected in the input image
        st.error(f"❌ Face detection failed. Please ensure your face is clearly visible. Details: {e}")
        if os.path.exists(temp_path): os.remove(temp_path)
    except Exception as e:
        st.error(f"An unexpected error occurred during attendance check: {e}")
        if os.path.exists(temp_path): os.remove(temp_path)


# --- Streamlit UI Layout ---

st.title("👨‍🏫 DeepFace Automated Attendance System")
st.markdown("Use this application to **Register** new faces and **Mark Attendance** automatically.")

# Initialize the state and folders
ensure_directories_and_state()

# Create tabs for clean separation
tab_register, tab_attendance, tab_log = st.tabs(["1. Register Faces", "2. Mark Attendance", "3. View Log"])


# --- TAB 1: Register Faces ---
with tab_register:
    st.header("👤 Face Registration")
    st.markdown("Upload one or more clear photos for each person to build the database.")

    col1, col2 = st.columns(2)

    with col1:
        new_name = st.text_input("Full Name:", key="reg_name").strip()
        new_roll_no = st.text_input("Roll Number:", key="reg_roll").strip()

        # Input for image file upload
        uploaded_reg_file = st.file_uploader(
            "Upload Registration Photo (JPEG/PNG)", 
            type=['jpg', 'jpeg', 'png'],
            key="reg_file"
        )
        
        if st.button("Register Person", type="primary"):
            if uploaded_reg_file:
                register_face(uploaded_reg_file, new_name, new_roll_no)
            else:
                st.error("Please upload an image.")

    with col2:
        st.subheader("Current Registered Database")
        
        if os.path.exists(DATABASE_PATH) and os.listdir(DATABASE_PATH):
            registered_df_data = []
            for folder_name in os.listdir(DATABASE_PATH):
                folder_path = os.path.join(DATABASE_PATH, folder_name)
                if os.path.isdir(folder_path):
                    file_count = len([f for f in os.listdir(folder_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
                    roll = st.session_state.roll_map.get(folder_name, "UNKNOWN")
                    registered_df_data.append({"Name": folder_name, "Roll No.": roll, "Image Count": file_count})
            
            if registered_df_data:
                st.dataframe(pd.DataFrame(registered_df_data), use_container_width=True)
            else:
                st.info("Database directory is empty.")
        else:
             st.info("No faces registered yet.")


# --- TAB 2: Mark Attendance ---
with tab_attendance:
    st.header("📸 Check-in for Attendance")
    st.markdown("Use your webcam or upload a photo to check for a match against the registered faces.")

    col_camera, col_check = st.columns(2)
    
    with col_camera:
        # Option 1: Live Camera Input
        st.subheader("Option A: Use Webcam")
        camera_input = st.camera_input("Take a photo now")

    with col_check:
        # Option 2: File Upload Input
        st.subheader("Option B: Upload Photo")
        uploaded_check_file = st.file_uploader(
            "Upload Photo (JPEG/PNG)", 
            type=['jpg', 'jpeg', 'png'],
            key="check_file"
        )

    # Attendance check logic (triggered by button, handling either input)
    if st.button("Mark Attendance", type="primary"):
        if camera_input:
            check_attendance(camera_input)
        elif uploaded_check_file:
            check_attendance(uploaded_check_file)
        else:
            st.error("Please use the camera or upload a file.")


# --- TAB 3: View Log ---
with tab_log:
    st.header("📋 Attendance Log (CSV Output)")
    
    st.dataframe(st.session_state.log_df, use_container_width=True)
    
    # Download button for the CSV file
    csv = st.session_state.log_df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="Download Attendance Log as CSV",
        data=csv,
        file_name='attendance_log.csv',
        mime='text/csv',
    )
    
    st.markdown("*(This log file is saved locally as `attendance_log.csv`)*") 