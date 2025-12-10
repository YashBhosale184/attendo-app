# Configuration Constants
DATABASE_PATH = "face_database"
LOG_FILE = "attendance_log.csv"
MODEL = "VGG-Face"  
DETECTOR = "mtcnn" 

st.set_page_config(layout="wide", page_title="DeepFace Attendance System (Webcam Only - MTCNN)")

# Initialization and Setup Functions

def ensure_directories_and_state():
    """Initializes session state, database directory, and attendance log."""
    if 'roll_map' not in st.session_state:
        st.session_state.roll_map = {}
    if 'log_df' not in st.session_state:
        st.session_state.log_df = pd.DataFrame(columns=["Date", "Time", "Name", "Roll_No", "Status"])

    # 1. Database Directory
    os.makedirs(DATABASE_PATH, exist_ok=True)
    
    # 2. Attendance Log
    if not os.path.exists(LOG_FILE):
        st.session_state.log_df.to_csv(LOG_FILE, index=False)
    else:
        # Load existing log
        try:
            st.session_state.log_df = pd.read_csv(LOG_FILE)
        except Exception as e:
            st.error(f"Error loading log file: {e}. Starting with an empty log.")
            st.session_state.log_df = pd.DataFrame(columns=["Date", "Time", "Name", "Roll_No", "Status"])
    
    # 3. Load Roll Map from existing folders
    for person_name in os.listdir(DATABASE_PATH):
        person_dir = os.path.join(DATABASE_PATH, person_name)
        if os.path.isdir(person_dir):
            if person_name not in st.session_state.roll_map:
                 st.session_state.roll_map[person_name] = "UNKNOWN" 


def log_attendance(name, roll_no, status="PRESENT"):
    """Appends attendance record, ensuring a person is logged only once per day."""
    now = datetime.now()
    date_str = now.strftime("%Y-%m-%d")
    time_str = now.strftime("%H:%M:%S")

    df_current = st.session_state.log_df
    already_present = ((df_current['Date'] == date_str) & (df_current['Name'] == name)).any()
    
    if already_present:
        st.warning(f"⚠️ {name} (Roll: {roll_no}) is already marked present today. Skipping log update.")
        return False
            
    new_entry = pd.DataFrame([{
        "Date": date_str,
        "Time": time_str,
        "Name": name,
        "Roll_No": roll_no,
        "Status": status
    }])
    
    st.session_state.log_df = pd.concat([df_current, new_entry], ignore_index=True)
    st.session_state.log_df.to_csv(LOG_FILE, index=False)
    return True

def register_face(uploaded_file, name, roll_no):
    """Saves the image to the database and updates the roll map."""
    
    name = name.strip()
    roll_no = roll_no.strip()

    if not name or not roll_no:
        st.error("Name and Roll Number cannot be empty.")
        return
    if uploaded_file is None:
        st.error("Please upload an image for registration.")
        return

    try:
        person_dir = os.path.join(DATABASE_PATH, name)
        os.makedirs(person_dir, exist_ok=True)
        image_path = os.path.join(person_dir, f"face_{len(os.listdir(person_dir)) + 1}.jpg")
        
        with open(image_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        img = cv2.imread(image_path)
        if img is None:
            st.error("Could not load the saved image file.")
            os.remove(image_path)
            return

        # Check face presence using the powerful MTCNN detector
        detected_faces = DeepFace.extract_faces(img, detector_backend=DETECTOR, enforce_detection=False)
        
        if not detected_faces or detected_faces[0]['confidence'] < 0.9: 
            st.warning(f"Face not confidently detected in {name}'s image. Please re-register with a clearer photo.")
        else:
            st.success(f"Face confidently detected and registered for {name}.")

        st.session_state.roll_map[name] = roll_no

        st.success(f"✅ Successfully registered {name} (Roll: {roll_no})!")

    except Exception as e:
        st.error(f"An error occurred during registration: {e}")
        st.exception(e) 

# --- Attendance Tab Functions ---

def check_attendance(captured_image):
    """
    Processes a captured image to check for a match and logs attendance.
    Includes robust error handling.
    """
    
    if captured_image is None:
        st.error("Please capture a photo using the webcam to check attendance.")
        return

    temp_path = "temp_capture.jpg"
    try:
        st.info("Converting image for analysis...")
        
        img_pil = Image.open(captured_image).convert('RGB')
        img_np = np.array(img_pil)
        img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
        
        cv2.imwrite(temp_path, img_bgr)
        
        if not os.path.exists(DATABASE_PATH) or not os.listdir(DATABASE_PATH):
            st.error("❌ Face database is empty. Please register faces first in the 'Register Faces' tab.")
            return

        st.info(f"🔎 Analyzing face for match")
        
        # DeepFace Identification
        results = DeepFace.find(
            img_path=temp_path,
            db_path=DATABASE_PATH,
            model_name=MODEL,
            detector_backend=DETECTOR, # Using MTCNN here
            enforce_detection=True  
        )
        
        os.remove(temp_path)

        # FIX FOR ALL "Length of values does not match" ERRORS
        if not results:
             st.warning("❌ Face not detected, or the DeepFace search returned no results. Ensure your face is clear and centered.")
             return
        
        results_df = results[0] 

        if results_df.empty:
            st.warning("❌ No matching face found in the database. Please try again.")
            return
        

        closest_match_path = results_df.iloc[0]['identity']
        person_name = os.path.basename(os.path.dirname(closest_match_path))
        roll_no = st.session_state.roll_map.get(person_name, "N/A - Check Map")

        # Log and Display
        if log_attendance(person_name, roll_no):
            st.balloons()
            st.success(f"🎉 ATTENDANCE MARKED! Welcome, **{person_name}** (Roll: **{roll_no}**)!")
        
    except ValueError as e:
        st.error(f"❌ Face detection failed. Please ensure your face is clearly visible and centered. Details: {e}")
    except Exception as e:
        st.error(f"An unexpected system error occurred during attendance check: {e}")
        st.exception(e)
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)


# Streamlit UI Layout

st.title(f"👨‍🏫 DeepFace Attendance System (Detector: {DETECTOR.upper()})")
st.markdown("This system uses face recognition for **Webcam-Only** attendance.") 
ensure_directories_and_state()
tab_register, tab_attendance, tab_log = st.tabs(["1. Register Faces", "2. Mark Attendance (Webcam Only)", "3. View Log"])



with tab_register:
    st.header("👤 Face Registration")
    st.markdown("Upload a clear photo to register a new person and build the face database.")

    col1, col2 = st.columns(2)

    with col1:
        new_name = st.text_input("Full Name:", key="reg_name").strip()
        new_roll_no = st.text_input("Roll Number:", key="reg_roll").strip()

        uploaded_reg_file = st.file_uploader(
            "Upload Registration Photo (JPEG/PNG)", 
            type=['jpg', 'jpeg', 'png'],
            key="reg_file"
        )
        
        if st.button("Register Person", type="primary"):
            register_face(uploaded_reg_file, new_name, new_roll_no)

    with col2:
        st.subheader("Current Registered Database")
        
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
            st.info("No faces registered yet. Database is empty.")



with tab_attendance:
    st.header("📸 Check-in for Attendance")
    st.markdown("Use your webcam for check-in.")

    
    camera_input = st.camera_input("Webcam Capture", key="camera_check")

    if st.button("Mark Attendance", type="primary"):
        if camera_input:
            check_attendance(camera_input)
        else:
            st.error("Please capture a photo using the webcam first.")


with tab_log:
    st.header("📋 Attendance Log (CSV Output)")
    
    st.dataframe(st.session_state.log_df, use_container_width=True)
    
    csv = st.session_state.log_df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="Download Attendance Log as CSV",
        data=csv,
        file_name='attendance_log.csv',
        mime='text/csv',
        key="download_log"
    )
    
    st.markdown("*(This log file is permanently saved locally as `attendance_log.csv`)*")    

