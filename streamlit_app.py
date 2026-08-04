import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
import smtplib
from email.mime.text import MIMEText
import tempfile
import os

st.set_page_config(page_title="People Counter", layout="wide")

# ============= EMAIL SETTINGS =============
ADMIN_EMAIL = "manojtayal07@gmail.com"
SENDER_EMAIL = "priyanshutayal35@gmail.com"
SENDER_PASSWORD = "grwo yjxm dwwa rinn"

def send_alert_email(new_count):
    """Send alert email when people count increases"""
    subject = "People Gathering Alert!"
    body = f"Alert! People gathering increased.\nCurrent Count: {new_count}"
    
    msg = MIMEText(body)
    msg['Subject'] = subject
    msg['From'] = SENDER_EMAIL
    msg['To'] = ADMIN_EMAIL
    
    try:
        server = smtplib.SMTP("smtp.gmail.com", 587)
        server.starttls()
        server.login(SENDER_EMAIL, SENDER_PASSWORD)
        server.sendmail(SENDER_EMAIL, ADMIN_EMAIL, msg.as_string())
        server.quit()
        st.success("✉️ Alert email sent!")
    except Exception as e:
        st.error(f"Error sending email: {e}")

# ============= LOAD MODEL =============
@st.cache_resource
def load_model():
    """Load YOLOv8 model (cached for performance)"""
    return YOLO("yolov8n.pt")

model = load_model()

# ============= MAIN UI =============
st.title("👥 People Counting System")
st.markdown("Upload a video to detect and count people using YOLOv8")

# Create tabs
tab1, tab2 = st.tabs(["Video Upload", "Real-time Camera"])

# ============= TAB 1: VIDEO UPLOAD =============
with tab1:
    st.subheader("Upload Video File")
    
    uploaded_file = st.file_uploader("Choose a video file", type=["mp4", "avi", "mov", "mkv"])
    
    if uploaded_file is not None:
        # Create temp file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp_file:
            tmp_file.write(uploaded_file.read())
            temp_path = tmp_file.name
        
        st.info("Processing video... This may take a moment.")
        
        # Process video
        cap = cv2.VideoCapture(temp_path)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Settings")
            confidence_threshold = st.slider("Detection Confidence", 0.0, 1.0, 0.5)
            alert_threshold = st.number_input("Alert when count increases by:", min_value=1, value=5)
            show_boxes = st.checkbox("Show detection boxes", value=True)
        
        with col2:
            st.subheader("Statistics")
            placeholder_stats = st.empty()
        
        # Process frames
        frame_placeholder = st.empty()
        progress_bar = st.progress(0)
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        frame_count = 0
        person_counts = []
        last_alert_count = 0
        
        while True:
            success, frame = cap.read()
            if not success:
                break
            
            # Resize for faster processing
            frame_resized = cv2.resize(frame, (640, 480))
            
            # Run detection
            results = model(frame_resized, conf=confidence_threshold, verbose=False)
            
            person_count = 0
            
            # Draw boxes and count people
            if show_boxes:
                for r in results:
                    for box in r.boxes:
                        cls = int(box.cls[0])
                        if cls == 0:  # Person class
                            person_count += 1
                            if show_boxes:
                                x1, y1, x2, y2 = map(int, box.xyxy[0])
                                cv2.rectangle(frame_resized, (x1, y1), (x2, y2), (0, 255, 0), 2)
                                cv2.putText(frame_resized, "Person", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            else:
                for r in results:
                    for box in r.boxes:
                        cls = int(box.cls[0])
                        if cls == 0:
                            person_count += 1
            
            person_counts.append(person_count)
            
            # Check for alert
            if person_count - last_alert_count >= alert_threshold:
                send_alert_email(person_count)
                last_alert_count = person_count
            
            # Display
            frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
            
            # Add text to frame
            cv2.putText(frame_rgb, f"Count: {person_count}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2)
            
            frame_placeholder.image(frame_rgb, channels="RGB")
            
            # Update stats
            with placeholder_stats.container():
                st.metric("Current Count", person_count)
                st.metric("Average Count", f"{np.mean(person_counts):.1f}")
                st.metric("Max Count", max(person_counts))
            
            # Progress
            progress_bar.progress(min(frame_count / total_frames, 1.0))
            frame_count += 1
        
        cap.release()
        os.unlink(temp_path)
        
        st.success("✅ Video processing complete!")
        
        # Show final statistics
        st.subheader("Final Statistics")
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Frames", total_frames)
        col2.metric("Average People Count", f"{np.mean(person_counts):.1f}")
        col3.metric("Max People Count", max(person_counts))
        col4.metric("Min People Count", min(person_counts))

# ============= TAB 2: REAL-TIME CAMERA =============
with tab2:
    st.subheader("Real-time Camera Feed")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Settings")
        confidence_threshold_cam = st.slider("Detection Confidence", 0.0, 1.0, 0.5, key="cam_conf")
        alert_threshold_cam = st.number_input("Alert when count increases by:", min_value=1, value=5, key="cam_alert")
        show_boxes_cam = st.checkbox("Show detection boxes", value=True, key="cam_boxes")
    
    with col2:
        st.subheader("Live Stats")
        placeholder_count = st.empty()
    
    frame_placeholder_cam = st.empty()
    
    start_button = st.button("Start Camera")
    stop_button = st.button("Stop Camera")
    
    if start_button:
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        last_alert_count_cam = 0
        
        while not stop_button:
            success, frame = cap.read()
            if not success:
                st.error("Cannot access camera")
                break
            
            # Run detection
            results = model(frame, conf=confidence_threshold_cam, verbose=False)
            
            person_count_cam = 0
            
            # Draw boxes and count
            for r in results:
                for box in r.boxes:
                    cls = int(box.cls[0])
                    if cls == 0:
                        person_count_cam += 1
                        if show_boxes_cam:
                            x1, y1, x2, y2 = map(int, box.xyxy[0])
                            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Alert check
            if person_count_cam - last_alert_count_cam >= alert_threshold_cam:
                send_alert_email(person_count_cam)
                last_alert_count_cam = person_count_cam
            
            # Add text
            cv2.putText(frame, f"Count: {person_count_cam}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2)
            
            # Display
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_placeholder_cam.image(frame_rgb, channels="RGB")
            placeholder_count.metric("People Count", person_count_cam)
        
        cap.release()

st.markdown("---")
st.markdown("Made with ❤️ using Streamlit & YOLOv8")
