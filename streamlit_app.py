from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, WebRtcMode
import av
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
# ============= TAB 2: REAL-TIME CAMERA =============
class VideoProcessor(VideoProcessorBase):
    def __init__(self):
        self.last_alert_count = 0

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")

        results = model(img, conf=0.5, verbose=False)

        person_count = 0

        for r in results:
            for box in r.boxes:
                cls = int(box.cls[0])

                if cls == 0:
                    person_count += 1

                    x1, y1, x2, y2 = map(int, box.xyxy[0])

                    cv2.rectangle(
                        img,
                        (x1, y1),
                        (x2, y2),
                        (0,255,0),
                        2
                    )

        cv2.putText(
            img,
            f"People : {person_count}",
            (20,40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0,255,0),
            2
        )

        return av.VideoFrame.from_ndarray(img, format="bgr24")


with tab2:

    st.subheader("Real Time Camera")

    webrtc_streamer(
        key="people-counter",
        mode=WebRtcMode.SENDRECV,
        video_processor_factory=VideoProcessor,
        media_stream_constraints={
            "video": True,
            "audio": False
        },
        async_processing=True,
    )