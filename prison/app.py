# import cv2
# import numpy as np
# from ultralytics import YOLO

# # Load YOLO model
# model = YOLO('yolov8n.pt')  # Use 'yolov8s.pt' for better accuracy

# # Define line coordinates (virtual boundary)
# line_start = (100, 200)
# line_end = (500, 200)

# # Alarm function 
# def trigger_alarm():
#     print("ALARM! Prison Break Detected!")

# # Initialize video capture for mobile camera
# # Replace 'http://192.168.1.2:8080/video' with your mobile's IP Webcam URL
# cap = cv2.VideoCapture('http://192.168.85.242:8080/video')

# while True:
#     ret, frame = cap.read()
#     if not ret:
#         print("Failed to grab frame. Check your connection.")
#         break

#     # Detect objects in the frame using YOLO
#     results = model(frame)

#     # Get the detections from results
#     detections = results[0].boxes  # Access the boxes from the results

#     # Iterate over detections
#     for box in detections:
#         x_min, y_min, x_max, y_max = box.xyxy[0]  # Get box coordinates
#         conf = box.conf[0]  # Get the confidence score
#         cls = box.cls[0]    # Get the class label

#         if cls == 0:  # Class 0 corresponds to 'person'
#             bottom_center = ((int(x_min + x_max) // 2), int(y_max))
            
#             # Draw bounding box and bottom-center point
#             cv2.rectangle(frame, (int(x_min), int(y_min)), (int(x_max), int(y_max)), (0, 255, 0), 2)
#             cv2.circle(frame, bottom_center, 5, (0, 0, 255), -1)

#             # Check for line crossing
#             if bottom_center[1] > line_start[1]:  # Simple check for crossing the horizontal line
#                 trigger_alarm()

#     # Draw the line (boundary) on the frame
#     cv2.line(frame, line_start, line_end, (255, 0, 0), 2)

#     # Display the video with bounding boxes and line
#     cv2.imshow("Prison Break Detection", frame)

#     if cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to exit
#         break

# cap.release()
# cv2.destroyAllWindows()

# # python prison_break_detection.py 

# import cv2
# import numpy as np
# from ultralytics import YOLO

# # Load YOLO model
# model = YOLO('yolov8n.pt')  # Use 'yolov8s.pt' for better accuracy if needed

# # Define line coordinates (virtual boundary)
# line_start = (100, 200)
# line_end = (500, 200)

# # Alarm function
# def trigger_alarm():
#     print("ALARM! Prison Break Detected!")

# # Initialize video capture for input video file
# # Use raw string (r"") or properly escaped paths
# video_path = r'C:\Users\sayeeswarnan\Downloads\prison_break_detection\prison_break_detection\prison_break_env\Assests\video2.mp4'
# cap = cv2.VideoCapture(video_path)

# if not cap.isOpened():
#     print(f"Error: Could not open video file: {video_path}")
#     exit()

# while True:
#     ret, frame = cap.read()
#     if not ret:
#         print("End of video or failed to grab frame.")
#         break

#     # Detect objects in the frame using YOLO
#     results = model(frame)

#     # Get the detections from results
#     if len(results) > 0 and results[0].boxes:
#         detections = results[0].boxes.data.cpu().numpy()  # Convert boxes to numpy format

#         # Iterate over detections
#         for box in detections:
#             x_min, y_min, x_max, y_max, conf, cls = box

#             if int(cls) == 0:  # Class 0 corresponds to 'person'
#                 bottom_center = (int((x_min + x_max) // 2), int(y_max))
                
#                 # Draw bounding box and bottom-center point
#                 cv2.rectangle(frame, (int(x_min), int(y_min)), (int(x_max), int(y_max)), (0, 255, 0), 2)
#                 cv2.circle(frame, bottom_center, 5, (0, 0, 255), -1)

#                 # Check for line crossing
#                 if bottom_center[1] > line_start[1]:  # Simple check for crossing the horizontal line
#                     trigger_alarm()

#     # Draw the line (boundary) on the frame
#     cv2.line(frame, line_start, line_end, (255, 0, 0), 2)

#     # Display the video with bounding boxes and line
#     cv2.imshow("Prison Break Detection", frame)

#     if cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to exit
#         break

# cap.release()
# cv2.destroyAllWindows()
import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
from PIL import Image
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

# Load YOLO model
@st.cache_resource
def load_model():
    try:
        return YOLO('yolov8n.pt')
    except Exception as e:
        st.error(f"Error loading YOLO model: {e}")
        return None

model = load_model()

# Virtual boundary
LINE_START = (100, 200)
LINE_END = (500, 200)

# Email configuration
SENDER_EMAIL = "notifysystemclg@gmail.com"
RECEIVER_EMAIL = "ssayeeswarnan@gmail.com"
EMAIL_PASSWORD = "wzgu ktek roma hkgl"  # Use app-specific password if 2FA is enabled

def send_email():
    """Send an alert email when a prison break is detected."""
    subject = "Prison Break Detected!"
    body = "Alert: A prison break has been detected by the system."

    msg = MIMEMultipart()
    msg['From'] = SENDER_EMAIL
    msg['To'] = RECEIVER_EMAIL
    msg['Subject'] = subject
    msg.attach(MIMEText(body, 'plain'))

    try:
        server = smtplib.SMTP('smtp.gmail.com', 587)
        server.starttls()
        server.login(SENDER_EMAIL, EMAIL_PASSWORD)
        server.sendmail(SENDER_EMAIL, RECEIVER_EMAIL, msg.as_string())
        server.quit()
        st.success("Email sent successfully!")
    except Exception as e:
        st.error(f"Failed to send email: {e}")

# Streamlit UI
st.title("YOLO Video Detection")

# Video file uploader
video_file = st.file_uploader("Upload a video file", type=["mp4", "avi", "mov"])

if video_file:
    # Save uploaded video temporarily
    with open("uploaded_video.mp4", "wb") as f:
        f.write(video_file.read())

    cap = cv2.VideoCapture("uploaded_video.mp4")

    stframe = st.empty()
    email_sent = False

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Run YOLO model on the frame
        try:
            results = model(frame)
        except Exception as e:
            st.error(f"Error during model inference: {e}")
            break

        alarm_triggered = False

        if results[0].boxes:
            detections = results[0].boxes.data.cpu().numpy()
            for box in detections:
                x_min, y_min, x_max, y_max, conf, cls = box
                if int(cls) == 0:  # Class 0 is 'person'
                    bottom_center = (int((x_min + x_max) // 2), int(y_max))
                    cv2.rectangle(frame, (int(x_min), int(y_min)), (int(x_max), int(y_max)), (0, 255, 0), 2)
                    cv2.circle(frame, bottom_center, 5, (0, 0, 255), -1)
                    if bottom_center[1] > LINE_START[1]:
                        alarm_triggered = True

        # Draw boundary line
        cv2.line(frame, LINE_START, LINE_END, (255, 0, 0), 2)

        if alarm_triggered and not email_sent:
            send_email()
            email_sent = True

        # Display the frame
        stframe.image(frame, channels="BGR")

    cap.release()
else:
    st.info("Please upload a video file to start detection.")