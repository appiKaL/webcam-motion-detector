import cv2
import os
import subprocess
import threading
import time
import sqlite3
import numpy as np
from datetime import datetime
import tensorflow as tf
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email.mime.text import MIMEText
from email import encoders

# Set environment variable for compatibility
os.environ["QT_QPA_PLATFORM"] = "xcb"

# Paths to Haar Cascade files and sound
frontal_face_cascade_path = "/path/to/haarcascade_frontalface_default.xml"
profile_face_cascade_path = "/path/to/haarcascade_profileface.xml"
sound_file = "/path/to/soundfile.mp3"

# Email configuration
SMTP_SERVER = 'smtp.example.com'  # Replace with your SMTP server
SMTP_PORT = 587  # Replace with your SMTP port
SENDER_EMAIL = 'your_email@example.com'
SENDER_PASSWORD = 'your_password'
RECEIVER_EMAIL = 'receiver@example.com'

# Load Haar Cascades
frontal_face_cascade = cv2.CascadeClassifier(frontal_face_cascade_path)
profile_face_cascade = cv2.CascadeClassifier(profile_face_cascade_path)

# Load the trained model
model = tf.keras.models.load_model("/path/to/face_recognition_model.h5")

# Class names corresponding to your classmates
class_names = ["Adrien", "Aliser", "Alper", "AntoineNotCoach", "AntoineNotNotCoach", "Ben", "Christian", "Colin", 
               "Damien", "Ezgi", "Geoffroy", "Georgina", "Hui", "Kyllian", "Laura", "Loic", "Mathieu", "Mehmet", 
               "Minh", "Mustafa", "Ness", "Ridvan", "Volodymyr"]

# Folder to save images
save_folder = "Late_Arrivals_Faces"
if not os.path.exists(save_folder):
    os.makedirs(save_folder)

# Initialize database connection
conn = sqlite3.connect('late_arrivals.db')
cursor = conn.cursor()
cursor.execute('''
    CREATE TABLE IF NOT EXISTS late_arrivals (
        id INTEGER PRIMARY KEY,
        timestamp TEXT,
        image_filename TEXT,
        classmate_name TEXT
    )
''')
conn.commit()

# Email function
def send_email_with_attachment(filename, recipient_email):
    subject = "Late Arrival Notification"
    body = f"{os.path.basename(filename)} was detected as arriving late."

    # Set up MIME message
    msg = MIMEMultipart()
    msg['From'] = SENDER_EMAIL
    msg['To'] = recipient_email
    msg['Subject'] = subject
    msg.attach(MIMEText(body, 'plain'))

    # Attach image
    attachment = MIMEBase('application', 'octet-stream')
    with open(filename, 'rb') as file:
        attachment.set_payload(file.read())
    encoders.encode_base64(attachment)
    attachment.add_header('Content-Disposition', f'attachment; filename={os.path.basename(filename)}')
    msg.attach(attachment)

    # Send email
    try:
        with smtplib.SMTP(SMTP_SERVER, SMTP_PORT) as server:
            server.starttls()
            server.login(SENDER_EMAIL, SENDER_PASSWORD)
            server.sendmail(SENDER_EMAIL, recipient_email, msg.as_string())
        print(f"Email sent to {recipient_email} with attachment {filename}")
    except Exception as e:
        print(f"Error sending email: {e}")

# Function to save the detection to the database
def log_late_arrival(filename, classmate_name):
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    cursor.execute("INSERT INTO late_arrivals (timestamp, image_filename, classmate_name) VALUES (?, ?, ?)", 
                   (timestamp, filename, classmate_name))
    conn.commit()

# Function to play sound in a separate thread with error handling
def play_sound():
    try:
        subprocess.run(["mpg123", sound_file], check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error playing sound: {e}")
    except FileNotFoundError:
        print("Sound file not found. Please check the file path.")

def recognize_face(frame):
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = frontal_face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5)

    recognized_names = []  # List to keep track of recognized names
    for (x, y, w, h) in faces:
        face_img = frame[y:y+h, x:x+w]
        face_img = cv2.resize(face_img, (128, 128))
        face_img = np.expand_dims(face_img, axis=0) / 255.0  # Normalize

        prediction = model.predict(face_img)
        class_index = np.argmax(prediction)
        class_name = class_names[class_index]
        recognized_names.append(class_name)  # Add the recognized name to the list

        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(frame, class_name, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    return frame, recognized_names  # Return the frame and recognized names

# Initialize video capture
video = cv2.VideoCapture(0)
static_back = None
motion_list = [None, None]
sound_played = False  # Flag for sound playback
last_capture_time = time.time()  # Timer for capture frequency

while True:
    # Capture frame-by-frame
    check, frame = video.read()
    if not check:
        break

    # Motion detection setup
    motion = 0
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (21, 21), 0)
    if static_back is None:
        static_back = gray
        continue

    # Frame difference for motion detection
    diff_frame = cv2.absdiff(static_back, gray)
    thresh_frame = cv2.threshold(diff_frame, 30, 255, cv2.THRESH_BINARY)[1]
    thresh_frame = cv2.dilate(thresh_frame, None, iterations=2)

    # Detect contours to identify motion
    contours, _ = cv2.findContours(thresh_frame.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        if cv2.contourArea(contour) < 10000:
            continue
        motion = 1  # Motion detected
        (x, y, w, h) = cv2.boundingRect(contour)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 3)

    # Update motion tracking
    motion_list.append(motion)
    motion_list = motion_list[-2:]

    # Check for faces when motion is detected and it's after 9 am
    current_time = datetime.now()
    if motion == 1 and current_time.hour >= 9:
        frame, recognized_names = recognize_face(frame)  # Call the recognition function

        if recognized_names:
            if time.time() - last_capture_time >= 2:
                last_capture_time = time.time()
                if not sound_played:
                    threading.Thread(target=play_sound).start()
                    sound_played = True

                for name in recognized_names:
                    filename = os.path.join(save_folder, f"{name}_Late_Arrival_Face_{current_time.strftime('%Y-%m-%d_%H-%M-%S')}.jpg")
                    cv2.imwrite(filename, frame)
                    log_late_arrival(filename, name)  # Log to database with classmate's name
                    send_email_with_attachment(filename, RECEIVER_EMAIL)  # Send email

        else:
            sound_played = False

    else:
        sound_played = False

    cv2.imshow("Color Frame", frame)
    key = cv2.waitKey(1)
    if key == ord('q'):
        break

video.release()
cv2.destroyAllWindows()
conn.close()
