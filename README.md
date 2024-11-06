
# Face Recognition Late Arrival Detection System

This project is a facial recognition system that detects and logs classmates who arrive late to class. The system captures images of individuals detected after a certain time, recognizes them using a trained model, and logs their details in a database. Additionally, a sound alert is played when a late arrival is detected.

## Table of Contents

- [Project Overview](#project-overview)
- [Features](#features)
- [Project Structure](#project-structure)
- [Setup Instructions](#setup-instructions)
- [Code Explanation](#code-explanation)
- [Database Logging](#database-logging)
- [Image Storage](#image-storage)
- [Usage](#usage)
- [Requirements](#requirements)

## Project Overview

The purpose of this system is to automate the detection and logging of late arrivals using a facial recognition model. When the system detects motion in the video feed after a specified time (e.g., 9:00 AM), it captures the individual's face, recognizes them using a pre-trained convolutional neural network (CNN) model, saves their image, and logs the occurrence in a database. 

## Features

- **Real-time Face Recognition**: Uses a CNN model to recognize classmates based on pre-labeled images.
- **Motion Detection**: Detects when someone enters the frame, triggering the face recognition process.
- **Sound Alert**: Plays a sound when a late arrival is detected.
- **Image Storage**: Saves images of late arrivals in a designated folder.
- **Database Logging**: Logs each late arrival with a timestamp, image filename, and recognized name in a SQLite database.

## Project Structure

Here's an overview of the project's folder structure:

```
project/
│
├── app.py                     # Main application code
├── face_recognition.py        # Face recognition code with model loading and prediction functions
├── face_recognition_model.h5  # Pre-trained CNN model for recognizing faces
├── classmates_faces/          # Folder containing known faces for each classmate
├── facereco/                  # Folder with Haar Cascade files for face detection
│   ├── haarcascade_frontalface_default.xml
│   └── haarcascade_profileface.xml
├── soundfiles/                # Folder with sound files for the alert
│   └── anime-wow-sound-effect.mp3
├── Late_Arrivals_Faces/       # Folder where late arrival images are saved
└── late_arrivals.db           # SQLite database file for logging late arrivals
```

## Setup Instructions

1. **Clone the Repository**: Clone the project repository to your local machine.
2. **Install Dependencies**: Make sure you have the required libraries installed by running:
   ```bash
   pip install -r requirements.txt
   ```
3. **Set Up Folders**: Ensure the following folders exist:
   - `Late_Arrivals_Faces/`: Folder where images of late arrivals will be saved.
   - `classmates_faces/`: Folder containing one image per classmate for face recognition.
4. **Configure Paths**: Update paths in `app.py` to reflect your system's file structure for Haar Cascade files, model file, and sound file.
5. **Run the Application**: Start the application by running:
   ```bash
   python app.py
   ```

## Code Explanation

### Motion Detection

The program uses a webcam feed to detect motion. It compares each frame with a static background to identify motion, using frame differences. Once motion is detected, the system proceeds with face recognition if the time is after 9:00 AM.

### Face Recognition

Using OpenCV's Haar Cascade, the system detects faces in the frame and extracts them. These faces are passed to a pre-trained CNN model (`face_recognition_model.h5`) to identify the classmate based on the `class_names` array, which includes the names of classmates.

The function `recognize_face` performs the following steps:
- Converts the frame to grayscale.
- Detects faces using Haar Cascade.
- Resizes and normalizes each face image.
- Uses the CNN model to predict the classmate's identity.
- Draws a rectangle around the recognized face and labels it with the classmate's name.

### Sound Alert

A sound is played when a late arrival is detected, using the specified sound file (`anime-wow-sound-effect.mp3`). This sound is played in a separate thread to avoid blocking the main process.

### Database Logging

The program logs each late arrival event in a SQLite database (`late_arrivals.db`). The database stores the following information for each event:
- `timestamp`: The exact date and time the late arrival was detected.
- `image_filename`: The filename of the saved image for the late arrival.
- `classmate_name`: The name of the recognized classmate.

### Image Storage

Each recognized face is saved as an image file in the `Late_Arrivals_Faces/` folder. The filenames are formatted with the classmate's name and timestamp, making it easy to organize and review the images later.

## Usage

- **Running the App**: Run the application by executing `python app.py`.
- **Capturing Late Arrivals**: The system will start detecting motion and recognizing faces. If it detects someone after the set time (e.g., 9:00 AM), it will capture and log their image and details.
- **Exiting the Program**: To stop the program, press `q` in the OpenCV display window.

## Requirements

Make sure the following dependencies are installed:
- `opencv-python`
- `tensorflow`
- `numpy`
- `sqlite3`
- `datetime`

You can install these dependencies with the following command:
```bash
pip install opencv-python tensorflow numpy
```

---
![GIF](https://i.giphy.com/media/v1.Y2lkPTc5MGI3NjExb3RjY3FwcXBodGtqbzdxbGEzNmR2MmVlZmc0eGJyZWhlYnVzaGV4OCZlcD12MV9pbnRlcm5hbF9naWZfYnlfaWQmY3Q9Zw/bYpYW17E9MA7iKod9O/giphy.gif)