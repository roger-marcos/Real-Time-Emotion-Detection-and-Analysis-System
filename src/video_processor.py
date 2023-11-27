import cv2
import numpy as np

# Import the emotion detection function from emotion_detector.py
from emotion_detector import detect_emotion

def preprocess_frame(frame):
    """
    Apply necessary preprocessing to the video frame
    before emotion detection. This might include resizing,
    color adjustments, etc.
    """
    # Convert frame to grayscale
    processed_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return processed_frame

def face_detection(frame):
    """
    Detect faces in the given frame using Haar Cascade Classifier.
    Returns the coordinates of detected faces.
    """
    # Load the Haar Cascade Classifier for face detection
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    # Detect faces in the frame
    faces = face_cascade.detectMultiScale(frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    return faces

def process_video_stream():
    """
    Capture video from the webcam and process each frame
    for emotion detection.
    """
    # Initialize webcam
    cap = cv2.VideoCapture(0)

    while True:
        # Read frame from webcam
        ret, frame = cap.read()
        if not ret:
            break

        # Preprocess the frame
        processed_frame = preprocess_frame(frame)

        # Detect faces in the frame
        faces = face_detection(processed_frame)

        for (x, y, w, h) in faces:
            # Extract the face from the frame
            face = processed_frame[y:y+h, x:x+w]

            # Detect emotion for the extracted face
            emotion = detect_emotion(face)

            # Display the detected emotion
            cv2.putText(frame, emotion, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            # Draw a rectangle around the face
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

        # Display the resulting frame
        cv2.imshow('Video Stream', frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the webcam and close all windows
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    process_video_stream()
