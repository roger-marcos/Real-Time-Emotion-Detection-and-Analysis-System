import cv2
import numpy as np
import tensorflow as tf

# Load the pre-trained emotion detection model
model = tf.keras.models.load_model('path_to_your_model.h5')

# Define the list of emotions that the model can detect
emotions = ['Happy', 'Sad', 'Angry', 'Surprise', 'Fear', 'Disgust']

def preprocess_image(image):
    """
    Preprocess the image for emotion detection model.
    Resizes and normalizes the image.
    """
    # Resize the image to match the input shape required by the model
    processed_image = cv2.resize(image, (48, 48))
    # Normalize pixel values
    processed_image = processed_image / 255.0
    return processed_image

def detect_emotion(image):
    """
    Detects the emotion from the image using the loaded model.
    Returns the detected emotion as a string.
    """
    # Preprocess the image for the model
    processed_image = preprocess_image(image)
    # Predict the emotion
    predictions = model.predict(np.array([processed_image]))
    # Find the index of the emotion with the highest probability
    emotion_index = np.argmax(predictions)
    return emotions[emotion_index]

def main():
    # Initialize the webcam
    cap = cv2.VideoCapture(0)

    while True:
        # Capture frame-by-frame from the webcam
        ret, frame = cap.read()
        if not ret:
            break

        # Convert the captured frame to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detect faces in the frame
        # Note: face_detector needs to be defined with a face detection model
        faces = face_detector.detectMultiScale(gray, 1.1, 4)

        for (x, y, w, h) in faces:
            # Extract the face from the frame
            face = gray[y:y+h, x:x+w]
            # Detect emotion in the face
            emotion = detect_emotion(face)
            # Display the detected emotion
            cv2.putText(frame, emotion, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
            # Draw a rectangle around the face
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 3)

        # Display the resulting frame with detected emotions
        cv2.imshow('Emotion Detector', frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the webcam and close all OpenCV windows
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
