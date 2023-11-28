import cv2
import numpy as np
import tensorflow as tf

# Load the pre-trained emotion detection model
model = tf.keras.models.load_model('models/emotion_detection_model.h5')

def preprocess_image(image, target_size=(48, 48)):
    """Resizes and scales the image for model prediction."""
    if len(image.shape) == 2 or image.shape[2] == 1:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    image = cv2.resize(image, target_size)
    image = image.astype('float32') / 255  # Normalize
    return np.expand_dims(image, axis=0)

def predict_emotion(image):
    """Predicts the emotion from the given image."""
    processed_image = preprocess_image(image)
    emotions = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']
    prediction = model.predict(processed_image)
    return emotions[np.argmax(prediction)]

def process_video_stream():
    # Load Haar Cascade for face detection
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    # Initialize Webcam
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Convert to grayscale
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect faces
        faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        # Process each face and display emotion
        for (x, y, w, h) in faces:
            face_img = gray_frame[y:y+h, x:x+w]
            emotion = predict_emotion(face_img)
            cv2.putText(frame, emotion, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

        cv2.imshow('Real-Time Emotion Detection', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    process_video_stream()
