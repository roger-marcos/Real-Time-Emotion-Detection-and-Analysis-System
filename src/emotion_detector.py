import cv2
import numpy as np
import tensorflow as tf

# Load the pre-trained emotion detection model
model = tf.keras.models.load_model('models/emotion_detection_model.h5')

def preprocess_image(image, target_size=(48, 48)):
    """
    Resizes and scales the image for model prediction.
    Assumes images are grayscale.
    """
    if len(image.shape) == 2 or image.shape[2] == 1:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    image = cv2.resize(image, target_size)
    image = image.astype('float32') / 255  # Normalize
    return np.expand_dims(image, axis=0)

def detect_emotion(image):
    """
    Detects the emotion from the given image.
    """
    processed_image = preprocess_image(image)
    emotions = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']
    prediction = model.predict(processed_image)
    return emotions[np.argmax(prediction)]
