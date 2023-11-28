import cv2
import numpy as np
import matplotlib.pyplot as plt

def resize_image(image, size=(48, 48)):
    """
    Resize an image to the given size.
    """
    return cv2.resize(image, size)

def normalize_image(image):
    """
    Normalize pixel values in the image to be between 0 and 1.
    """
    return image / 255.0

def plot_emotion_distribution(predictions, emotions):
    """
    Plot a bar chart showing the distribution of predicted emotions.
    """
    plt.bar(emotions, predictions)
    plt.xlabel('Emotions')
    plt.ylabel('Probability')
    plt.title('Emotion Distribution')
    plt.show()

def convert_to_grayscale(image):
    """
    Convert an image to grayscale.
    """
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

def apply_clahe(image):
    """
    Apply CLAHE (Contrast Limited Adaptive Histogram Equalization) 
    to improve image contrast.
    """
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    return clahe.apply(image)

def display_image(image, title="Image"):
    """
    Display a single image.
    """
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.title(title)
    plt.axis('off')
    plt.show()
