import cv2
import numpy as np
import matplotlib.pyplot as plt

def resize_image(image, size=(48, 48)):
    """
    Resize an image to the given size.
    """
    resized_image = cv2.resize(image, size)
    return resized_image

def normalize_image(image):
    """
    Normalize pixel values in the image.
    """
    normalized_image = image / 255.0
    return normalized_image

def display_emotion_distribution(emotions, predictions):
    """
    Display a bar chart showing the distribution of predicted emotions.
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
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return gray_image
