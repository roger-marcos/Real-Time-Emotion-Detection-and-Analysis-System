import unittest
import numpy as np
import tensorflow as tf
from emotion_detector import preprocess_image, detect_emotion, model

class TestEmotionDetector(unittest.TestCase):

    def setUp(self):
        # Setup for the tests; load the model if necessary, etc.
        # Assuming the model is loaded in the emotion_detector.py script
        self.model = model

    def test_model_loaded(self):
        # Test to verify that the model is loaded correctly
        self.assertIsInstance(self.model, tf.keras.Model)

    def test_preprocess_image(self):
        # Test to verify image preprocessing
        test_image = np.zeros((100, 100, 3), dtype=np.uint8)  # Black test image
        processed_image = preprocess_image(test_image)
        self.assertEqual(processed_image.shape, (48, 48))  # Assuming model expects 48x48 input

    def test_detect_emotion(self):
        # Test to verify emotion detection
        test_image = np.zeros((48, 48), dtype=np.uint8)  # Black test image
        emotion = detect_emotion(test_image)
        self.assertIn(emotion, ['Happy', 'Sad', 'Angry', 'Surprise', 'Fear', 'Disgust', 'Neutral'])  # List of expected emotions

    def tearDown(self):
        # Teardown code, if necessary
        pass

if __name__ == '__main__':
    unittest.main()
