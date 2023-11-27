import unittest
import cv2
import numpy as np
from video_processor import preprocess_frame, face_detection, predict_emotion

class TestVideoProcessor(unittest.TestCase):

    def setUp(self):
        # Setup code, if necessary
        pass

    def test_preprocess_frame(self):
        # Test the frame preprocessing function
        test_frame = np.zeros((100, 100, 3), dtype=np.uint8)  # Black square
        processed_frame = preprocess_frame(test_frame)
        self.assertEqual(processed_frame.shape, (48, 48))  # Assuming your model expects 48x48 input

    def test_face_detection(self):
        # Test the face detection function
        test_frame = cv2.imread('path_to_test_image_with_face.jpg')  # Replace with an actual image path
        faces = face_detection(test_frame)
        self.assertGreater(len(faces), 0)  # Assuming there is at least one face in the image

    def test_predict_emotion(self):
        # Test the emotion prediction function
        test_frame = cv2.imread('path_to_test_image_with_face.jpg')  # Replace with an actual image path
        preprocessed_frame = preprocess_frame(test_frame)
        emotion = predict_emotion(preprocessed_frame)
        self.assertIn(emotion, ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral'])  # List of expected emotions

    def tearDown(self):
        # Tear down code, if necessary
        pass

if __name__ == '__main__':
    unittest.main()
