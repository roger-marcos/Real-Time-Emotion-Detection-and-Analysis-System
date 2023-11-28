# Real-Time Emotion Detection and Analysis System

## Introduction
The "Real-Time Emotion Detection and Analysis System" is a cutting-edge solution designed to identify and analyze human emotions in real time. Leveraging advanced machine learning and computer vision techniques, this system interprets facial expressions captured through a webcam and categorizes them into distinct emotion classes. It has potential applications in various fields, including human-computer interaction, psychological research, and user experience studies.

## Project Overview
My system uses a pre-trained convolutional neural network (CNN) to classify facial expressions into seven primary emotions: Angry, Disgust, Fear, Happy, Neutral, Sad, and Surprise. The project includes scripts for real-time video processing, a Jupyter Notebook for model training and validation, and a demonstration notebook showcasing the system in action.

### Key Components
- **Emotion Detector**: A core component utilizing a CNN model to detect emotions from facial expressions.
- **Video Processor**: A script that processes live video streams, integrating the emotion detection model to analyze and display detected emotions in real time.
- **Utility Functions**: A set of helper functions for image processing and data visualization, enhancing the efficiency and scalability of the system.
- **Model Training Notebook**: An interactive notebook that guides through training, validating, and saving the emotion detection model.
- **Demo Notebook**: A demonstration of the system's capabilities in a live setting, utilizing a webcam feed for real-time emotion detection.

## Installation and Usage

### Prerequisites
- Python 3.6+
- TensorFlow 2.x
- OpenCV
- Jupyter Notebook (for running notebooks)

### Installation
Clone the repository and install the required dependencies:
```bash
git clone https://github.com/roger-marcos/Real-Time-Emotion-Detection-and-Analysis-System.git
cd Real-Time-Emotion-Detection-and-Analysis-System
pip install -r requirements.txt bash
```
### Running the Demo
Launch the Demo.ipynb notebook in Jupyter to see the real-time emotion detection system in action. Ensure you have a functioning webcam for the live feed.

### Training the Model
Run the Model_Training.ipynb notebook for detailed steps on training and validating the emotion detection model. The notebook provides insights into the model architecture, training process, and performance metrics.

### Real-Time Video Processing
Execute src/video_processor.py to process live video streams for emotion detection. This script utilizes the trained model and webcam feed to display detected emotions dynamically.

### Testing
Unit tests for the emotion detector and video processor are available in the tests/ directory. These tests ensure the reliability and accuracy of key components.

### Contributing
Contributions are welcome! Please read through the CONTRIBUTING.md file for guidelines on how to contribute effectively to this project.

### License
This project is licensed under the MIT License. For more information, see the LICENSE.md file.

### Credits and Acknowledgments
· Dataset: [[Fer-2013](https://www.kaggle.com/datasets/msambare/fer2013)]<br>
· OpenCV Library: Used for image processing and video handling<br>
· TensorFlow: For building and training the CNN model

