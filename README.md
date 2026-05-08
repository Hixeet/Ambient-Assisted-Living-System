# Ambient Assisted Living System

This project is a thesis-based real-time Human Activity Recognition (HAR) system designed to support Ambient Assisted Living (AAL) environments through computer vision and deep learning techniques. The system detects and classifies human activities from live webcam input using MediaPipe Pose for skeletal landmark extraction and a TensorFlow/Keras sequence model for temporal activity recognition.

The application processes video streams in real-time by extracting human body pose landmarks, generating custom skeletal visualizations, preprocessing pose regions, and feeding sequential frame data into a trained deep learning model. The system is capable of recognizing activities such as Standing, Walking, Sitting, and Falling, while maintaining activity logs and displaying predictions through an interactive desktop GUI built with Tkinter.

To improve robustness and interpretability, the system applies pose-based preprocessing rather than relying directly on RGB images, making it more resilient to lighting changes and background variations. The application also provides separate User Mode and Developer Mode, where the developer interface includes real-time skeletal visualization for debugging and analysis purposes.

--------------------------------------------------

## DEMO

--------------------------------------------------

## FEATURES

- Real-time human activity recognition using webcam input
- Pose estimation using MediaPipe Pose
- Deep learning-based activity classification using TensorFlow/Keras
- Sequence-based temporal prediction using multiple consecutive frames
- Detection of:
  - Standing
  - Walking
  - Sitting
  - Falling
- Real-time confidence score display
- Custom skeleton visualization with colored body-part grouping
- Automatic pose region cropping and normalization
- User Mode and Developer Mode interface
- Activity logging with timestamp and duration tracking
- Automatic detection when no person is present
- Multi-camera support
- Dynamic .h5 model selection
- Real-time activity history table with sorting support
- Save activity logs to text file
- Modern desktop GUI using Tkinter

--------------------------------------------------

## INSTALLATION

1. Clone the repository

  ```bash
  git clone https://github.com/Hixeet/ambient-assisted-living-system.git
  cd ambient-assisted-living-system
  ```
2. Install dependencies

  ```bash
  pip install -r requirements.txt
  ```

Example dependencies:

```text
opencv-python
numpy
tensorflow
mediapipe
pillow
tk
```
--------------------------------------------------

## HOW IT WORKS

1. Video Input

The system captures live video frames from a webcam using OpenCV.

2. Pose Estimation

Each frame is processed using MediaPipe Pose to detect 33 human body landmarks.

3. Skeleton Visualization

- Landmarks are grouped into body regions:
  - Head
  - Body
  - Left Hand
  - Right Hand
  - Left Leg
  - Right Leg

- Custom markers and colors are drawn for each body part.

4. Pose-Based Preprocessing

- The body region is automatically cropped using landmark coordinates
- Cropped pose image is resized to 128x128
- Pixel values are normalized into range 0-1

5. Temporal Sequence Buffer

- Processed frames are stored in a sequence buffer
- The model uses 6 consecutive frames for activity prediction

6. Activity Classification

- Sequential pose data is fed into the TensorFlow model
- The model predicts activity classes:
  - Standing
  - Walking
  - Sitting
  - Falling

7. Confidence Estimation

- Prediction confidence is calculated from softmax probabilities
- Results are displayed in real-time

8. Activity Logging

- Activity changes are automatically logged
- Log includes:
  - Activity label
  - Start time
  - Duration

9. Developer Visualization

Developer Mode displays:
- Original webcam feed
- Skeleton-only visualization
- Real-time logs

10. Result Display

- GUI updates continuously using Tkinter
- Falling events are highlighted in red
- Activity history can be sorted and exported

--------------------------------------------------

## PROJECT STRUCTURE

project/
│
├── appz.py
├── augmentasi_zigzag_seq6_acc_98.75_.h5
├── requirements.txt
└── README.md

--------------------------------------------------

## MODEL INFORMATION

The activity recognition model uses a sequence-based deep learning approach trained on skeletal pose data.

Input:
- Sequence Length: 6 frames
- Image Size: 128x128

Output Classes:
- 0 = Standing
- 1 = Walking
- 2 = Sitting
- 3 = Falling

--------------------------------------------------

## GUI MODES

User Mode:
- Live webcam feed
- Activity prediction
- Timestamp
- Activity history log

Developer Mode:
- Live webcam feed
- Real-time skeleton visualization
- Activity prediction
- Timestamp
- Activity history log

--------------------------------------------------

## NOTES

- The system uses pose-based activity recognition instead of raw RGB classification
- Performance depends on pose visibility and camera positioning
- Adequate lighting improves landmark detection accuracy
- Falling detection is highlighted as an abnormal event
- The sequence model requires several consecutive frames before prediction starts
- GPU acceleration can improve TensorFlow inference speed
- The application currently runs prediction in the main GUI thread and may benefit from threading optimization

--------------------------------------------------

## FUTURE IMPROVEMENTS

- Add alarm/notification system for fall detection
- Integrate database logging
- Add support for video file input
- Optimize inference speed using threading or multiprocessing
- Add FPS monitoring
- Improve activity recognition with LSTM/Transformer architectures
- Add elderly behavior analytics for smart healthcare applications

--------------------------------------------------

## TECHNOLOGIES USED

- Python
- OpenCV
- MediaPipe
- TensorFlow / Keras
- NumPy
- Tkinter
- Pillow

--------------------------------------------------

## LICENSE

This project is intended for research and educational purposes.
