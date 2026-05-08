# Ambient Assisted Living System

This project is a thesis based real-time Human Activity Recognition (HAR) system designed to support Ambient Assisted Living (AAL) environments through computer vision and deep learning techniques. The system detects and classifies human activities from live webcam input using MediaPipe Pose for skeletal landmark extraction and a TensorFlow/Keras sequence model for temporal activity recognition. The application processes video streams in real-time by extracting human body pose landmarks, generating custom skeletal visualizations, preprocessing pose regions, and feeding sequential frame data into a trained deep learning model. The system is capable of recognizing activities such as Standing, Walking, Sitting, and Falling, while maintaining activity logs and displaying predictions through an interactive desktop GUI built with Tkinter.

This project is intended to address the problems described in the study by Takase (2023), which analyzed 4,176 fall incident reports in healthcare facilities in Japan during the 2016–2020 period. The study showed that approximately 60.3% of fall incidents occurred inside patient rooms, while around 81.09% of victims were aged between 60 and 89 years old. About 93% of the fall incidents resulted in injuries, with 68.1% involving fractures, and 1.7% of the total incidents resulting in death. Most concerningly, 79.0% of fall incidents were unwitnessed by nurses. To improve robustness and interpretability, the system applies pose-based preprocessing rather than relying directly on RGB images, making it more resilient to lighting changes and background variations. The application also provides separate User Mode and Developer Mode, where the developer interface includes real-time skeletal visualization for debugging and analysis purposes.

Reference:
Takase, M. (2023). Falls as the result of interplay between nurses, patient and the environment: Using text-mining to uncover how and why falls happen. International Journal of Nursing Sciences, 10(1), 30–37.
https://doi.org/10.1016/j.ijnss.2022.12.003

Alternative link:
https://pmc.ncbi.nlm.nih.gov/articles/PMC9969063/

--------------------------------------------------

## DEMO

https://github.com/user-attachments/assets/45ff3fab-328a-4d34-9635-db9ead501659


Full Video: https://drive.google.com/file/d/1QflGH1Ad8S_-VXyuuzZUckOjYsAyjDOo/view?usp=drive_link

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

## How It Works

1. **Video Input**:
   The system captures live video frames from a webcam using OpenCV.

2. **Pose Estimation**:
   Each frame is processed using MediaPipe Pose to detect 33 human body landmarks.

3. **Skeleton Visualization**:
   - Landmarks are grouped into body regions:
     - Head
     - Body
     - Left Hand
     - Right Hand
     - Left Leg
     - Right Leg
   - Custom markers and colors are drawn for each body part.

4. **Pose-Based Preprocessing**:
   - The body region is automatically cropped using landmark coordinates.
   - Cropped pose image is resized to 128x128.
   - Pixel values are normalized into range 0-1.

5. **Temporal Sequence Buffer**:
   - Processed frames are stored in a sequence buffer.
   - The model uses 6 consecutive frames for activity prediction.

6. **Activity Classification**:
   - Sequential pose data is fed into the TensorFlow model.
   - The model predicts activity classes:
     - Standing
     - Walking
     - Sitting
     - Falling

7. **Confidence Estimation**:
   - Prediction confidence is calculated from softmax probabilities.
   - Results are displayed in real-time.

8. **Activity Logging**:
   - Activity changes are automatically logged.
   - Log includes:
     - Activity label
     - Start time
     - Duration

9. **Developer Visualization**:
   Developer Mode displays:
   - Original webcam feed
   - Skeleton-only visualization
   - Real-time logs

10. **Result Display**:
   - GUI updates continuously using Tkinter.
   - Falling events are highlighted in red.
   - Activity history can be sorted and exported.

--------------------------------------------------

## PROJECT STRUCTURE
```
project/
│
├── appz.py
├── augmentasi_zigzag_seq6_acc_98.75_.h5
├── requirements.txt
└── README.md
```
--------------------------------------------------

## MODEL INFORMATION

The activity recognition model uses a sequence-based deep learning approach trained on skeletal pose data.

**Input**:
- Sequence Length: 6 frames
- Image Size: 128x128

**Output Classes**:
- 0 = Standing
- 1 = Walking
- 2 = Sitting
- 3 = Falling

**Training Result**:

<img width="1189" height="390" alt="acc dan loss" src="https://github.com/user-attachments/assets/94253a02-d9de-45da-890f-d374bb00d240" />


--------------------------------------------------

## GUI MODES

**User Mode**:
- Live webcam feed
- Activity prediction
- Timestamp
- Activity history log

**Developer Mode**:
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
- The system is designed as a surveillance system using a camera installed at the center of the room ceiling, and the model is trained using the same type of data (An example of this setup can be seen in the demo video)

--------------------------------------------------

## TECHNOLOGIES USED

- Python 3.10
- OpenCV
- MediaPipe
- TensorFlow / Keras
- NumPy
- Tkinter
- Pillow

--------------------------------------------------

## Acknowledgments

* This project was developed as a final thesis by **Muhammad Ilham** at Universitas Airlangga, Department of Robotics and Artificial Intelligence Engineering.
* Special thanks to thesis supervisors and peers who contributed insight and support.

-----------------

## Contact

For questions or collaboration, please contact: \[[muhammadilham121102@gmail.com](mailto:muhammadilham121102@gmail.com)]

