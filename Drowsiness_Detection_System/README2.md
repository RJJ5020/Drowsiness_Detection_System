## Head Pose Estimation with Gaze Monitoring
This Python script uses the Mediapipe library along with OpenCV to perform head pose estimation and gaze monitoring in real-time. The application detects facial landmarks, calculates the 3D head pose, and alerts the user if their gaze deviates from a forward position for a specified duration.
### Install the libraries
 we will install the OpenCV and the Mediapipe library by using pip. On your terminal, please write this command:

-pip install opencv-python
-pip install mediapipe
### Download the Music File:

Download an audio file (e.g., "music.wav") to be played as an alarm when the user is not looking forward. Save it in the project directory.

### Functionality
The script captures video from the default camera (VideoCapture(0)).
It uses the face mesh model to detect facial landmarks and estimates the 3D head pose.Gaze direction is monitored based on the orientation of the head.If the user's gaze deviates from the forward position for a specified duration (5 seconds by default), an alarm is triggered.The alarm includes a visual alert on the video feed and plays the specified audio file.
###  Controls
Press q to exit the application.
### Dependencies
OpenCV
Mediapipe
NumPy
Pygame
