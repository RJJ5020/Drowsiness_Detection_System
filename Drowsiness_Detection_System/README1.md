## Drowsiness Detection System 
This Python script uses computer vision techniques to detect eye blinks in real-time using a webcam feed. When the script identifies that the user's eyes are closed for a prolonged period (indicating drowsiness or fatigue), it triggers an alert by playing an alert sound.

### Dependencies:
scipy: Used for calculating the Euclidean distance between facial landmarks.
imutils: Provides convenient functions for resizing, rotating, and displaying images.
pygame: Enables audio playback for the alert sound.
dlib: A powerful library for facial landmark detection and shape prediction.
cv2 (OpenCV): Used for computer vision tasks, such as reading webcam feeds and image processing.
### Installation:
1.Install Required Packages:

You can install the required packages using the following:
pip install scipy imutils pygame dlib opencv-python
2.Add an Alert Sound:

Provide an audio file (e.g., "music.wav") for the alert sound and place it in the same directory as your script.

### Usage
1.Run the Script:
Execute the script by running the Python file. Ensure your webcam is connected and properly functioning.
python your_script_name.py
2.Terminate the Script:
Press 'q' to exit the script and close the webcam feed window.
### Script Explanation:
-The script uses the dlib library to detect facial landmarks, particularly focusing on the eyes.
-Euclidean distances between specific facial landmarks are utilized to calculate the Eye Aspect Ratio (EAR).
-The EAR is a crucial metric for determining eye openness, and when it falls below a certain threshold, it signifies a blink.
-A consecutive frame check is implemented to avoid false positives, ensuring that sustained eye closure triggers the alert.
-An alert message is displayed on the video feed, and an alert sound is played using the pygame mixer.
### Notes:
Make sure to customize the script according to your requirements, such as adjusting the EAR threshold or choosing a different alert sound.