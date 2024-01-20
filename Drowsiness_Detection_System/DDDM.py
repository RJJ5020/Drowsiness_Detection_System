# Import necessary libraries
from scipy.spatial import distance
from imutils import face_utils
from pygame import mixer
import imutils
import dlib
import cv2

# Initialize the sound mixer for the alarm
mixer.init()
mixer.music.load("music.wav")  # Load the alarm sound

# Function to calculate the Eye Aspect Ratio (EAR) given the landmarks of an eye
def eye_aspect_ratio(eye):
    A = distance.euclidean(eye[1], eye[5])
    B = distance.euclidean(eye[2], eye[4])
    C = distance.euclidean(eye[0], eye[3])
    ear = (A + B) / (2.0 * C)
    return ear

# Set threshold and frame check for detecting drowsiness
thresh = 0.25
frame_check = 20

# Initialize face detector and facial landmarks predictor
detect = dlib.get_frontal_face_detector()
predict = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# Define the indices for the left and right eyes in the facial landmarks
(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_68_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_68_IDXS["right_eye"]

# Open the camera
cap = cv2.VideoCapture(0)

# Initialize flag for tracking consecutive frames with closed eyes
flag = 0

# Main loop for processing video frames
while True:
    # Read a frame from the camera
    ret, frame = cap.read()

    # Resize the frame for better processing speed
    frame = imutils.resize(frame, width=450)

    # Convert the frame to grayscale for facial landmark detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the grayscale frame
    subjects = detect(gray, 0)

    # Loop through each detected face
    for subject in subjects:
        # Predict facial landmarks for the detected face
        shape = predict(gray, subject)
        shape = face_utils.shape_to_np(shape)

        # Extract the left and right eyes using the facial landmarks
        leftEye = shape[lStart:lEnd]
        rightEye = shape[rStart:rEnd]

        # Calculate the Eye Aspect Ratio (EAR) for each eye
        leftEAR = eye_aspect_ratio(leftEye)
        rightEAR = eye_aspect_ratio(rightEye)

        # Average the EAR for both eyes
        ear = (leftEAR + rightEAR) / 2.0

        # Create convex hulls around the eyes and draw them on the frame
        leftEyeHull = cv2.convexHull(leftEye)
        rightEyeHull = cv2.convexHull(rightEye)
        cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
        cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)

        # Check if the EAR is below the threshold
        if ear < thresh:
            flag += 1
            print(flag)

            # If consecutive frames have low EAR, trigger the alarm
            if flag >= frame_check:
                cv2.putText(frame, "****************ALERT!****************", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                cv2.putText(frame, "****************ALERT!****************", (10, 325),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                mixer.music.play()
        else:
            flag = 0

    # Display the frame with annotations
    cv2.imshow("Frame", frame)

    # Check for user input to quit the program
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break

# Close all windows and release the camera
cv2.destroyAllWindows()
cap.release()
