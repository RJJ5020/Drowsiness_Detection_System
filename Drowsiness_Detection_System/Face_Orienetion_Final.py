import cv2
import mediapipe as mp
import numpy as np
import time
from pygame import mixer

# Initialize the FaceMesh object from Mediapipe
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Open the camera
cap = cv2.VideoCapture(0)

# Initialize the sound mixer for the alarm
mixer.init()
mixer.music.load("music.wav")

# Set the threshold time in seconds
look_forward_time_threshold = 5

# Variables to track head pose and alarm status
start_time_not_looking_forward = None
alarm_triggered = False

while cap.isOpened():
    # Capture the image from the camera
    success, image = cap.read()

    # Process the image
    image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = face_mesh.process(image)
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    # Get the 2D and 3D coordinates of facial landmarks
    img_h, img_w, _ = image.shape
    face_3d = []
    face_2d = []

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            for idx, lm in enumerate(face_landmarks.landmark):
                # Select specific facial landmarks for head pose estimation
                if idx == 33 or idx == 263 or idx == 1 or idx == 61 or idx == 291 or idx == 199:
                    if idx == 1:
                        nose_2d = (lm.x * img_w, lm.y * img_h)
                        nose_3d = (lm.x * img_w, lm.y * img_h, lm.z * 8000)

                    x, y = int(lm.x * img_w), int(lm.y * img_h)
                    face_2d.append([x, y])
                    face_3d.append([x, y, lm.z])

            face_2d = np.array(face_2d, dtype=np.float64)
            face_3d = np.array(face_3d, dtype=np.float64)

            # Camera matrix for perspective transformation
            focal_length = 1 * img_w
            cam_matrix = np.array([[focal_length, 0, img_h / 2],
                                   [0, focal_length, img_w / 2],
                                   [0, 0, 1]])
            dist_matrix = np.zeros((4, 1), dtype=np.float64)

            # Solve PnP for head pose estimation
            success, rot_vec, trans_vec = cv2.solvePnP(face_3d, face_2d, cam_matrix, dist_matrix)

            # Convert the rotational vector into a matrix
            rmat, _ = cv2.Rodrigues(rot_vec)

            # Get the angles of head pose
            angles, _, _, _, _, _ = cv2.RQDecomp3x3(rmat)
            x, y = angles[0] * 360, angles[1] * 360

            # Determine the direction of head tilt
            if y < -10:
                text = "Looking Left"
            elif y > 10:
                text = "Looking Right"
            elif x < -10:
                text = "Looking Down"
            else:
                text = "Forward"

                # Reset the timer when looking forward
                start_time_not_looking_forward = None
                alarm_triggered = False

            # Check if not looking forward for a certain time
            if text != "Forward":
                if start_time_not_looking_forward is None:
                    start_time_not_looking_forward = time.time()
                else:
                    elapsed_time = time.time() - start_time_not_looking_forward
                    if elapsed_time >= look_forward_time_threshold and not alarm_triggered:
                        # Trigger alarm here (you can use sound, display, etc.)
                        print("ALARM: Subject not looking forward!")
                        alarm_triggered = True
                        cv2.putText(image, "****************ALERT!****************", (10, 30),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                        cv2.putText(image, "****************ALERT!****************", (10, 325),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                        mixer.music.play()

            # Project 3D coordinates of nose to 2D and draw a line
            nose_3d_projection, _ = cv2.projectPoints(nose_3d, rot_vec, trans_vec, cam_matrix, dist_matrix)
            p1 = (int(nose_2d[0]), int(nose_2d[1]))
            p2 = (int(nose_3d_projection[0][0][0]), int(nose_3d_projection[0][0][1]))

            cv2.line(image, p1, p2, (255, 0, 0), 2)
            cv2.putText(image, text, (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # Display the output
    cv2.imshow('Head Pose Estimation', image)

    # Break the loop if the 'Esc' key is pressed
    if cv2.waitKey(5) & 0xFF == 27:
        break

    # Check for user input to quit the program
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break

# Release the camera and close all windows
cap.release()
cv2.destroyAllWindows()
