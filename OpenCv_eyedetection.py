import cv2
import dlib
import imutils
from scipy.spatial import distance as dist
from imutils import face_utils
import matplotlib.pyplot as plt
import numpy as np
import time

# EAR calculation function
def calculate_EAR(eye):
    y1 = dist.euclidean(eye[1], eye[5])
    y2 = dist.euclidean(eye[2], eye[4])
    x1 = dist.euclidean(eye[0], eye[3])
    EAR = (y1 + y2) / (2.0 * x1)
    return EAR

# Webcam capture
cam = cv2.VideoCapture(0)

# Load dlib models
detector = dlib.get_frontal_face_detector()
landmark_predict = dlib.shape_predictor('/Users/khushitanwar/Downloads/shape_predictor_68_face_landmarks.dat')

# Eye landmark indexes
(L_start, L_end) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(R_start, R_end) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

# Blink logic variables
blink_thresh = 0.25
succ_frame = 2
count_frame = 0
total_blinks = 0
blink_start_time = None
blink_durations = []

# Graph setup
plt.ion()  # turn on interactive mode
fig, ax = plt.subplots()
ear_values = []
frame_numbers = []
line, = ax.plot([], [], label="EAR")
ax.axhline(y=blink_thresh, color='r', linestyle='--', label="Threshold")
ax.set_xlim(0, 100)
ax.set_ylim(0, 0.4)
ax.set_xlabel("Frame")
ax.set_ylabel("EAR")
ax.set_title("Real-time EAR Plot")
ax.legend()
plt.tight_layout()

frame_count = 0

while True:
    ret, frame = cam.read()
    if not ret:
        break

    frame = imutils.resize(frame, width=640)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = detector(gray)
    for face in faces:
        shape = landmark_predict(gray, face)
        shape = face_utils.shape_to_np(shape)

        lefteye = shape[L_start:L_end]
        righteye = shape[R_start:R_end]

        left_EAR = calculate_EAR(lefteye)
        right_EAR = calculate_EAR(righteye)
        avg_EAR = (left_EAR + right_EAR) / 2.0

        # Draw landmarks
        for (x, y) in lefteye:
            cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)
        for (x, y) in righteye:
            cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)

        # Blink detection and duration
        if avg_EAR < blink_thresh:
            count_frame += 1
            if blink_start_time is None:
                blink_start_time = time.time()
        else:
            if count_frame >= succ_frame and blink_start_time is not None:
                total_blinks += 1
                blink_duration = time.time() - blink_start_time
                blink_durations.append(blink_duration)
                cv2.putText(frame, f'Blink Duration: {blink_duration:.2f}s', (30, 30),
                            cv2.FONT_HERSHEY_DUPLEX, 1, (0, 200, 0), 2)
            count_frame = 0
            blink_start_time = None

        # Update EAR graph
        frame_count += 1
        frame_numbers.append(frame_count)
        ear_values.append(avg_EAR)

        if len(frame_numbers) > 100:
            frame_numbers = frame_numbers[-100:]
            ear_values = ear_values[-100:]

        line.set_xdata(frame_numbers)
        line.set_ydata(ear_values)
        ax.set_xlim(max(0, frame_count - 100), frame_count)
        fig.canvas.draw()
        fig.canvas.flush_events()

    # Show total blinks
    cv2.putText(frame, f'Total Blinks: {total_blinks}', (30, 70),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
    cv2.imshow("Camera Feed", frame)

    # Exit when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Clean exit
cam.release()
cv2.destroyAllWindows()
plt.ioff()
plt.show()

