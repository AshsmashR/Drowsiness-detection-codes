import cv2
import mediapipe as mp
import numpy as np
import time
from collections import deque

# ---------- Parameters ----------
CAPTURE_DURATION = 160  # total no. of seconds
FPS = 30
CONSEC_FRAMES = 3       # minimum consecutive frames EAR < threshold to count as blink
BLINK_LOCK_FRAMES = 10  # frames to lock after a blink (~0.33s)

# Eye landmark indices
LEFT_EYE_IDX = [33, 160, 158, 133, 153, 144]
RIGHT_EYE_IDX = [362, 385, 387, 263, 373, 380]

# Default baseline EAR
INITIAL_DYNAMIC_EAR = 0.26

# Rolling EAR window for smoothing
EAR_WINDOW_SIZE = 5

# ---------- EAR Calculation ----------
def calculate_EAR(landmarks, eye_indices):
    p = [np.array(landmarks[i]) for i in eye_indices]
    vertical_1 = np.linalg.norm(p[1] - p[5])
    vertical_2 = np.linalg.norm(p[2] - p[4])
    horizontal = np.linalg.norm(p[0] - p[3])
    return (vertical_1 + vertical_2) / (2.0 * horizontal)

# ---------- Robust Dynamic EAR (median-based) ----------
def robust_dynamic_EAR(EAR_list, margin=0.01):
    if not EAR_list:
        return INITIAL_DYNAMIC_EAR
    median_EAR = np.median(np.array(EAR_list))
    return max(median_EAR - margin, 0.18)

# ---------- Initialize ----------
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("❌ Camera not accessible.")
    exit()

fps = cap.get(cv2.CAP_PROP_FPS) or FPS
start_time = time.time()
mp_face_mesh = mp.solutions.face_mesh

# Phase info: start_sec, end_sec, name
phase_info = [
    (0, 30, "Adjusting"),
    (30, 90, "Baseline"),
    (90, 120, "Phase 1"),
    (120, 150, "Phase 2"),
    (150, 160, "Final")
]

# ---------- Tracking variables ----------
total_blinks = 0
measurement_time_sec = 0.0
dynamic_EAR_threshold = INITIAL_DYNAMIC_EAR
status = "Unknown"
phase_blinks = {"Baseline": 0, "Phase 1": 0, "Phase 2": 0}
EAR_values_phase = []

frame_counter = 0
blink_lock_counter = 0

# EAR smoothing
ear_buffer = deque(maxlen=EAR_WINDOW_SIZE)

with mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
) as face_mesh:

    while cap.isOpened():
        elapsed = time.time() - start_time
        if elapsed >= CAPTURE_DURATION:
            break

        ret, frame = cap.read()
        if not ret:
            continue

        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb_frame)

        # ---------- Determine current phase ----------
        current_phase = "Final"
        for start, end, name in phase_info:
            if start <= elapsed < end:
                current_phase = name
                break

        # ---------- Set Dynamic EAR ----------
        if current_phase == "Baseline":
            dynamic_EAR_threshold = 0.25
        elif current_phase in ["Phase 1", "Phase 2"]:
            if EAR_values_phase:
                dynamic_EAR_threshold = robust_dynamic_EAR(EAR_values_phase)

        # ---------- Process face ----------
        if results.multi_face_landmarks and current_phase != "Final":
            for face_landmarks in results.multi_face_landmarks:
                ih, iw, _ = frame.shape
                landmarks = [(lm.x * iw, lm.y * ih) for lm in face_landmarks.landmark]

                left_EAR = calculate_EAR(landmarks, LEFT_EYE_IDX)
                right_EAR = calculate_EAR(landmarks, RIGHT_EYE_IDX)
                avg_EAR = (left_EAR + right_EAR) / 2.0

                # Append to rolling buffer for smoothing
                ear_buffer.append(avg_EAR)
                smooth_EAR = np.mean(ear_buffer)

                if current_phase in ["Baseline", "Phase 1", "Phase 2"]:
                    EAR_values_phase.append(avg_EAR)

                # Blink detection with lockout and smoothing
                if current_phase not in ["Adjusting", "Final"]:
                    measurement_time_sec += 1 / fps
                    if smooth_EAR < dynamic_EAR_threshold:
                        frame_counter += 1
                    else:
                        if frame_counter >= CONSEC_FRAMES and blink_lock_counter == 0:
                            total_blinks += 1
                            phase_blinks[current_phase] += 1
                            blink_lock_counter = BLINK_LOCK_FRAMES
                        frame_counter = 0

                if blink_lock_counter > 0:
                    blink_lock_counter -= 1

                # Draw eye landmarks
                for eye_indices in [LEFT_EYE_IDX, RIGHT_EYE_IDX]:
                    for idx in eye_indices:
                        x, y = int(landmarks[idx][0]), int(landmarks[idx][1])
                        cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)

        # ---------- Update phase status ----------
        if current_phase in ["Baseline", "Phase 1", "Phase 2"] and measurement_time_sec > 0:
            blinks_per_minute_phase = phase_blinks[current_phase] / (measurement_time_sec / 60)
            status = "Alert" if blinks_per_minute_phase <= 16 else "Drowsy"

        # ---------- Update overall status ----------
        if measurement_time_sec > 0:
            blinks_per_minute_total = total_blinks / (measurement_time_sec / 60)
            overall_status = "Alert" if blinks_per_minute_total <= 16 else "Drowsy"
        else:
            overall_status = status

        # ---------- HUD ----------
        cv2.putText(frame, f"Phase: {current_phase}", (30, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        cv2.putText(frame, f"Elapsed: {int(elapsed)}s", (30, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2)
        if current_phase in ["Baseline", "Phase 1", "Phase 2"]:
            cv2.putText(frame, f"Phase Status: {status}", (30, 150),
                        cv2.FONT_HERSHEY_SIMPLEX, 1,
                        (0, 255, 0) if status == "Alert" else (0, 0, 255), 2)
            cv2.putText(frame, f"Dynamic EAR: {dynamic_EAR_threshold:.2f}", (30, 200),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 165, 0), 2)

        cv2.putText(frame, f"Total Blinks: {total_blinks}", (30, 250),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
        if current_phase == "Final":
            cv2.putText(frame, f"Final Status: {overall_status}", (30, 300),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 200, 255), 2)

        cv2.imshow("Eye Detection + EAR", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()

# ---------- Final Computation ----------
effective_measurement_time = 120
blinks_per_minute_total = total_blinks / (effective_measurement_time / 60)
final_status = "Alert" if blinks_per_minute_total <= 16 else "Drowsy"

print("---- Per Phase Blinks ----")
for phase, blinks in phase_blinks.items():
    print(f"{phase}: {blinks} blinks")
print(f"Final Status: {final_status}")
print(f"Total blinks: {total_blinks}")















