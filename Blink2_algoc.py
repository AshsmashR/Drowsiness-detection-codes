import cv2
import mediapipe as mp
import numpy as np
import time
from collections import deque
import os


def gstreamer_pipeline(
    sensor_id=0,
    capture_width=1920,
    capture_height=1080,
    display_width=960,
    display_height=540,
    framerate=30,
    flip_method=0,
):
    return (
        "nvarguscamerasrc sensor-id=%d ! "
        "video/x-raw(memory:NVMM), width=(int)%d, height=(int)%d, framerate=(fraction)%d/1 ! "
        "nvvidconv flip-method=%d ! "
        "video/x-raw, width=(int)%d, height=(int)%d, format=(string)BGRx ! "
        "videoconvert ! "
        "video/x-raw, format=(string)BGR ! appsink"
        % (
            sensor_id,
            capture_width,
            capture_height,
            framerate,
            flip_method,
            display_width,
            display_height,
        )
    )
# ---------------- original settings ----------------
EAR_THRESHOLD = 0.2  # Initial for early detection; updated by calibration
PERCLOS_THRESHOLD = 5.0  # %
LIVE_WINDOW_SEC = 2.15 * 60  # 2 minutes 9 seconds

# MediaPipe setup
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
)

# Eye indices for precise EAR calculation (same as yours)
RIGHT_EYE_IDX = [33, 159, 158, 133, 153, 145]
LEFT_EYE_IDX = [362, 380, 374, 263, 386, 385]


# ---------------- helper funcs ----------------
def euclidean_dist(p1, p2):
    return np.linalg.norm(np.array(p1) - np.array(p2))

def compute_ear(landmarks, idx, w, h):
    p1 = (landmarks[idx[0]].x * w, landmarks[idx[0]].y * h)
    p2 = (landmarks[idx[1]].x * w, landmarks[idx[1]].y * h)
    p3 = (landmarks[idx[2]].x * w, landmarks[idx[2]].y * h)
    p4 = (landmarks[idx[3]].x * w, landmarks[idx[3]].y * h)
    p5 = (landmarks[idx[4]].x * w, landmarks[idx[4]].y * h)
    p6 = (landmarks[idx[5]].x * w, landmarks[idx[5]].y * h)
    vert1 = euclidean_dist(p2, p6)
    vert2 = euclidean_dist(p3, p5)
    horiz = euclidean_dist(p1, p4)
    ear = (vert1 + vert2) / (2.0 * horiz) if horiz > 0 else 0
    return ear

def draw_landmarks(frame, landmarks, indices, w, h, color=(0, 255, 0)):
    for idx in indices:
        x = int(landmarks[idx].x * w)
        y = int(landmarks[idx].y * h)
        cv2.circle(frame, (x, y), 2, color, -1)

# ---------------- robust extras ----------------
def eye_bbox_from_landmarks(landmarks, idx, w, h, pad=3):
    xs = [int(landmarks[i].x * w) for i in idx]
    ys = [int(landmarks[i].y * h) for i in idx]
    x1, x2 = max(0, min(xs)-pad), min(w, max(xs)+pad)
    y1, y2 = max(0, min(ys)-pad), min(h, max(ys)+pad)
    return x1, y1, x2, y2

def clahe_gray(img_gray):
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4,4))
    return clahe.apply(img_gray)

def pupil_dark_ratio(eye_gray):
    if eye_gray.size == 0:
        return 0.0
    thr = max(5, np.percentile(eye_gray, 35))
    dark = (eye_gray < thr).sum()
    return float(dark) / float(eye_gray.size)

# Optional linear model hook (w, b) saved as .npz (w as 1D array, b scalar)
def load_linear_eye_model(path="eye_linear_clf.npz"):
    if os.path.exists(path):
        try:
            data = np.load(path)
            return data["w"], float(data["b"])
        except Exception:
            return None
    return None

LINEAR_MODEL = load_linear_eye_model()  # None if not present

def eye_clf_score(eye_gray_resized_24x24):
    v = eye_gray_resized_24x24.astype(np.float32).reshape(-1)
    if v.size == 0:
        return 0.0
    v = (v - v.mean()) / (v.std() + 1e-6)
    if LINEAR_MODEL is not None:
        w, b = LINEAR_MODEL
        w = w.reshape(-1)
        v = v[:w.size] if v.size >= w.size else np.pad(v, (0, w.size - v.size))
        return float(np.dot(w, v) + b)  # model score (higher can mean "closed" depending on training)
    # fallback heuristic (darkness + inverse variance)
    darkness = 255 - eye_gray_resized_24x24.mean()
    inv_var = 1.0 / (eye_gray_resized_24x24.var() + 1e-3)
    return 0.5 * darkness + 5.0 * inv_var

# ---------------- Face-ID circle params ----------------
FACE_CIRCLE_RADIUS_FACTOR = 0.32  # fraction of min(h,w)
FACE_CIRCLE_PAD = 0               # not used currently

# alert/drowsy logic
ALERT_THRESHOLD = 8  # <=8 blinks/min for Alert
ONE_MINUTE = 60
baseline_duration = 10  # seconds for baseline collection
baseline_blink_count = 0
baseline_collected = False
personal_baseline_blinks = 0  # Will be set after 10s

# calibration & smoothing
calibration_ears = []
calibrated = False
EAR_SMOOTH = deque(maxlen=5)          # 3-5 frame smoothing
OPEN_EAR_WINDOW = deque(maxlen=300)   # recent open EARs for slow adaptation
CLOSED_STREAK = 0
OPEN_STREAK = 0
ADAPT_RATE = 0.01                     # slow EWMA adaptation
CONSEC_FRAMES_TO_BLINK = 1            # require 2-3 consecutive closed frames
VALID_BLINK_MIN = 0.08                # micro blink filter
VALID_BLINK_MAX = 0.60

# ---------------- start capture ----------------
class BlinkAlgo:
    def __init__(self, max_seconds: float = LIVE_WINDOW_SEC, show_window: bool = True,mode: str = "baseline"):
        self.max_seconds = max_seconds
        self.show_window = show_window
        self.mode = mode
        self._stop = False

    def stop(self):
        self._stop = True

    def run(self):
        # Local copies of state variables (same as your originals)
        global EAR_THRESHOLD  # will be updated by calibration
        blink_count = 0
        blink_start_time = None
        blink_durations = []
        blink_timestamps = []
        eye_states = []      # (state, timestamp)
        eye_state_log = deque()
        status = "Neutral"

        ALERT_THRESHOLD = 8
        ONE_MINUTE = 60
        baseline_duration = 10
        baseline_blink_count = 0
        baseline_collected = False
        personal_baseline_blinks = 0

        calibration_ears = []
        calibrated = False
        EAR_SMOOTH = deque(maxlen=5)
        OPEN_EAR_WINDOW = deque(maxlen=300)
        CLOSED_STREAK = 0
        OPEN_STREAK = 0
        ADAPT_RATE = 0.01
        CONSEC_FRAMES_TO_BLINK = 1
        VALID_BLINK_MIN = 0.08
        VALID_BLINK_MAX = 0.60

        FACE_CIRCLE_RADIUS_FACTOR = 0.32
        FACE_CIRCLE_PAD = 0

        # Initialize capture & MediaPipe INSIDE run (safe for threads)
        cap = cv2.VideoCapture(gstreamer_pipeline(flip_method=0), cv2.CAP_GSTREAMER)
        if not cap.isOpened():
            return {"error": "Error: Cannot open webcam"}

        start_time = time.time()
        prev_eye_open = True

        # Create a FaceMesh instance per run (thread-safe pattern)
        mp_face_mesh = mp.solutions.face_mesh
        face_mesh = mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )

        try:
            while not self._stop:
                ret, frame = cap.read()
                if not ret:
                    break

                orig = frame.copy()
                h, w, _ = orig.shape
                frame_rgb = cv2.cvtColor(orig, cv2.COLOR_BGR2RGB)
                results = face_mesh.process(frame_rgb)

                current_time = time.time()
                elapsed_time = current_time - start_time

                avgEAR = 0.0
                eye_open = True
                thr_display = EAR_THRESHOLD if EAR_THRESHOLD is not None else 0.0
                dark_ratio = 0.0
                clf_prob_closed = 0.0
                face_status = "OUT"

                center = (w // 2, h // 2)
                radius = int(min(h, w) * FACE_CIRCLE_RADIUS_FACTOR)

                if results.multi_face_landmarks:
                    landmarks = results.multi_face_landmarks[0].landmark

                    # center check
                    nx, ny = int(landmarks[1].x * w), int(landmarks[1].y * h)
                    dist = np.sqrt((nx - center[0])**2 + (ny - center[1])**2)
                    face_status = "IN" if dist <= radius else "OUT"

                    if face_status == "IN":
                        # EAR
                        leftEAR = compute_ear(landmarks, LEFT_EYE_IDX, w, h)
                        rightEAR = compute_ear(landmarks, RIGHT_EYE_IDX, w, h)
                        avgEAR = (leftEAR + rightEAR) / 2.0

                        # eye metrics
                        lx1, ly1, lx2, ly2 = eye_bbox_from_landmarks(landmarks, LEFT_EYE_IDX,  w, h, pad=4)
                        rx1, ry1, rx2, ry2 = eye_bbox_from_landmarks(landmarks, RIGHT_EYE_IDX, w, h, pad=4)
                        eye_metrics = []
                        for (x1, y1, x2, y2) in ((lx1, ly1, lx2, ly2), (rx1, ry1, rx2, ry2)):
                            crop = orig[y1:y2, x1:x2]
                            if crop.size == 0:
                                continue
                            gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
                            gray = clahe_gray(gray)
                            pr = pupil_dark_ratio(gray)
                            small = cv2.resize(gray, (24, 24), interpolation=cv2.INTER_AREA)
                            score = eye_clf_score(small)
                            eye_metrics.append((pr, score))
                        if eye_metrics:
                            dark_ratio = float(np.mean([m[0] for m in eye_metrics]))
                            clf_score = float(np.mean([m[1] for m in eye_metrics]))
                            clf_prob_closed = 1.0 / (1.0 + np.exp(-0.02 * clf_score))

                        # calibration window
                        if elapsed_time <= baseline_duration:
                            calibration_ears.append(avgEAR)
                            baseline_blink_count = blink_count  # Track blinks during baseline
                            status = "baseline"
                        elif not calibrated and calibration_ears:
                            top_open = sorted(calibration_ears)[-max(3, int(0.7 * len(calibration_ears))):]
                            mean_open = float(np.mean(top_open))
                            std_open = float(np.std(top_open))
                            EAR_THRESHOLD = max(0.15, min(0.32, mean_open - 0.35 * std_open))
                            calibrated = True
                            OPEN_EAR_WINDOW.extend(top_open)
                            print(f"[INFO] Dynamic EAR threshold set: {EAR_THRESHOLD:.3f}")

                        # smoothing
                        EAR_SMOOTH.append(avgEAR)
                        smooth_ear = float(np.mean(EAR_SMOOTH))

                        # fusion
                        likely_open = (avgEAR >= EAR_THRESHOLD) if EAR_THRESHOLD is not None else False
                        if (dark_ratio > 0.32 and clf_prob_closed < 0.45):
                            likely_open = True
                        if likely_open:
                            OPEN_EAR_WINDOW.append(avgEAR)

                        # slow adaptation
                        if calibrated and len(OPEN_EAR_WINDOW) >= 10:
                            open_med = float(np.median(OPEN_EAR_WINDOW))
                            open_std = float(np.std(OPEN_EAR_WINDOW))
                            target_thr = max(0.15, min(0.32, open_med - 0.35 * std_open))
                            EAR_THRESHOLD = (1 - ADAPT_RATE) * (EAR_THRESHOLD or target_thr) + ADAPT_RATE * target_thr

                        ear_term = 1.0 if (EAR_THRESHOLD is not None and smooth_ear < EAR_THRESHOLD) else 0.0
                        dark_term = 1.0 if dark_ratio < 0.22 else (0.5 if dark_ratio < 0.28 else 0.0)
                        clf_term  = 1.0 if clf_prob_closed > 0.55 else (0.5 if clf_prob_closed > 0.48 else 0.0)
                        closed_score = 0.50 * ear_term + 0.30 * dark_term + 0.20 * clf_term
                        eye_open = closed_score < 0.5

                        if eye_open != prev_eye_open:
                            eye_states.append(("Open" if eye_open else "Closed", current_time))
                        prev_eye_open = eye_open

                        # blink timing
                        if not eye_open:
                            CLOSED_STREAK += 1
                            OPEN_STREAK = 0
                            if blink_start_time is None and CLOSED_STREAK >= CONSEC_FRAMES_TO_BLINK:
                                blink_start_time = current_time
                        else:
                            OPEN_STREAK += 1
                            if blink_start_time is not None and CLOSED_STREAK >= CONSEC_FRAMES_TO_BLINK:
                                blink_duration = current_time - blink_start_time
                                if VALID_BLINK_MIN <= blink_duration <= VALID_BLINK_MAX:
                                    blink_durations.append(blink_duration)
                                    blink_count += 1
                                    blink_timestamps.append(current_time)
                                blink_start_time = None
                            CLOSED_STREAK = 0

                        eye_state_log.append(0 if not eye_open else 1)
                        max_log_length = int(LIVE_WINDOW_SEC * (cap.get(cv2.CAP_PROP_FPS) or 30))
                        if len(eye_state_log) > max_log_length:
                            eye_state_log.popleft()

                        # draw landmarks
                        draw_landmarks(orig, landmarks, LEFT_EYE_IDX, w, h, color=(0, 255, 0))
                        draw_landmarks(orig, landmarks, RIGHT_EYE_IDX, w, h, color=(0, 255, 0))
                    else:
                        avgEAR = 0.0
                        eye_open = False

                    cv2.circle(orig, (nx, ny), 3, (255, 0, 0), -1)
                else:
                    avgEAR = 0.0
                    eye_open = False

                # compose preview (optional)
                blurred = cv2.GaussianBlur(orig, (51, 51), 30)
                mask = np.zeros((h, w), dtype="uint8")
                cv2.circle(mask, center, radius, 255, -1)
                mask_3c = cv2.merge([mask, mask, mask])
                inside = cv2.bitwise_and(orig, mask_3c)
                outside = cv2.bitwise_and(blurred, cv2.bitwise_not(mask_3c))
                display_frame = cv2.add(inside, outside)
                circle_color = (0, 255, 0) if face_status == "IN" else (0, 0, 255)
                cv2.circle(display_frame, center, radius, circle_color, 2)

                # Draw text
                if self.show_window:
                    text_color = (20, 20, 20)
                    result_color = (255, 0, 0)
                    recent = [t for t in blink_timestamps if current_time - t <= ONE_MINUTE]
                    window_duration = min(elapsed_time, ONE_MINUTE)
                    blink_rate = (len(recent) / window_duration) * 60 if window_duration > 0 else 0.0
                    eye_state_text = "Open" if eye_open else "Closed"
                    y_offset = 30
                    line_h = 28
                    cv2.putText(display_frame, f"Time: {elapsed_time:.1f}s", (30, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, text_color, 2); y_offset += line_h
                    cv2.putText(display_frame, f"EAR: {avgEAR:.3f}", (30, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, text_color, 2); y_offset += line_h
                    cv2.putText(display_frame, f"Eye State: {eye_state_text}", (30, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, text_color, 2); y_offset += line_h
                    cv2.putText(display_frame, f"Blinks: {blink_count}", (30, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, text_color, 2); y_offset += line_h
                    cv2.putText(display_frame, f"Status: {status}", (30, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, result_color, 2); y_offset += line_h
                    cv2.putText(display_frame, f"Blink Rate: {blink_rate:.2f}/min", (30, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, text_color, 2); y_offset += line_h
                    cv2.imshow("Drowsiness_detector", display_frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break

                # Status logic (moved after blink count to use updated blink_count)
                if elapsed_time > baseline_duration:
                    if not baseline_collected:
                        personal_baseline_blinks = (baseline_blink_count / baseline_duration) * 60 or 7  # Default 7 if 0 (equivalent to ~16 total)
                        baseline_collected = True

                    recent = [t for t in blink_timestamps if current_time - t <= ONE_MINUTE]
                    window_duration = min(elapsed_time, ONE_MINUTE)
                    blink_rate = (len(recent) / window_duration) * 60 if window_duration > 0 else 0.0

                    if blink_rate <= ALERT_THRESHOLD:
                        status = "Alert"
                    elif blink_rate > personal_baseline_blinks * 1.3:
                        status = "Drowsy"
                    else:
                        status = "Neutral"

                if elapsed_time >= self.max_seconds:
                    break

            cap.release()
            cv2.destroyAllWindows()

            # Final metrics
            closed_time = (eye_state_log.count(0) / len(eye_state_log)) * 100 if eye_state_log else 0
            final_recent = [t for t in blink_timestamps if time.time() - t <= ONE_MINUTE]
            final_blink_rate = (len(final_recent) / (min(time.time() - start_time, ONE_MINUTE) or 1)) * 60

            # Final status based on total blinks (as per your simple logic)
            total_blinks = blink_count
            if total_blinks <= 8:
                final_status = "Alert"
            elif total_blinks > 16 * 1.3:
                final_status = "Drowsy"
            else:
                final_status = "Neutral"

            return {
                "total_blinks": total_blinks,
                "blink_rate_per_min": float(final_blink_rate),
                "avg_blink_duration": float(np.mean(blink_durations)) if blink_durations else 0.0,
                "perclos_percent": float(closed_time),
                "status": final_status,  # Use total-based final status
                "baseline_per_min": float(personal_baseline_blinks) if baseline_collected else 0.0,
                "eye_states": [(s, t - start_time) for s, t in eye_states],
                "ear_threshold": float(EAR_THRESHOLD) if EAR_THRESHOLD is not None else None,
            }
        except Exception as e:
            try:
                cap.release()
                cv2.destroyAllWindows()
            except:
                pass
            return {"error": str(e)}

# Optional: allow running this file directly for a quick test (no UI)
if __name__ == "__main__":
    runner = BlinkAlgo(max_seconds=LIVE_WINDOW_SEC, show_window=True)
    out = runner.run()
    if "error" in out:
        print(out["error"])
    else:
        print("\n--- RESULTS ---")
        print(f"Total Blinks: {out['total_blinks']}")
        print(f"Blink Rate (per minute): {out['blink_rate_per_min']:.2f}")
        abd = out.get("avg_blink_duration", 0.0)
        print(f"Average Blink Duration: {abd:.3f} s" if abd else "No blinks detected")
        print(f"PERCLOS: {out['perclos_percent']:.2f}%")
        print(f"Drowsiness Status: {out['status']}")
        print(f"EAR threshold used: {out.get('ear_threshold')}")
