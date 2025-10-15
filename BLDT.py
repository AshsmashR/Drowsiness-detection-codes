import os
import sys
import warnings



os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION'] = 'python'
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=DeprecationWarning)
warnings.filterwarnings('ignore', category=FutureWarning)


import io
from contextlib import redirect_stderr
stderr_buffer = io.StringIO()
with redirect_stderr(stderr_buffer):
    import cv2
    import mediapipe as mp
    import numpy as np
    import time
    from collections import deque
sys.stderr = sys.__stderr__


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


def clahe_gray(img_gray):
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4,4))
    return clahe.apply(img_gray)


def pupil_dark_ratio_fixed(eye_gray):
    if eye_gray.size == 0:
        return 0.0
    blurred = cv2.GaussianBlur(eye_gray, (5, 5), 0)
    _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    dark_pixels = np.sum(binary == 255)
    total_pixels = float(eye_gray.size)
    return dark_pixels / total_pixels


def eye_clf_score_fixed(eye_gray_24x24):
    if eye_gray_24x24.size == 0:
        return 0.0
    v = eye_gray_24x24.astype(np.float32)
    mean_intensity = v.mean()
    darkness_score = (255 - mean_intensity) / 255.0
    std_intensity = v.std()
    uniformity_score = 1.0 - (std_intensity / 128.0)
    raw_score = 5.0 * (0.6 * darkness_score + 0.4 * uniformity_score - 0.5) * 2
    return float(raw_score)


def draw_landmarks(frame, landmarks, indices, w, h, color=(0, 255, 0)):
    for idx in indices:
        x = int(landmarks[idx].x * w)
        y = int(landmarks[idx].y * h)
        cv2.circle(frame, (x, y), 1, color, -1)


def eye_bbox_from_landmarks(landmarks, idx, w, h, pad=3):
    xs = [int(landmarks[i].x * w) for i in idx]
    ys = [int(landmarks[i].y * h) for i in idx]
    x1, x2 = max(0, min(xs)-pad), min(w, max(xs)+pad)
    y1, y2 = max(0, min(ys)-pad), min(h, max(ys)+pad)
    return x1, y1, x2, y2


RIGHT_EYE_IDX = [33, 159, 158, 133, 153, 145]
LEFT_EYE_IDX = [362, 380, 374, 263, 386, 385]
FACE_CIRCLE_RADIUS_FACTOR = 0.32


class BlinkAlgo:
    def __init__(self, max_seconds=165, show_window=True, mode="baseline"):
        self.max_seconds = max_seconds
        self.show_window = show_window
        self.mode = mode
        self._stop = False


        # PHASE timing (seconds)
        self.warmup_end = 30
        self.calibration_end = 60
        self.detection_end = 150
        self.relax_end = 165


        self.checkpoints = [90, 120, 150]
        self.checkpoint_windows = {90: 45, 120: 75, 150: 105}
        self.checkpoint_results = {}        
        self.ear_min_threshold = 0.15
        self.ear_max_threshold = 0.32
        self.ear_threshold = None


        # BLINK detection parameters
        self.ear_smooth_window = 3
        self.consec_frames_to_blink = 1
        self.valid_blink_min = 0.05
        self.valid_blink_max = 0.50
        self.perclos_threshold_multiplier = 0.8
        self.blink_rate_threshold_standard = 16
        self.personal_baseline_multiplier = 1.25
        self.perclos_alert_threshold = 7.5
        self.perclos_drowsy_threshold = 15.0


        self.calibrated = False
        self.personal_baseline_blinks_per_min = None
        self.frame_timestamps = deque(maxlen=30)
        self.fps = 30


        # Interpupillary distance (WARMUP ONLY)
        self.ipd_measurements = []
        self.reference_ipd_pixels = None


    def stop(self):
        self._stop = True


    def _get_current_phase(self, elapsed_time):
        if elapsed_time < self.warmup_end: return "warmup"
        elif elapsed_time < self.calibration_end: return "calibration"
        elif elapsed_time < self.detection_end: return "detection"
        else: return "relax"


    def _estimate_fps(self, current_time):
        self.frame_timestamps.append(current_time)
        if len(self.frame_timestamps) >= 2:
            diff = self.frame_timestamps[-1] - self.frame_timestamps[0]
            if diff > 0:
                self.fps = (len(self.frame_timestamps) - 1) / diff
        return self.fps


    def _compute_eye_state_fusion(self, smooth_ear, dark_ratio, clf_score, ear_threshold):
        ear_term = 1.0 if smooth_ear < ear_threshold else 0.0
        dark_term = 1.0 if dark_ratio > 0.75 else (0.5 if dark_ratio > 0.70 else 0.0)
        clf_prob_closed = 1.0 / (1.0 + np.exp(-clf_score))
        clf_term = 1.0 if clf_prob_closed > 0.75 else (0.5 if clf_prob_closed > 0.65 else 0.0)
        closed_score = 0.75 * ear_term + 0.15 * dark_term + 0.15 * clf_term
        eye_open = closed_score < 0.5
        return eye_open, clf_prob_closed


    def compute_perclos(self, ear_log, ear_threshold, window_sec):
        if not ear_log or ear_threshold is None:
            return {'perclos_p80': 0.0, 'samples_analyzed': 0}
        current_time = ear_log[-1][1]
        window_start = current_time - window_sec
        window_data = [(ear, ts) for ear, ts in ear_log if ts >= window_start]
        if not window_data:
            return {'perclos_p80': 0.0, 'samples_analyzed': 0}
        p80_thresh = ear_threshold * self.perclos_threshold_multiplier
        closed_frames = sum(1 for ear, _ in window_data if ear < p80_thresh)
        perclos_p80 = closed_frames / len(window_data) * 100.0
        return {'perclos_p80': float(perclos_p80), 'samples_analyzed': len(window_data)}


    def compute_blink_rate(self, blink_timestamps, window_sec):
        if not blink_timestamps:
            return {'blink_rate_per_min': 0.0, 'total_blinks': 0, 'window_duration': 0.0}
        current_time = blink_timestamps[-1]
        window_start = current_time - window_sec
        sliding_blinks = [ts for ts in blink_timestamps if ts >= window_start]
        if not sliding_blinks:
            return {'blink_rate_per_min': 0.0, 'total_blinks': 0, 'window_duration': 0.0}
        duration = current_time - sliding_blinks[0]
        blink_rate = len(sliding_blinks) / duration * 60.0 if duration > 0 else 0.0
        return {'blink_rate_per_min': float(blink_rate), 'total_blinks': len(sliding_blinks), 'window_duration': float(duration)}


    def compute_average_blink_duration(self, blink_durations, window_blink_count):
        if not blink_durations or window_blink_count == 0:
            return {'avg_duration': 0.0, 'min_duration': 0.0, 'max_duration': 0.0}
        recent_durations = blink_durations[-window_blink_count:]
        return {
            'avg_duration': float(np.mean(recent_durations)),
            'min_duration': float(np.min(recent_durations)),
            'max_duration': float(np.max(recent_durations))
        }
        
    def classify_by_blink_rate(self, blink_rate_per_min):
        if self.mode in ['baseline', 'random']:
            threshold = self.blink_rate_threshold_standard
            status = "Alert" if blink_rate_per_min <= threshold else "Drowsy"
            return {'status': status, 'threshold_used': threshold, 'mode': self.mode, 'dicey_threshold': None}
        elif self.mode == 'after_baseline':
            if self.personal_baseline_blinks_per_min is None:
                return {'status': "Unknown", 'threshold_used': None, 'mode': self.mode}
            alert_thresh = self.personal_baseline_blinks_per_min
            dicey_thresh = alert_thresh * self.personal_baseline_multiplier
            if blink_rate_per_min <= alert_thresh:
                status = "Alert"
            elif blink_rate_per_min <= dicey_thresh:
                status = "Dicey"
            else:
                status = "Drowsy"
            return {'status': status, 'threshold_used': alert_thresh, 'dicey_threshold': dicey_thresh, 'mode': self.mode}
        else:
            return {'status': "Unknown", 'threshold_used': None, 'mode': self.mode}
    def classify_by_perclos(self, perclos_p80):
        if perclos_p80 <= self.perclos_alert_threshold:
            status = 'Alert'
        elif perclos_p80 <= self.perclos_drowsy_threshold:
            status = 'Questionable'
        else:
            status = 'Drowsy'
        return {'status': status, 'perclos_value': perclos_p80, 'alert_threshold': self.perclos_alert_threshold, 'drowsy_threshold': self.perclos_drowsy_threshold}
    def analyze_checkpoint(self, elapsed_time, ear_log, blink_timestamps, blink_durations, ear_threshold):
        checkpoint_time = None
        window_size = None
        for cp in self.checkpoints:
            if abs(elapsed_time - cp) < 1.0:
                checkpoint_time = cp
                window_size = self.checkpoint_windows[cp]
                break
        if checkpoint_time is None: return None
        perclos_metrics = self.compute_perclos(ear_log, ear_threshold, window_size)
        blink_rate_metrics = self.compute_blink_rate(blink_timestamps, window_size)
        blink_duration_metrics = self.compute_average_blink_duration(blink_durations, blink_rate_metrics['total_blinks'])
        blink_rate_class = self.classify_by_blink_rate(blink_rate_metrics['blink_rate_per_min'])
        perclos_class = self.classify_by_perclos(perclos_metrics['perclos_p80'])
        results = {
            'checkpoint_time': checkpoint_time,
            'window_size': window_size,
            'perclos': perclos_metrics,
            'blink_rate': blink_rate_metrics,
            'blink_duration': blink_duration_metrics,
            'classification_blink_rate': blink_rate_class,
            'classification_perclos': perclos_class,
            'timestamp': elapsed_time
        }
        self.checkpoint_results[checkpoint_time] = results
        return results


    def run(self):
        blink_count = 0
        calibration_blink_count = 0
        blink_start_time = None
        blink_durations = []
        blink_timestamps = []
        EAR_SMOOTH = deque(maxlen=self.ear_smooth_window)
        calibration_ears = []
        calibrated = False
        dynamic_ear_threshold = None
        CLOSED_STREAK = 0
        OPEN_STREAK = 0
        ear_log = deque()
        checkpoints_hit = set()
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            return {"error": "Cannot open webcam"}
        face_mesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.7,
        )
        start_time = time.time()
        calibration_start_time = None
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
                self._estimate_fps(current_time)
                current_phase = self._get_current_phase(elapsed_time)
                center = (w // 2, h // 2)
                radius = int(min(h, w) * FACE_CIRCLE_RADIUS_FACTOR)
                mask = np.zeros((h, w), np.uint8)
                cv2.circle(mask, center, radius, 255, -1)
                blurred = cv2.GaussianBlur(orig, (41, 41), 0)
                display = np.where(mask[..., None] == 255, orig, blurred)
                
                # --- Added code for relax phase ---
                if current_phase == "relax" and elapsed_time < self.max_seconds:
                    if self.show_window:
                        cv2.putText(display, "RELAX - TEST COMPLETE", (25, 35), cv2.FONT_HERSHEY_DUPLEX, 0.75, (0, 0, 255), 2)
                        cv2.circle(display, center, radius, (0, 0, 255), 3)
                        cv2.imshow("Blink_Detection", display)
                        if cv2.waitKey(1) & 0xFF == ord('q'):
                            break
                # --- End of added code ---
                
                if results.multi_face_landmarks:
                    landmarks = results.multi_face_landmarks[0].landmark
                    if len(landmarks) > 473:
                        left_pupil = (landmarks[468].x * w, landmarks[468].y * h)
                        right_pupil = (landmarks[473].x * w, landmarks[473].y * h)
                    else:
                        left_pupil = (landmarks[33].x * w, landmarks[33].y * h)
                        right_pupil = (landmarks[263].x * w, landmarks[263].y * h)
                    current_ipd = np.linalg.norm(np.array(left_pupil) - np.array(right_pupil))
                    # =========== WARMUP PHASE (ONLY IPD) ===========
                    if current_phase == "warmup":
                        self.ipd_measurements.append(current_ipd)
                        cv2.putText(
                            display, "WARMUP: Sit ~65cm from camera for 30s...",
                            (25, 35), cv2.FONT_HERSHEY_DUPLEX, 0.75, (0, 120, 255), 2
                        )
                        cv2.putText(
                            display, f"Interpupillary dist: {current_ipd:.1f} px", 
                            (25, 65), cv2.FONT_HERSHEY_DUPLEX, 0.7, (0,0,0), 2
                        )
                        cv2.circle(display, (int(left_pupil[0]), int(left_pupil[1])), 3, (255,0,0), -1)
                        cv2.circle(display, (int(right_pupil[0]), int(right_pupil[1])), 3, (255,0,0), -1)
                        if (elapsed_time >= self.warmup_end and self.reference_ipd_pixels is None and len(self.ipd_measurements) > 2):
                            self.reference_ipd_pixels = float(np.mean(self.ipd_measurements))
                            print(f"[INFO] Reference interpupillary pixel distance (warmup): {self.reference_ipd_pixels:.2f}")
                        cv2.imshow("Blink_Detection", display)
                        if cv2.waitKey(1) & 0xFF == ord('q'):
                            break
                        continue
                    # ========== CALIBRATION PHASE (EAR & BLINKS) ==========
                    leftEAR = compute_ear(landmarks, LEFT_EYE_IDX, w, h)
                    rightEAR = compute_ear(landmarks, RIGHT_EYE_IDX, w, h)
                    avgEAR = (leftEAR + rightEAR) / 2.0
                    if current_phase == "calibration":
                        if calibration_start_time is None: calibration_start_time = current_time
                        calibration_ears.append(avgEAR)
                        if avgEAR < 0.28:
                            CLOSED_STREAK += 1
                            OPEN_STREAK = 0
                            if blink_start_time is None and CLOSED_STREAK >= self.consec_frames_to_blink:
                                blink_start_time = current_time
                        else:
                            OPEN_STREAK += 1
                            if blink_start_time is not None and CLOSED_STREAK >= self.consec_frames_to_blink:
                                blink_duration = current_time - blink_start_time
                                if self.valid_blink_min <= blink_duration <= self.valid_blink_max:
                                    calibration_blink_count += 1
                                blink_start_time = None
                            CLOSED_STREAK = 0
                        if self.show_window:
                            overlay = display.copy()
                            box_x, box_y, box_w, box_h = 10, 10, 320, 80
                            cv2.rectangle(overlay, (box_x, box_y), (box_x + box_w, box_y + box_h), (180, 255, 180), -1)
                            alpha = 0.37
                            cv2.addWeighted(overlay, alpha, display, 1 - alpha, 0, display)
                            cv2.putText(display, "CALIBRATION PHASE", (25, 35), cv2.FONT_HERSHEY_DUPLEX, 0.55, (0, 0, 0), 1)
                            cv2.putText(display, f"Keep eyes OPEN - {max(0, self.calibration_end - elapsed_time):.1f}s remaining", (25, 55), cv2.FONT_HERSHEY_DUPLEX, 0.55, (0, 0, 0), 1)
                            cv2.putText(display, f"EAR: {avgEAR:.3f}", (25, 73), cv2.FONT_HERSHEY_DUPLEX, 0.55, (0, 0, 0), 1)
                            cv2.putText(display, f"Blinks detected: {calibration_blink_count}", (25, 91), cv2.FONT_HERSHEY_DUPLEX, 0.55, (255, 0, 0), 1)
                            cv2.circle(display, center, radius, (0, 255, 0), 2)
                            draw_landmarks(display, landmarks, LEFT_EYE_IDX, w, h, color=(0, 180, 0))
                            draw_landmarks(display, landmarks, RIGHT_EYE_IDX, w, h, color=(0, 180, 0))
                            cv2.imshow("Blink_Detection", display)
                            if cv2.waitKey(1) & 0xFF == ord('q'): break
                        continue
                    # ========== END OF CALIBRATION, SET BLINK BASELINE ==========
                    if not calibrated and len(calibration_ears) > 0 and elapsed_time >= self.calibration_end:
                        n_samples = len(calibration_ears)
                        top_n = max(5, int(0.7 * n_samples))
                        top_open_ears = sorted(calibration_ears)[-top_n:]
                        mean_open = float(np.mean(top_open_ears))
                        std_open = float(np.std(top_open_ears))
                        dynamic_ear_threshold = mean_open - 0.35 * std_open
                        dynamic_ear_threshold = max(self.ear_min_threshold,
                                                    min(self.ear_max_threshold, dynamic_ear_threshold))
                        calibrated = True
                        calibration_duration = current_time - calibration_start_time
                        self.personal_baseline_blinks_per_min = (calibration_blink_count / calibration_duration) * 60.0
                        blink_count = 0
                        blink_timestamps = []
                        blink_durations = []
                        blink_start_time = None
                        CLOSED_STREAK = 0
                        OPEN_STREAK = 0
                    # ========== DETECTION PHASE ==========
                    if current_phase == "detection" and calibrated:
                        lx1, ly1, lx2, ly2 = eye_bbox_from_landmarks(landmarks, LEFT_EYE_IDX, w, h)
                        rx1, ry1, rx2, ry2 = eye_bbox_from_landmarks(landmarks, RIGHT_EYE_IDX, w, h)
                        eye_metrics = []
                        for (x1, y1, x2, y2) in ((lx1, ly1, lx2, ly2), (rx1, ry1, rx2, ry2)):
                            crop = orig[y1:y2, x1:x2]
                            if crop.size == 0: continue
                            gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
                            gray = clahe_gray(gray)
                            dr = pupil_dark_ratio_fixed(gray)
                            small = cv2.resize(gray, (24, 24), interpolation=cv2.INTER_AREA)
                            score = eye_clf_score_fixed(small)
                            eye_metrics.append((dr, score))
                        dark_ratio = float(np.mean([m[0] for m in eye_metrics])) if eye_metrics else 0.0
                        clf_score = float(np.mean([m[1] for m in eye_metrics])) if eye_metrics else 0.0
                        EAR_SMOOTH.append(avgEAR)
                        smooth_ear = float(np.mean(EAR_SMOOTH))
                        eye_open, clf_prob_closed = self._compute_eye_state_fusion(smooth_ear, dark_ratio, clf_score, dynamic_ear_threshold)
                        if not eye_open:
                            CLOSED_STREAK += 1
                            OPEN_STREAK = 0
                            if blink_start_time is None and CLOSED_STREAK >= self.consec_frames_to_blink:
                                blink_start_time = current_time
                        else:
                            OPEN_STREAK += 1
                            if blink_start_time is not None and CLOSED_STREAK >= self.consec_frames_to_blink:
                                blink_duration = current_time - blink_start_time
                                if self.valid_blink_min <= blink_duration <= self.valid_blink_max:
                                    blink_durations.append(blink_duration)
                                    blink_count += 1
                                    blink_timestamps.append(current_time)
                            blink_start_time = None
                            CLOSED_STREAK = 0
                        ear_log.append((smooth_ear, elapsed_time))
                        max_log_duration = 120
                        while ear_log and (elapsed_time - ear_log[0][1]) > max_log_duration:
                            ear_log.popleft()
                        for checkpoint in self.checkpoints:
                            if checkpoint not in checkpoints_hit and abs(elapsed_time - checkpoint) < 1.0:
                                self.analyze_checkpoint(elapsed_time, ear_log, blink_timestamps, blink_durations, dynamic_ear_threshold)
                                checkpoints_hit.add(checkpoint)
                        if self.show_window:
                            overlay = display.copy()
                            box_x, box_y, box_w, box_h = 10, h - 100, 300, 85
                            cv2.rectangle(overlay, (box_x, box_y), (box_x + box_w, box_y + box_h), (255, 255, 255), -1)
                            alpha = 0.55
                            cv2.addWeighted(overlay, alpha, display, 1 - alpha, 0, display)
                            cv2.putText(display, f"Blinks: {blink_count}", (box_x + 10, box_y + 25), cv2.FONT_HERSHEY_DUPLEX, 0.55, (255, 0, 0), 1)
                            cv2.putText(display, f"Eye: {'OPEN' if eye_open else 'CLOSED'}", (box_x + 10, box_y + 45), cv2.FONT_HERSHEY_DUPLEX, 0.55, (0, 0, 0), 1)
                            cv2.putText(display, f"EAR: {smooth_ear:.3f}", (box_x + 10, box_y + 65), cv2.FONT_HERSHEY_DUPLEX, 0.55, (0, 0, 0), 1)
                            avg_blink_duration = np.mean(blink_durations) if blink_durations else 0.0
                            cv2.putText(display, f"Avg Blink Dur: {avg_blink_duration * 1000:.0f} ms", (box_x + 10, box_y + 85), cv2.FONT_HERSHEY_DUPLEX, 0.55, (0, 0, 0), 1)
                            cv2.circle(display, center, radius, (0, 255, 0), 2)
                            draw_landmarks(display, landmarks, LEFT_EYE_IDX, w, h, (0, 255, 0) if eye_open else (0, 0, 255))
                            draw_landmarks(display, landmarks, RIGHT_EYE_IDX, w, h, (0, 255, 0) if eye_open else (0, 0, 255))
                            cv2.imshow("Blink_Detection", display)
                            if cv2.waitKey(1) & 0xFF == ord('q'): break
                else:
                    display = np.where(mask[..., None] == 255, orig, blurred)
                    if self.show_window:
                        cv2.circle(display, center, radius, (0, 0, 255), 2)
                        cv2.putText(display, "NO FACE DETECTED", (25, 30), cv2.FONT_HERSHEY_DUPLEX, 0.7, (0, 0, 0), 1)
                        cv2.imshow("Blink_Detection", display)
                        if cv2.waitKey(1) & 0xFF == ord('d'): break
                if elapsed_time >= self.max_seconds:
                    break
            cap.release()
            cv2.destroyAllWindows()
            final_results = {
                'mode': self.mode,
                'total_blinks': blink_count,
                'personal_baseline_bpm': self.personal_baseline_blinks_per_min,
                'ear_threshold': float(dynamic_ear_threshold) if dynamic_ear_threshold else None,
                'checkpoint_results': self.checkpoint_results,
                'total_duration': elapsed_time
            }
            return final_results
        except Exception as e:
            try:
                cap.release()
                cv2.destroyAllWindows()
            except:
                pass
            return {"error": str(e)}


if __name__ == "__main__":
    test_mode = "baseline"
    print(f"\n{'='*70}")
    print("Blink Detection & Drowsiness Detection System")
    print(f"{'='*70}")
    print(f"Test Mode: {test_mode}")
    print(f"Duration: 165 seconds")
    print("Phases:")
    print("  0-30s:   Warmup (distance setup)")
    print("  30-60s:  Calibration (EAR + baseline capture)")
    print("  60-150s: Detection (checkpoints at 90s, 120s, 150s)")
    print("  150-165s: Relax (no capture)")
    print(f"{'='*70}\n")
    detector = BlinkAlgo(
        max_seconds=165,
        show_window=True,
        mode=test_mode
    )
    print("[INFO] Starting test... Press 'q' to quit early.\n")
    results = detector.run()
    if "error" in results:
        print(f"\n[ERROR] {results['error']}")
    else:
        print(f"\n{'='*70}")
        print("FINAL TEST RESULTS")
        print(f"{'='*70}")
        print(f"Mode: {results['mode']}")
        print(f"Total Duration: {results['total_duration']:.1f}s")
        print(f"Total Blinks: {results['total_blinks']}")
        if results['personal_baseline_bpm']:
            print(f"Personal Baseline: {results['personal_baseline_bpm']:.2f} blinks/min")
        print(f"EAR Threshold: {results['ear_threshold']:.3f}")
        print(f"\n{'='*70}")
        print("CHECKPOINT SUMMARY")
        print(f"{'='*70}")
        for cp_time, cp_data in sorted(results['checkpoint_results'].items()):
            print(f"\nCheckpoint {int(cp_time)}s:")
            print(f"  Blink Rate Status: {cp_data['classification_blink_rate']['status']}")
            print(f"  PERCLOS Status: {cp_data['classification_perclos']['status']}")
            print("  Metrics:")
            print(f"    - Blink Rate: {cp_data['blink_rate']['blink_rate_per_min']:.2f}/min")
            print(f"    - PERCLOS P80: {cp_data['perclos']['perclos_p80']:.2f}%")
            print(f"    - Avg Blink Duration: {cp_data['blink_duration']['avg_duration']*1000:.0f}ms")
        print(f"\n{'='*70}\n")
