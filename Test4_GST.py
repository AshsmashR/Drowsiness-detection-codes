import sys
import os
import re
import csv
import cv2
import mediapipe as mp
import numpy as np
import time
from collections import deque
from PyQt6.QtWidgets import (
    QApplication, QWidget, QLabel, QPushButton, QVBoxLayout, QHBoxLayout, QFrame,
    QStackedWidget, QLineEdit, QMessageBox
)
from PyQt6.QtGui import QPixmap, QFont
from PyQt6.QtCore import Qt

# ------------------ LOGO BAR ------------------
def create_logo_bar():
    logo_layout = QHBoxLayout()
    logo_layout.setSpacing(30)
    logo_layout.setContentsMargins(0, 0, 0, 0)

    logo_paths = [   
        "/Users/khushitanwar/Desktop/Drowsiness_Detect/CSIR-Logo.jpg",
        "/Users/khushitanwar/Desktop/Drowsiness_Detect/IAF.png",
        "/Users/khushitanwar/Desktop/Drowsiness_Detect/IGIB Logo.jpeg",
        "/Users/khushitanwar/Desktop/Drowsiness_Detect/CEERI logo.png",
        "/Users/khushitanwar/Desktop/Drowsiness_Detect/csio logo.jpeg",
    ]

    for path in logo_paths:
        pixmap = QPixmap(path)
        if not pixmap.isNull():
            pixmap = pixmap.scaled(150, 150, Qt.AspectRatioMode.KeepAspectRatio,
                                   Qt.TransformationMode.SmoothTransformation)
            label = QLabel()
            label.setPixmap(pixmap)
            label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            label.setStyleSheet("background: transparent;")
            logo_layout.addWidget(label)

    return logo_layout


# ------------------ Algorithm constants & helpers (unchanged logic) ----------
CAPTURE_DURATION = 160  # total no. of seconds
FPS = 30
CONSEC_FRAMES = 5       # minimum consecutive frames EAR < threshold to count as blink
BLINK_LOCK_FRAMES = 10  # frames to lock after a blink (~0.33s)

LEFT_EYE_IDX = [33, 160, 158, 133, 153, 144]
RIGHT_EYE_IDX = [362, 385, 387, 263, 373, 380]

INITIAL_DYNAMIC_EAR = 0.26
EAR_WINDOW_SIZE = 5

def calculate_EAR(landmarks, eye_indices):
    p = [np.array(landmarks[i]) for i in eye_indices]
    vertical_1 = np.linalg.norm(p[1] - p[5])
    vertical_2 = np.linalg.norm(p[2] - p[4])
    horizontal = np.linalg.norm(p[0] - p[3])
    return (vertical_1 + vertical_2) / (2.0 * horizontal)

def robust_dynamic_EAR(EAR_list, margin=0.02):
    """Compute robust EAR threshold excluding blink-like dips."""
    if not EAR_list:
        return INITIAL_DYNAMIC_EAR
    valid_EARs = [e for e in EAR_list if e > 0.15]  # filter noise
    if not valid_EARs:
        return INITIAL_DYNAMIC_EAR
    median_EAR = np.median(valid_EARs)
    mean_EAR = np.mean(valid_EARs)
    # Combine stability: weighted median + mean
    dynamic = 0.6 * median_EAR + 0.4 * mean_EAR
    # Drop margin and clamp
    return max(dynamic - margin, 0.18)

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

# ------------------ CSV helpers (single users.csv) ------------------
def get_users_csv_path():
    folder_path = os.path.join(os.path.expanduser("~"),  "Desktop", "Drowsiness_Detect", "CSV Files")
    os.makedirs(folder_path, exist_ok=True)
    return os.path.join(folder_path, "user2.csv")

# canonical order of columns (we will ensure these exist, appended if missing)
def canonical_fieldnames():
    names = ["User ID"]
    for i in range(1, 6):
        names += [
            f"Baseline{i}_TotalBlinks",
            f"Baseline{i}_Timestamp",
            f"Baseline{i}_Status"
        ]
    names += ["Average_Baseline_Blinks"]
    names += ["Testing_TotalBlinks", "Testing_Timestamp", "Testing_FinalStatus"]
    names += ["Random_TotalBlinks", "Random_Timestamp", "Random_FinalStatus"]
    return names

def read_users_csv():
    csv_file = get_users_csv_path()
    if not os.path.isfile(csv_file):
        return [], None
    with open(csv_file, mode="r", newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        fieldnames = reader.fieldnames if reader.fieldnames else None
    return rows, fieldnames

def write_users_csv(rows, fieldnames):
    csv_file = get_users_csv_path()
    # Ensure fieldnames contains all canonical columns
    canon = canonical_fieldnames()
    if fieldnames is None:
        fieldnames = canon.copy()
    # add any missing canonical columns
    for c in canon:
        if c not in fieldnames:
            fieldnames.append(c)
    with open(csv_file, mode="w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            # ensure all keys exist
            out = {fn: r.get(fn, "") for fn in fieldnames}
            writer.writerow(out)

def update_user_row(user_id, updates: dict):
    """
    updates: dict of column->value
    Ensures file/row exists. Creates row if user not found.
    """
    rows, fieldnames = read_users_csv()
    if fieldnames is None:
        fieldnames = canonical_fieldnames()
    # ensure rows all have fields
    for r in rows:
        for fn in fieldnames:
            if fn not in r:
                r[fn] = ""
    # find user
    found = False
    for r in rows:
        if r.get("User ID", "") == user_id:
            for k, v in updates.items():
                r[k] = v
            found = True
            break
    if not found:
        new_row = {fn: "" for fn in fieldnames}
        new_row["User ID"] = user_id
        for k, v in updates.items():
            new_row[k] = v
        rows.append(new_row)
    write_users_csv(rows, fieldnames)

def get_user_row(user_id):
    rows, fieldnames = read_users_csv()
    if not rows:
        return None
    for r in rows:
        if r.get("User ID", "") == user_id:
            return r
    return None

def get_next_baseline_slot(user_id):
    """
    Returns baseline index 1..5 for the next empty slot, or None if all filled.
    We don't overwrite existing baseline entries automatically.
    """
    row = get_user_row(user_id)
    if row is None:
        return 1
    for i in range(1, 6):
        tb_col = f"Baseline{i}_TotalBlinks"
        if (row.get(tb_col, "") is None) or (str(row.get(tb_col, "")).strip() == ""):
            return i
    return None

def compute_and_save_average_baseline(user_id):
    row = get_user_row(user_id)
    if row is None:
        return
    vals = []
    for i in range(1, 6):
        tb_col = f"Baseline{i}_TotalBlinks"
        v = row.get(tb_col, "")
        try:
            if v is not None and str(v).strip() != "":
                vals.append(float(v))
        except:
            pass
    avg = ""
    if vals:
        avg = str(sum(vals) / len(vals))
    update_user_row(user_id, {"Average_Baseline_Blinks": avg})

def get_user_average_baseline(user_id):
    row = get_user_row(user_id)
    if row is None:
        return None
    v = row.get("Average_Baseline_Blinks", "")
    try:
        if v is None or str(v).strip() == "":
            return None
        return float(v)
    except:
        return None

# ------------------ New helper: append dynamic TestingN columns ------------------
def append_testing_result(user_id, total_blinks, final_status, timestamp):
    """
    Adds new columns Testing{n}_TotalBlinks, Testing{n}_Timestamp, Testing{n}_FinalStatus
    and writes the values for the given user_id without overwriting previous Testing columns.
    """
    rows, fieldnames = read_users_csv()
    if fieldnames is None:
        # create file with canonical header if it doesn't exist
        write_users_csv([], canonical_fieldnames())
        rows, fieldnames = read_users_csv()

    # ensure each row has all current fieldnames
    for r in rows:
        for fn in fieldnames:
            if fn not in r:
                r[fn] = ""

    # find or create user row
    user_row = None
    for r in rows:
        if r.get("User ID", "") == user_id:
            user_row = r
            break
    if user_row is None:
        # create new row with empty values for existing fieldnames
        user_row = {fn: "" for fn in fieldnames}
        user_row["User ID"] = user_id
        rows.append(user_row)

    # find highest existing TestingN index based on fieldnames
    testing_nums = []
    pattern = re.compile(r"^Testing(\d+)_TotalBlinks$")
    for fn in fieldnames:
        m = pattern.match(fn)
        if m:
            try:
                testing_nums.append(int(m.group(1)))
            except:
                pass
    next_num = max(testing_nums) + 1 if testing_nums else 1

    # new column names
    blink_col = f"Testing{next_num}_TotalBlinks"
    time_col = f"Testing{next_num}_Timestamp"
    status_col = f"Testing{next_num}_FinalStatus"

    # append these columns to fieldnames and ensure all rows have them
    if blink_col not in fieldnames:
        fieldnames.append(blink_col)
    if time_col not in fieldnames:
        fieldnames.append(time_col)
    if status_col not in fieldnames:
        fieldnames.append(status_col)

    for r in rows:
        # add keys if missing
        if blink_col not in r:
            r[blink_col] = ""
        if time_col not in r:
            r[time_col] = ""
        if status_col not in r:
            r[status_col] = ""

    # set values for user
    user_row[blink_col] = str(total_blinks)
    user_row[time_col] = timestamp
    user_row[status_col] = final_status

    # write back to CSV
    write_users_csv(rows, fieldnames)

# ------------------ Core: run_blink_test (keeps algorithm intact) ----------
def run_blink_test(test_type, user_id, show_instructions=True):
    """
    Runs webcam-based blink detection using the algorithm you provided unchanged.
    Returns a tuple: (total_blinks:int, final_status:str, timestamp:str)
    Also updates the CSV through callers (or callers can call update_user_row).
    """
    if show_instructions:
        msg = QMessageBox()
        msg.setWindowTitle(f"{test_type} Instructions")
        msg.setText(
            "Soon the camera will start.\n"
            "Make sure you look into the camera.\n"
            "There will be a 30-second adjusting period at the beginning where blinks are NOT registered.\n"
            "Baseline: up to 5 readings allowed.\n"
            "Press OK to start."
        )
        msg.exec()

    cap = cv2.VideoCapture(gstreamer_pipeline(flip_method=0), cv2.CAP_GSTREAMER)
    if not cap.isOpened():
        return {"error": "Error: Cannot open webcam"}


    start_time = time.time()
    mp_face_mesh = mp.solutions.face_mesh

    total_blinks = 0
    measurement_time_sec = 0.0
    dynamic_EAR_threshold = INITIAL_DYNAMIC_EAR
    status = "Unknown"
    phase_blinks = {"Baseline": 0, "Phase 1": 0, "Phase 2": 0}
    EAR_values_phase = []
    frame_counter = 0
    blink_lock_counter = 0
    ear_buffer = deque(maxlen=EAR_WINDOW_SIZE)

    phase_info = [
        (0, 30, "Adjusting"),
        (30, 90, "Baseline"),
        (90, 120, "Phase 1"),
        (120, 150, "Phase 2"),
        (150, 160, "Final")
    ]

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

            # Determine current phase
            current_phase = "Final"
            for start, end, name in phase_info:
                if start <= elapsed < end:
                    current_phase = name
                    break

            # Set dynamic EAR threshold 
            if current_phase == "Baseline":
                dynamic_EAR_threshold = 0.25
            elif current_phase in ["Phase 1", "Phase 2"]:
                if EAR_values_phase:
                    dynamic_EAR_threshold = robust_dynamic_EAR(EAR_values_phase)


            # Process face
            if results.multi_face_landmarks and current_phase != "Final":
                for face_landmarks in results.multi_face_landmarks:
                    ih, iw, _ = frame.shape
                    landmarks = [(lm.x * iw, lm.y * ih) for lm in face_landmarks.landmark]

                    left_EAR = calculate_EAR(landmarks, LEFT_EYE_IDX)
                    right_EAR = calculate_EAR(landmarks, RIGHT_EYE_IDX)
                    avg_EAR = (left_EAR + right_EAR) / 2.0

                    ear_buffer.append(avg_EAR)
                    smooth_EAR = np.mean(ear_buffer)

                    if current_phase in ["Baseline", "Phase 1", "Phase 2"]:
                        # only collect EARs when eyes are open (avoid blink contamination)
                        if avg_EAR > 0.18:
                            EAR_values_phase.append(avg_EAR)

                    # Blink detection with lockout and smoothing
                    if current_phase not in ["Adjusting", "Final"]:
                        measurement_time_sec += 1 / FPS
                        if smooth_EAR < dynamic_EAR_threshold:
                            frame_counter += 1
                        else:
                            if frame_counter >= CONSEC_FRAMES and blink_lock_counter == 0:
                                total_blinks += 1
                                # After detecting a blink, skip updating EAR threshold for next few frames
                                skip_dynamic_update_counter = 10
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

            # Update phase status
            if current_phase in ["Baseline", "Phase 1", "Phase 2"] and measurement_time_sec > 0:
                blinks_per_minute_phase = phase_blinks[current_phase] / (measurement_time_sec / 60)
                status = "Alert" if blinks_per_minute_phase <= 16 else "Drowsy"

            # Overall status
            if measurement_time_sec > 0:
                blinks_per_minute_total = total_blinks / (measurement_time_sec / 60)
                overall_status = "Alert" if blinks_per_minute_total <= 16 else "Drowsy"
            else:
                overall_status = status

            # HUD
            hud_y = 50
            cv2.putText(frame, f"Phase: {current_phase}", (30, hud_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
            cv2.putText(frame, f"Elapsed: {int(elapsed)}s", (30, hud_y+50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2)

            if current_phase in ["Baseline", "Phase 1", "Phase 2"]:
                cv2.putText(frame, f"Phase Status: {status}", (30, hud_y+100),
                            cv2.FONT_HERSHEY_SIMPLEX, 1,
                            (0, 255, 0) if status == "Alert" else (0, 0, 255), 2)
                cv2.putText(frame, f"Dynamic EAR: {dynamic_EAR_threshold:.2f}", (30, hud_y+150),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 165, 0), 2)

            cv2.putText(frame, f"Total Blinks: {total_blinks}", (30, hud_y+200),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)

            # For Testing runs: during 150-160s, show relative classification using average baseline (live)
            if test_type == "Testing" and 150 <= elapsed < 160:
                avg_baseline = get_user_average_baseline(user_id)
                if avg_baseline is not None:
                    if total_blinks <= avg_baseline:
                        t_status = "Alert"
                        color = (0, 255, 0)
                    elif total_blinks <= 1.25 * avg_baseline:
                        t_status = "Ambiguous"
                        color = (0, 255, 255)
                    else:
                        t_status = "Drowsy"
                        color = (0, 0, 255)
                    cv2.putText(frame, f"Testing Status (live): {t_status}", (30, hud_y+260),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

            # Final HUD
            if current_phase == "Final":
                cv2.putText(frame, f"Final Status: {overall_status}", (30, hud_y+260),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 200, 255), 2)

            cv2.imshow(f"{test_type} - Eye Detection", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()

    # Final computations (use effective time of 120s as original)
    effective_measurement_time = 120
    blinks_per_minute_total = total_blinks / (effective_measurement_time / 60) if effective_measurement_time > 0 else 0

    # Classification
    if test_type == "Testing":
        avg_baseline = get_user_average_baseline(user_id)
        if avg_baseline is not None:
            if total_blinks <= avg_baseline:
                final_status = "Alert"
            elif total_blinks <= 1.25 * avg_baseline:
                final_status = "Ambiguous"
            else:
                final_status = "Drowsy"
        else:
            final_status = "Alert" if blinks_per_minute_total <= 16 else "Drowsy"
    else:
        final_status = "Alert" if blinks_per_minute_total <= 16 else "Drowsy"

    timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    return int(total_blinks), final_status, timestamp

# ------------------ UI pages (preserve your UI; add baseline/testing/random logic) ------------------

# Page 1: Welcome (unchanged)
class WelcomePage(QWidget):
    def __init__(self, stacked_widget):
        super().__init__()
        self.stacked_widget = stacked_widget
        self.setWindowTitle("DROWSINESS DETECTION TEST")
        self.setGeometry(200, 200, 1000, 700)

        self.setStyleSheet("""
            QWidget {
                background-image: url('/Users/khushitanwar/Desktop/Drowsiness_Detect/Background.jpeg');
                background-position: center;
                background-repeat: no-repeat;
                background-size: cover;
            }
        """)

        main_layout = QVBoxLayout()
        main_layout.setSpacing(20)
        main_layout.setContentsMargins(30, 30, 30, 30)

        main_layout.addLayout(create_logo_bar())

        title = QLabel("Welcome to the Drowsiness Detection Test 👁")
        title.setFont(QFont("Arial", 28, QFont.Weight.Bold))
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title.setStyleSheet("color: #000000;")
        main_layout.addWidget(title)

        # Instruction Box
        instruction_box = QFrame()
        instruction_box.setFixedWidth(500)
        instruction_box.setStyleSheet("""
            QFrame {
                background: rgba(255, 255, 255, 0.85);
                border-radius: 15px;
                padding: 20px;
                border: 1px solid #cccccc;
            }
        """)
        instruction_layout = QVBoxLayout()
        instruction_layout.setSpacing(10)

        instructions = [
            "👤 Sit comfortably in front of the camera",
            "📷 Make sure you are at eye level with the camera",
            "💡 Ensure good lighting",
            "▶️ Click 'Start' when ready"
        ]

        for text in instructions:
            lbl = QLabel(text)
            lbl.setFont(QFont("Arial", 16))
            lbl.setStyleSheet("color: #000000;")
            lbl.setAlignment(Qt.AlignmentFlag.AlignLeft)
            instruction_layout.addWidget(lbl)

        instruction_box.setLayout(instruction_layout)
        main_layout.addWidget(instruction_box, alignment=Qt.AlignmentFlag.AlignCenter)

        # Start button
        start_button = QPushButton("👁 Start the Test")
        start_button.setFont(QFont("Arial", 18, QFont.Weight.Bold))
        start_button.setFixedSize(260, 70)
        start_button.setStyleSheet("""
            QPushButton {
                background-color: #0b3d91;
                color: white;
                border-radius: 20px;
                padding: 12px;
                font-weight: bold;
            }
            QPushButton:hover { background-color: #062c6e; }
        """)
        start_button.clicked.connect(lambda: self.stacked_widget.setCurrentIndex(1))
        main_layout.addWidget(start_button, alignment=Qt.AlignmentFlag.AlignCenter)
        main_layout.addStretch()
        self.setLayout(main_layout)

# Page 2: User Choice (unchanged)
class UserChoicePage(QWidget):
    def __init__(self, stacked_widget):
        super().__init__()
        self.stacked_widget = stacked_widget

        self.setStyleSheet("""
    QWidget {
        background-image: url('/Users/khushitanwar/Desktop/Drowsiness_Detect/Background.jpeg');
        background-position: center;
        background-repeat: no-repeat;
        background-size: cover;
    }
""")

        main_layout = QVBoxLayout()
        main_layout.setSpacing(20)
        main_layout.setContentsMargins(30, 30, 30, 30)

        main_layout.addLayout(create_logo_bar())

        title = QLabel("Select your profile")
        title.setFont(QFont("Arial", 28, QFont.Weight.Bold))
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title.setStyleSheet("color: black;")
        main_layout.addWidget(title)
        main_layout.addSpacing(30)

       # Buttons
        existing_user_btn = QPushButton("👤 Existing User")
        new_user_btn = QPushButton("🆕 Register New ID")

        for btn in [existing_user_btn, new_user_btn]:
            btn.setFont(QFont("Arial", 16, QFont.Weight.Bold))
            btn.setFixedSize(280, 60)
            btn.setStyleSheet("""
                QPushButton {
                    background-image: url('/Users/khushitanwar/Desktop/Drowsiness_Detect/Background.jpeg');
                    background-position: center;
                    background-repeat: no-repeat;
                    background-size: cover;
                    color: white;
                    font-weight: bold;
                    border-radius: 8px;
                    border: 2px solid rgba(255, 255, 255, 0.7);
                    padding: 8px;
                }
                QPushButton:hover {
                    background-color: rgba(255, 255, 255, 0.2);
                    border: 2px solid white;
                }
            """)
            main_layout.addWidget(btn, alignment=Qt.AlignmentFlag.AlignCenter)

    
    
        new_user_btn.clicked.connect(lambda: self.stacked_widget.setCurrentIndex(2))
        existing_user_btn.clicked.connect(lambda: self.stacked_widget.setCurrentIndex(3))

        main_layout.addSpacing(30)
        main_layout.addStretch()

        back_btn = QPushButton("⬅️ Back")
        back_btn.setFont(QFont("Arial", 14, QFont.Weight.Bold))
        back_btn.setFixedSize(120, 50)
        back_btn.setStyleSheet("""
            QPushButton { background-color: #0b3d91; color: white; border-radius: 10px; }
            QPushButton:hover { background-color: #062c6e; }
        """)
        back_layout = QHBoxLayout()
        back_layout.addWidget(back_btn, alignment=Qt.AlignmentFlag.AlignLeft)
        main_layout.addLayout(back_layout)

        self.setLayout(main_layout)

# Page 3: New User Registration (unchanged except using CSV helpers)
class NewUserPage(QWidget):
    def __init__(self, stacked_widget):
        super().__init__()
        self.stacked_widget = stacked_widget
        self.setStyleSheet("""
            QWidget {
                background-image: url('/Users/khushitanwar/Desktop/Drowsiness_Detect/Background.jpeg');
                background-position: center;
                background-repeat: no-repeat;
                background-size: cover;
            }
        """)
        main_layout = QVBoxLayout()
        main_layout.addLayout(create_logo_bar())

        title = QLabel("Register New User")
        title.setFont(QFont("Arial", 28, QFont.Weight.Bold))
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        main_layout.addWidget(title)
        main_layout.addSpacing(30)

        self.new_user_id_input = QLineEdit()
        self.new_user_id_input.setPlaceholderText("Create a new User ID (6 characters)")
        self.new_user_id_input.setFixedHeight(40)
        self.new_user_id_input.setFixedWidth(300)
        self.new_user_id_input.setStyleSheet("""
            QLineEdit { background-color: white; color: black; border-radius: 8px; padding: 5px; }
        """)
        main_layout.addWidget(self.new_user_id_input, alignment=Qt.AlignmentFlag.AlignCenter)

        create_btn = QPushButton("Create New ID")
        create_btn.setFont(QFont("Arial", 16, QFont.Weight.Bold))
        create_btn.setFixedSize(250, 60)
        create_btn.setStyleSheet("""
            QPushButton { background-color: #0b3d91; color: black; border-radius: 15px; }
            QPushButton:hover { background-color: #062c6e; }
        """)
        create_btn.clicked.connect(self.create_new_id)
        main_layout.addWidget(create_btn, alignment=Qt.AlignmentFlag.AlignCenter)
        main_layout.addStretch()

        back_btn = QPushButton("⬅️ Back")
        back_btn.setFont(QFont("Arial", 14, QFont.Weight.Bold))
        back_btn.setFixedSize(120, 50)
        back_btn.setStyleSheet("""
            QPushButton {
                    background-image: url('/Users/khushitanwar/Desktop/Drowsiness_Detect/Background.jpeg');
                    background-position: center;
                    background-repeat: no-repeat;
                    background-size: cover;
                    color: white;
                    font-weight: bold;
                    border-radius: 8px;
                    border: 2px solid rgba(255, 255, 255, 0.7);
                    padding: 8px;
                }
                QPushButton:hover {
                    background-color: rgba(255, 255, 255, 0.2);
                    border: 2px solid white;
                }
            """)
        back_btn.clicked.connect(lambda: self.stacked_widget.setCurrentIndex(1))
        main_layout.addWidget(back_btn, alignment=Qt.AlignmentFlag.AlignLeft)
        self.setLayout(main_layout)

    def create_new_id(self):
        user_id = self.new_user_id_input.text().strip()
        if len(user_id) != 6 or not user_id.isalnum():
            QMessageBox.warning(self, "Error", "User ID must be 6 alphanumeric characters")
            return

        csv_file = get_users_csv_path()
        file_exists = os.path.isfile(csv_file)
        if not file_exists:
            # create with canonical header
            write_users_csv([], canonical_fieldnames())

        # append new user row
        rows, fieldnames = read_users_csv()
        if fieldnames is None:
            fieldnames = canonical_fieldnames()
        new_row = {fn: "" for fn in fieldnames}
        new_row["User ID"] = user_id
        rows.append(new_row)
        write_users_csv(rows, fieldnames)

        QMessageBox.information(self, "Success", f"New User ID created: {user_id}")
        self.new_user_id_input.clear()

# Page 4: Existing User Login (store current_user)
class ExistingUserPage(QWidget):
    def __init__(self, stacked_widget):
        super().__init__()
        self.stacked_widget = stacked_widget
        self.setStyleSheet("""
            QWidget {
                background-image: url('/Users/khushitanwar/Desktop/Drowsiness_Detect/Background.jpeg');
                background-position: center;
                background-repeat: no-repeat;
                background-size: cover;
            }
        """)
        main_layout = QVBoxLayout()
        main_layout.addLayout(create_logo_bar())

        title = QLabel("Existing User Login")
        title.setFont(QFont("Arial", 28, QFont.Weight.Bold))
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        main_layout.addWidget(title)
        main_layout.addSpacing(30)

        self.user_id_input = QLineEdit()
        self.user_id_input.setPlaceholderText("Enter your User ID")
        self.user_id_input.setFixedHeight(40)
        self.user_id_input.setFixedWidth(300)
        # keep white background + black text explicitly
        self.user_id_input.setStyleSheet("QLineEdit { background-color: white; color: black; border-radius: 8px; padding: 5px; }")
        main_layout.addWidget(self.user_id_input, alignment=Qt.AlignmentFlag.AlignCenter)

        login_btn = QPushButton("Login")
        login_btn.setFont(QFont("Arial", 16, QFont.Weight.Bold))
        login_btn.setFixedSize(250, 60)
        login_btn.setStyleSheet("""
            QPushButton {
                    background-image: url('/Users/khushitanwar/Desktop/Drowsiness_Detect/Background.jpeg');
                    background-position: center;
                    background-repeat: no-repeat;
                    background-size: cover;
                    color: white;
                    font-weight: bold;
                    border-radius: 8px;
                    border: 2px solid rgba(255, 255, 255, 0.7);
                    padding: 8px;
                }
                QPushButton:hover {
                    background-color: rgba(255, 255, 255, 0.2);
                    border: 2px solid white;
                }
            """)
        login_btn.clicked.connect(self.verify_user)
        main_layout.addWidget(login_btn, alignment=Qt.AlignmentFlag.AlignCenter)
        main_layout.addStretch()

        back_btn = QPushButton("⬅️ Back")
        back_btn.setFont(QFont("Arial", 14, QFont.Weight.Bold))
        back_btn.setFixedSize(120, 50)
        back_btn.setStyleSheet("""
            QPushButton { background-color: #0b3d91; color: white; border-radius: 10px; }
            QPushButton:hover { background-color: #062c6e; }
        """)
        back_btn.clicked.connect(lambda: self.stacked_widget.setCurrentIndex(1))
        main_layout.addWidget(back_btn, alignment=Qt.AlignmentFlag.AlignLeft)
        self.setLayout(main_layout)

    def verify_user(self):
        user_id = self.user_id_input.text().strip()
        if not user_id:
            QMessageBox.warning(self, "Error", "Please enter your User ID")
            return

        csv_file = get_users_csv_path()
        if not os.path.isfile(csv_file):
            QMessageBox.warning(self, "Error", "No user data found!")
            return

        found = False
        rows, _ = read_users_csv()
        for row in rows:
            if row.get("User ID", "") == user_id:
                found = True
                break

        if found:
            # store current user on the stacked widget (MainApp)
            try:
                self.stacked_widget.current_user = user_id
            except Exception:
                pass
            QMessageBox.information(self, "Success", f"Welcome back, {user_id}!")
            self.stacked_widget.setCurrentIndex(4)
        else:
            QMessageBox.warning(self, "Error", "User ID not found!")

# Page 5: Eye Blink Capture (links to baseline/testing/random)
# ------------------ Page 5: Eye Blink Capture (Baseline / Testing / Random) ------------------
class EyeBlinkPage(QWidget):
    def __init__(self, stacked_widget):
        super().__init__()
        self.stacked_widget = stacked_widget

        # 🌄 Set page-specific background image
        self.setStyleSheet("""
            QWidget {
                background-image: url(/Users/khushitanwar/Desktop/Drowsiness_Detect/Background.jpeg);
                background-position: center;
                background-repeat: no-repeat;
                background-size: cover;
            }
            QLabel {
                color: white;
                background: transparent;
            }
        """)

        # 🔹 Layout setup
        main_layout = QVBoxLayout(self)
        main_layout.addLayout(create_logo_bar())

        # Title
        title = QLabel("Eye Blink Capture")
        title.setFont(QFont("Poppins", 32, QFont.Weight.Bold))
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title.setStyleSheet("color: black; text-shadow: 1px 1px 2px #000000;")
        main_layout.addWidget(title)

        subtitle = QLabel("Select the type of test to begin 👁️")
        subtitle.setFont(QFont("Poppins", 18))
        subtitle.setAlignment(Qt.AlignmentFlag.AlignCenter)
        subtitle.setStyleSheet("color: black; background: transparent;")
        main_layout.addWidget(subtitle)
        main_layout.addSpacing(10)

        # Small note
        baseline_note = QLabel("You can register up to 5 baseline measurements.")
        baseline_note.setFont(QFont("Poppins", 13))
        baseline_note.setAlignment(Qt.AlignmentFlag.AlignCenter)
        baseline_note.setStyleSheet("color: red; background: transparent;")
        main_layout.addWidget(baseline_note)
        main_layout.addSpacing(30)

        # Buttons
        button_layout = QVBoxLayout()
        button_layout.setSpacing(20)

        btn_baseline = QPushButton("📋 Register Baseline")
        btn_testing = QPushButton("👁️ Testing")
        btn_random = QPushButton("🎲 Random Testing")

        for btn in [btn_baseline, btn_testing, btn_random]:
            btn.setFont(QFont("Poppins", 16, QFont.Weight.Bold))
            btn.setFixedSize(300, 70)
            btn.setStyleSheet("""
                QPushButton {
                    background-color: rgba(255, 255, 255, 0.2);
                    color: black;
                    border: 2px solid rgba(255, 255, 255, 0.6);
                    border-radius: 15px;
                    font-weight: bold;
                }
                QPushButton:hover {
                    background-color: rgba(255, 255, 255, 0.35);
                    border: 2px solid white;
                }
            """)
            button_layout.addWidget(btn, alignment=Qt.AlignmentFlag.AlignCenter)

        main_layout.addLayout(button_layout)
        main_layout.addStretch()

        # Back button
        back_btn = QPushButton("⬅️ Back")
        back_btn.setFont(QFont("Poppins", 14, QFont.Weight.Bold))
        back_btn.setFixedSize(140, 50)
        back_btn.setStyleSheet("""
            QPushButton {
                background-color: rgba(0, 0, 0, 0.4);
                color: white;
                border-radius: 10px;
                border: 2px solid rgba(255,255,255,0.5);
            }
            QPushButton:hover {
                background-color: rgba(0, 0, 0, 0.6);
            }
        """)
        back_btn.clicked.connect(lambda: self.stacked_widget.setCurrentIndex(3))
        main_layout.addWidget(back_btn, alignment=Qt.AlignmentFlag.AlignLeft)

        # Connect buttons
        btn_baseline.clicked.connect(self.open_baseline)
        btn_testing.clicked.connect(self.open_testing)
        btn_random.clicked.connect(self.open_random)


    def get_logged_in_user(self):
        return getattr(self.stacked_widget, "current_user", None)

    def open_baseline(self):
        user_id = self.get_logged_in_user()
        if not user_id:
            QMessageBox.warning(self, "Error", "Please login as an existing user first.")
            return
        slot = get_next_baseline_slot(user_id)
        if slot is None:
            QMessageBox.information(self, "Baselines Full", "You have already recorded 5 baselines.")
            return
        # Run the algorithm (unchanged)
        total_blinks, final_status, timestamp = run_blink_test(f"Baseline{slot}", user_id)
        if total_blinks is None:
            return
        # Save into corresponding baseline columns (do not overwrite other baseline slots)
        updates = {
            f"Baseline{slot}_TotalBlinks": str(total_blinks),
            f"Baseline{slot}_Timestamp": timestamp,
            f"Baseline{slot}_Status": final_status
        }
        update_user_row(user_id, updates)
        # recompute average
        compute_and_save_average_baseline(user_id)
        QMessageBox.information(self, "Baseline Saved",
                                f"Baseline {slot} saved.\nTotal blinks: {total_blinks}\nStatus: {final_status}")
        # 🔹 After saving baseline, go to Thank You Page
        self.stacked_widget.setCurrentIndex(5)

    def open_testing(self):
        user_id = self.get_logged_in_user()
        if not user_id:
            QMessageBox.warning(self, "Error", "Please login as an existing user first.")
            return
        avg = get_user_average_baseline(user_id)
        if avg is None:
            QMessageBox.warning(self, "Error", "No baseline found for this user. Please register at least one baseline first.")
            return
        total_blinks, final_status, timestamp = run_blink_test("Testing", user_id)
        if total_blinks is None:
            return

        # Use dynamic appending: Testing1, Testing2, ...
        append_testing_result(user_id, total_blinks, final_status, timestamp)

        QMessageBox.information(self, "Testing Complete",
                                f"Testing saved.\nTotal blinks: {total_blinks}\nStatus: {final_status}")
        # 🔹 After saving testing, go to Thank You Page
        self.stacked_widget.setCurrentIndex(5)

    def open_random(self):
        user_id = self.get_logged_in_user()
        if not user_id:
            QMessageBox.warning(self, "Error", "Please login as an existing user first.")
            return
        total_blinks, final_status, timestamp = run_blink_test("Random Testing", user_id)
        if total_blinks is None:
            return
        updates = {
            "Random_TotalBlinks": str(total_blinks),
            "Random_Timestamp": timestamp,
            "Random_FinalStatus": final_status
        }
        update_user_row(user_id, updates)
        QMessageBox.information(self, "Random Test Saved",
                                f"Random test saved.\nTotal blinks: {total_blinks}\nStatus: {final_status}")
        # 🔹 After saving random test, go to Thank You Page
        self.stacked_widget.setCurrentIndex(5)

# ------------------ Page: Thank You Page ------------------
class ThankYouPage(QWidget):
    def __init__(self, stacked_widget):
        super().__init__()
        self.stacked_widget = stacked_widget
        self.init_ui()

    def init_ui(self):
        # 🌅 Set custom background image
        self.setStyleSheet("""
            QWidget {
                background-image: url('/Users/khushitanwar/Desktop/Drowsiness_Detect/Background.jpeg');
                background-position: center;
                background-repeat: no-repeat;
                background-size: cover;
            }
            QLabel {
                color: white;
                background: transparent;
            }
        """)

        layout = QVBoxLayout()
        layout.setSpacing(40)
        layout.setContentsMargins(80, 80, 80, 80)

        title = QLabel("✅ Thank You!")
        title.setFont(QFont("Poppins", 40, QFont.Weight.Bold))
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title.setStyleSheet("color: black; text-shadow: 1px 1px 2px #000000;")
        layout.addWidget(title)

        msg = QLabel("Your test has been completed successfully.")
        msg.setFont(QFont("Poppins", 20))
        msg.setAlignment(Qt.AlignmentFlag.AlignCenter)
        msg.setStyleSheet("color: black;")
        layout.addWidget(msg)

        btn_layout = QHBoxLayout()
        btn_layout.setSpacing(40)
        btn_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)

        home_btn = QPushButton("🏠 Return Home")
        home_btn.setFont(QFont("Poppins", 16, QFont.Weight.Bold))
        home_btn.setFixedSize(220, 60)
        home_btn.setStyleSheet("""
            QPushButton {
                background-color: rgba(255, 255, 255, 0.2);
                color: white;
                border-radius: 15px;
                border: 2px solid rgba(255,255,255,0.6);
            }
            QPushButton:hover {
                background-color: rgba(255, 255, 255, 0.35);
                border: 2px solid white;
            }
        """)
        home_btn.clicked.connect(self.go_home)

        retest_btn = QPushButton("🔁 Run Another Test")
        retest_btn.setFont(QFont("Poppins", 16, QFont.Weight.Bold))
        retest_btn.setFixedSize(220, 60)
        retest_btn.setStyleSheet("""
            QPushButton {
                background-color: rgba(0, 0, 0, 0.4);
                color: white;
                border-radius: 15px;
                border: 2px solid rgba(255,255,255,0.6);
            }
            QPushButton:hover {
                background-color: rgba(0, 0, 0, 0.6);
                border: 2px solid white;
            }
        """)
        retest_btn.clicked.connect(self.run_again)

        btn_layout.addWidget(home_btn)
        btn_layout.addWidget(retest_btn)

        layout.addLayout(btn_layout)
        layout.addStretch()
        self.setLayout(layout)

    def go_home(self):
        for i in range(self.stacked_widget.count()):
            if self.stacked_widget.widget(i).__class__.__name__ == "WelcomePage":
                self.stacked_widget.setCurrentIndex(i)
                break

    def run_again(self):
        for i in range(self.stacked_widget.count()):
            if self.stacked_widget.widget(i).__class__.__name__ == "EyeBlinkPage":
                self.stacked_widget.setCurrentIndex(i)
                break


# ------------------ Main Application ------------------
class MainApp(QStackedWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Drowsiness Detection Test")
        self.setGeometry(200, 200, 1000, 700)

        # pages
        self.welcome_page = WelcomePage(self)
        self.user_choice_page = UserChoicePage(self)
        self.new_user_page = NewUserPage(self)
        self.existing_user_page = ExistingUserPage(self)
        self.eye_blink_page = EyeBlinkPage(self)
        self.thank_you_page = ThankYouPage(self)
        
        # add pages
        self.addWidget(self.welcome_page)        # 0
        self.addWidget(self.user_choice_page)    # 1
        self.addWidget(self.new_user_page)       # 2
        self.addWidget(self.existing_user_page)  # 3
        self.addWidget(self.eye_blink_page)      # 4
        self.addWidget(self.thank_you_page)      # 5

        # current user holder
        self.current_user = None

        self.setCurrentIndex(0)


# ------------------ Run Application ------------------
if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainApp()
    window.show()
    sys.exit(app.exec())