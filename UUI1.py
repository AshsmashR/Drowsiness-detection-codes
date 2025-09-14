import mediapipe as mp
from PyQt5 import QtWidgets, QtGui, QtCore
import sys, csv, os, time
import numpy as np
from PyQt5.QtWidgets import QLabel, QHBoxLayout
from PyQt5.QtGui import QPixmap
# ---------- External modules for detection ----------
import cv2
from collections import deque
from Blink_algoc import BlinkAlgo

# Use a raw string for Windows path to avoid escape issues
CSV_FILE = r"E:\datablink.csv"
CSV_FIELDS = [
    "serial_no", "name", "ID", "result", "baseline_per_min",
    "total_blinks", "blink_rate_per_min", "avg_blink_duration",
    "perclos_percent", "eye_state", "ear_threshold"
]
CSV_RANDOM_FIELDS_PREFIX = "random_test_"  # for additional random result columns

def ensure_csv_exists():
    directory = os.path.dirname(CSV_FILE)
    if directory and not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)
    if not os.path.exists(CSV_FILE) or os.path.getsize(CSV_FILE) == 0:
        with open(CSV_FILE, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=CSV_FIELDS, delimiter=",")
            writer.writeheader()

def read_all_rows():
    ensure_csv_exists()
    try:
        with open(CSV_FILE, "r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f, delimiter=",")
            rows = list(reader)
    except Exception as e:
        print(f"[!] Failed to read CSV: {e}. Recreating file.")
        try: os.remove(CSV_FILE)
        except: pass
        ensure_csv_exists()
        return []
    # Reset if headers mismatch
    header = rows[0] if rows else {}
    if not all(k in header.keys() for k in CSV_FIELDS):
        print("[!] CSV header mismatch: missing required fields. Recreating CSV with correct header.")
        try: os.remove(CSV_FILE)
        except: pass
        ensure_csv_exists()
        return []
    return rows

def get_next_serial():
    rows = read_all_rows()
    max_sn = 0
    for r in rows:
        try:
            sn = int(r.get("serial_no", "") or 0)
            if sn > max_sn:
                max_sn = sn
        except: continue
    return max_sn + 1

def id_exists(uid: str) -> bool:
    if not uid:
        return False
    rows = read_all_rows()
    for r in rows:
        rid = (r.get("ID") or "").strip()
        if rid == uid:
            return True
    return False

def append_new_user(name: str, uid: str, result: str = "") -> bool:
    ensure_csv_exists()
    name = (name or "").strip()
    uid = (uid or "").strip()
    if not name or not uid:
        return False
    if id_exists(uid):
        print(f"[!] Duplicate ID: {uid} - not appended.")
        return False
    serial_no = get_next_serial()
    d = {k: "" for k in CSV_FIELDS}
    d["serial_no"] = serial_no
    d["name"] = name
    d["ID"] = uid
    d["result"] = result
    try:
        with open(CSV_FILE, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=CSV_FIELDS, delimiter=",")
            writer.writerow(d)
        print(f"[+] Appended user: SN={serial_no}, name={name}, ID={uid}")
        return True
    except Exception as e:
        print(f"[!] Failed to append user: {e}")
        return False

def update_test_results_in_csv(user_id: str, mode: str, result_data: dict):
    """
    Save detection test result for a user.
    All fields in CSV_FIELDS are updated if keys are present in result_data.
    For "random_test" mode, values are appended in a new random_test_X column.
    """
    rows = read_all_rows()
    updated = False
    random_count = 1
    # Collect any existing random_test columns
    all_fieldnames = CSV_FIELDS.copy()
    for row in rows:
        for k in row.keys():
            if k.startswith(CSV_RANDOM_FIELDS_PREFIX) and k not in all_fieldnames:
                all_fieldnames.append(k)
                random_count = max(random_count, int(k[len(CSV_RANDOM_FIELDS_PREFIX):] or "1")+1)
    for row in rows:
        if (row.get("ID") or "").strip() == user_id:
            if mode == "baseline":
                for f in CSV_FIELDS:
                    if f in result_data:
                        row[f] = str(result_data[f])
            else:
                # Add results to a new random_test_X column
                col_name = f"{CSV_RANDOM_FIELDS_PREFIX}{random_count}"
                row[col_name] = "|".join([f"{k}:{result_data.get(k,'')}" for k in result_data.keys()])
                if col_name not in all_fieldnames:
                    all_fieldnames.append(col_name)
                random_count += 1
                # Also update the main result field for summary status
                if "status" in result_data:
                    row["result"] = result_data["status"]
            updated = True
            break
    if updated:
        with open(CSV_FILE, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=all_fieldnames, delimiter=",")
            writer.writeheader()
            for r in rows:
                writer.writerow(r)
    return updated


#===============Drowsiness class ==========#
LIVE_WINDOW_SEC = 2.15 * 60
class BlinkTestRunner(QtCore.QObject):
    """
    Runs the blink/drowsiness detection loop in a worker thread
    and emits signals with final results to the UI.
    """
    finished = QtCore.pyqtSignal(dict)
    error = QtCore.pyqtSignal(str)

    def __init__(self, mode="baseline", parent=None):
        super().__init__(parent)
        self.mode = mode  # "baseline" or "testing"
        self._stop = False

    def stop(self):
        self._stop = True

    def run(self):
        try:
            # Create BlinkAlgo instance and run it
            detector = BlinkAlgo(max_seconds=LIVE_WINDOW_SEC, show_window=True, mode=self.mode)
            result = detector.run()

            if "error" in result:
                self.error.emit(result["error"])
            else:
                self.finished.emit(result)
        except Exception as e:
            self.error.emit(str(e))

# ============== MAIN APP UI ==============
class WelcomeApp(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Drowsiness detection test")
        self.setGeometry(100, 100, 800, 800)
        self.setFixedSize(800, 800)
        self.current_widget = None
        self.user_type = None
        try:
            self.setWindowIcon(QtGui.QIcon(r"F:\csir.png"))
        except:
            pass

        self.setStyleSheet("""
    QMainWindow { background-color: #f0f0f0; }
    QLabel { color: #57316B; }
    QLabel[role="title"] {
        color: #000000;
        background-color: transparent;
        padding: 10px;
    }
    QLabel[role="panel"] {
        color: #000000;
        background-color: transparent;
        padding: 10px;
    }
    QPushButton {
        background-color: #00008B;  /* dark blue */
        color: white;
        padding: 10px 20px;
        border: none;
        border-radius: 5px;
        font-weight: bold;
        min-width: 100px;       /* Specify a smaller min width */
        max-width: 200px;
    }
    QPushButton:hover { background-color: #154360; }
    QLineEdit {
        background-color: #D6EAF8;  /* light blue */
        color: #154360;            /* dark blue */
        padding: 6px 10px;
        border: 1px solid #154360;  /* dark blue */
        border-radius: 4px;
    }
""")

        self.mode = None  # "baseline" or "testing"
        self.thread = None
        self.worker = None

        self.user_selected = False  # Track selection state for page 2 "Next" button
        self.user_type = None  # Store new/existing user choice
        

        ensure_csv_exists()
        self.show_welcome_page()

    def _update_page(self, widget):
        """Replace the central widget with the given widget."""
        if self.current_widget:
            self.current_widget.setParent(None)
        self.setCentralWidget(widget)
        self.current_widget = widget

    # ---------------- PAGE 1 ----------------
    def show_welcome_page(self):
        widget = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(widget)
        layout.setAlignment(QtCore.Qt.AlignCenter)

        # Background
        try:
            movie = QtGui.QMovie(r"F:\gifs\1smooth_loop.gif")  # path to your fade transition GIF
            background = QtWidgets.QLabel(widget)
            background.setMovie(movie)
            background.setGeometry(0, 0, 800, 800)
            movie.start()
            background.lower()
        except:
            widget.setStyleSheet("background-color: #f0f0f0;")

        logos_layout = QHBoxLayout()
        logos_layout.setSpacing(50)
        logo_paths = [
            r"F:\csir.png",
            r"F:\Central_Electronics_Engineering_Research_Institute_Logo.png",
            r"F:\csio.jpg",
            r"F:\Ias.jpeg",
            r"F:\IGIB_LOGO.png"
        ]

        for path in logo_paths:
            try:
                pixmap = QPixmap(path).scaled(90, 90, QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation)
                label = QLabel()
                label.setPixmap(pixmap)
                logos_layout.addWidget(label)
            except Exception as e:
                print(f"Error loading logo {path}: {e}")

        layout.addLayout(logos_layout)
        layout.addSpacerItem(QtWidgets.QSpacerItem(20, 20))

        # Logo/GIF
        try:
            movie = QtGui.QMovie(r"D:\eyevideo.gif")
            gif_label = QtWidgets.QLabel()
            gif_label.setMovie(movie)
            gif_label.setFixedSize(500, 200)
            movie.start()
            layout.addWidget(gif_label, alignment=QtCore.Qt.AlignCenter)
        except:
            try:
                logo = QtGui.QPixmap(r"F:\IGIB_LOGO.png").scaled(200, 200, QtCore.Qt.KeepAspectRatio)
                logo_label = QtWidgets.QLabel()
                logo_label.setPixmap(logo)
                layout.addWidget(logo_label, alignment=QtCore.Qt.AlignCenter)
            except:
                pass

        title_label = QtWidgets.QLabel("Welcome to Drowsiness Detection Test")
        title_label.setProperty("role", "title")
        title_label.setFont(QtGui.QFont("Arial", 16, QtGui.QFont.Bold))
        layout.addWidget(title_label, alignment=QtCore.Qt.AlignCenter)

        desc_label = QtWidgets.QLabel(
            "This app uses webcam to detect blinks and assess drowsiness with EAR and PERCLOS algorithms."
        )
        desc_label.setProperty("role", "panel")
        desc_label.setFont(QtGui.QFont("Times New Roman", 10, QtGui.QFont.Bold))
        desc_label.setWordWrap(True)
        desc_label.setFixedWidth(500)
        layout.addWidget(desc_label, alignment=QtCore.Qt.AlignCenter)
        
        layout.addSpacerItem(QtWidgets.QSpacerItem(50, 50))
        next_button = QtWidgets.QPushButton("Next")
        next_button.setFont(QtGui.QFont("Arial", 14, QtGui.QFont.Bold))
        next_button.clicked.connect(self.show_user_type_page)
        layout.addWidget(next_button, alignment=QtCore.Qt.AlignCenter)

        self._update_page(widget)

    # ---------------- PAGE 2 ----------------
    def show_user_type_page(self):
        widget = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(widget)
        layout.setAlignment(QtCore.Qt.AlignCenter)

        try:
             movie = QtGui.QMovie(r"F:\gifs\2smooth_loop.gif") #second transition
             background = QtWidgets.QLabel(widget)
             background.setMovie(movie)
             background.setGeometry(0, 0, 800, 800)
             movie.start()
             background.lower()
        except:
            widget.setStyleSheet("background-color: #f0f0f0;")

        logo_label = QtWidgets.QLabel()
        logo_pixmap = QtGui.QPixmap(r"F:\csir.png").scaled(200, 200, QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation)
        logo_label.setPixmap(logo_pixmap)
        layout.addWidget(logo_label, alignment=QtCore.Qt.AlignCenter)

        title = QtWidgets.QLabel("HELLO, HOW ARE YOU? :-)")
        title.setProperty("role", "title")
        title.setFont(QtGui.QFont("Arial", 16, QtGui.QFont.Bold))
        layout.addWidget(title)

        # New user
        new_panel = QtWidgets.QLabel("Are you a new user?")
        new_panel.setProperty("role", "panel")
        new_font = QtGui.QFont()
        new_font.setBold(True)
        new_panel.setFont(new_font)
        new_btn = QtWidgets.QPushButton("New User")
        new_btn.clicked.connect(self.handle_new_user_selected)
        layout.addWidget(new_panel, alignment=QtCore.Qt.AlignCenter)
        layout.addWidget(new_btn, alignment=QtCore.Qt.AlignCenter)

        # Existing user
        exist_panel = QtWidgets.QLabel("Continue as an existing user")
        exist_panel.setProperty("role", "panel")
        exist_font = QtGui.QFont()
        exist_font.setBold(True)
        exist_panel.setFont(exist_font)
        exist_btn = QtWidgets.QPushButton("Existing User")
        exist_btn.clicked.connect(self.handle_existing_user_selected)
        layout.addWidget(exist_panel, alignment=QtCore.Qt.AlignCenter)
        layout.addWidget(exist_btn, alignment=QtCore.Qt.AlignCenter)
          
        nav_layout = QtWidgets.QHBoxLayout()
        layout.addSpacerItem(QtWidgets.QSpacerItem(30, 30))
        prev_btn = QtWidgets.QPushButton("Previous")
        prev_btn.clicked.connect(self.show_welcome_page)
        
        layout.addSpacerItem(QtWidgets.QSpacerItem(30, 30))
        self.next_btn = QtWidgets.QPushButton("Next")
        self.next_btn.setDisabled(True)  # Disable Next until selection is made
        self.next_btn.clicked.connect(self.handle_next_clicked)
        nav_layout.addWidget(prev_btn)
        nav_layout.addWidget(self.next_btn)

        layout.addLayout(nav_layout)

        self._update_page(widget)

    # Handlers for user type selection
    def handle_new_user_selected(self):
        self.user_selected = True
        self.user_type = "new"
        self.next_btn.setEnabled(True)

    def handle_existing_user_selected(self):
        self.user_selected = True
        self.user_type = "existing"
        self.next_btn.setEnabled(True)

    # Handler for Next button on page 2
    def handle_next_clicked(self):
        if not getattr(self, 'user_selected', False):
            QtWidgets.QMessageBox.warning(self, "Selection Required", "Please click on the page options first.")
            return

        if self.user_type == "new":
            self.show_new_user_page()
        elif self.user_type == "existing":
            self.show_existing_user_page()

    # ---------------- PAGE 3 NEW USER ----------------
    def show_new_user_page(self):
        widget = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(widget)
        layout.setAlignment(QtCore.Qt.AlignCenter)

        try:
             movie = QtGui.QMovie(r"F:\gifs\3smooth_loop.gif") #third transition
             background = QtWidgets.QLabel(widget)
             background.setMovie(movie)
             background.setGeometry(0, 0, 800, 800)
             movie.start()
             background.lower()
        except:
            widget.setStyleSheet("background-color: #f0f0f0;")

        title = QtWidgets.QLabel("Register as New User")
        title.setProperty("role", "title")
        title.setFont(QtGui.QFont("Arial", 18, QtGui.QFont.Bold))
        layout.addWidget(title, alignment=QtCore.Qt.AlignCenter)

        name_label = QtWidgets.QLabel("Enter your name:")
        name_label.setProperty("role", "panel")
        name_label.setFont(QtGui.QFont("Arial", 14, QtGui.QFont.Bold))
        name_entry = QtWidgets.QLineEdit()
        name_entry.setPlaceholderText("Your name (e.g., JOHN)")

        id_label = QtWidgets.QLabel("Enter your Unique ID:")
        id_label.setProperty("role", "panel")
        id_label.setFont(QtGui.QFont("Arial", 14, QtGui.QFont.Bold))

        id_entry = QtWidgets.QLineEdit()
        id_entry.setPlaceholderText("Enter Unique ID (Special & Uppercase Allowed)")
        id_entry.setEchoMode(QtWidgets.QLineEdit.Password)

        # Add eye icon toggle for password visibility
        eye_btn = QtWidgets.QPushButton()
        eye_btn.setCheckable(True)
        eye_btn.setFixedSize(40, 30)
        eye_btn.setIcon(QtGui.QIcon(r"E:\eyeyey.png"))
        eye_btn.setStyleSheet("border: none; background: transparent;")

        def toggle_eye():
            if eye_btn.isChecked():
                id_entry.setEchoMode(QtWidgets.QLineEdit.Normal)
            else:
                id_entry.setEchoMode(QtWidgets.QLineEdit.Password)
        eye_btn.toggled.connect(toggle_eye)

        id_layout = QtWidgets.QHBoxLayout()
        id_layout.addWidget(id_entry)
        id_layout.addWidget(eye_btn)

        layout.addWidget(name_label, alignment=QtCore.Qt.AlignLeft)
        layout.addWidget(name_entry)
        layout.addWidget(id_label, alignment=QtCore.Qt.AlignLeft)
        layout.addLayout(id_layout)

        def save_data():
            name = name_entry.text().strip()
            uid = id_entry.text().strip()
            if not name or not uid:
                QtWidgets.QMessageBox.warning(self, "Error", "Name and ID required!")
                return
            # NEW: Prevent duplicate IDs
            if id_exists(uid):
                QtWidgets.QMessageBox.warning(self, "Duplicate ID", "This ID already exists. Please choose a different ID.")
                return
            saved = append_new_user(name, uid, result="")  # result will be added later
            if not saved:
                QtWidgets.QMessageBox.warning(self, "Duplicate ID", "This ID already exists or could not be saved. Please choose a different ID.")
                return
            self.current_user_id = uid 
            self.show_test_selection_page()
            
        layout.addSpacerItem(QtWidgets.QSpacerItem(20, 20))
        nav_layout = QtWidgets.QHBoxLayout()
        prev_btn = QtWidgets.QPushButton("Previous")
        prev_btn.clicked.connect(self.show_user_type_page)
        layout.addSpacerItem(QtWidgets.QSpacerItem(20, 20))
        save_btn = QtWidgets.QPushButton("Save and Continue")
        save_btn.clicked.connect(save_data)
        nav_layout.addWidget(prev_btn)
        nav_layout.addWidget(save_btn)

        layout.addLayout(nav_layout)
        self._update_page(widget)

    # ---------------- PAGE 3 EXISTING USER ----------------
    def show_existing_user_page(self):
        widget = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(widget)
        layout.setAlignment(QtCore.Qt.AlignCenter)
        try:
             movie = QtGui.QMovie("F:\gifs\5smooth_loop.gif")#4th transition
             background = QtWidgets.QLabel(widget)
             background.setMovie(movie)
             background.setGeometry(0, 0, 800, 800)
             movie.start()
             background.lower()
        except:
            widget.setStyleSheet("background-color: #f0f0f0;")

        title = QtWidgets.QLabel("Existing User Login")
        title.setProperty("role", "title")
        title.setFont(QtGui.QFont("Arial", 18, QtGui.QFont.Bold))
        layout.addWidget(title)

        id_entry = QtWidgets.QLineEdit()
        id_entry.setPlaceholderText("Enter your Unique ID")
        id_entry.setFont(QtGui.QFont("Arial", 12, QtGui.QFont.Bold))
        id_entry.setEchoMode(QtWidgets.QLineEdit.Password)

        eye_btn = QtWidgets.QPushButton()
        eye_btn.setCheckable(True)
        eye_btn.setFixedSize(40, 40)
        eye_btn.setIcon(QtGui.QIcon(r"E:\eyeyey.png"))  # Use your eye icon path
        eye_btn.setStyleSheet("border: none; background: transparent;")
           
        def toggle_eye():
            if eye_btn.isChecked():
                id_entry.setEchoMode(QtWidgets.QLineEdit.Normal)
            else:
                id_entry.setEchoMode(QtWidgets.QLineEdit.Password)
        eye_btn.toggled.connect(toggle_eye)

        id_layout = QtWidgets.QHBoxLayout()
        id_layout.addWidget(id_entry)
        id_layout.addWidget(eye_btn)
        layout.addLayout(id_layout)

        def verify_user():
            uid = id_entry.text().strip()
            if not uid:
                QtWidgets.QMessageBox.warning(self, "Error", "ID required!")
                return
            # NEW: Verify against CSV
            if not id_exists(uid):
                QtWidgets.QMessageBox.warning(self, "User Not Found", "No user with this ID exists. Please register as a new user.")
                return  # block proceed
            self.current_user_id = uid
            self.show_test_selection_page()

        nav_layout = QtWidgets.QHBoxLayout()
        prev_btn = QtWidgets.QPushButton("Previous")
        prev_btn.clicked.connect(self.show_user_type_page)
        layout.addSpacerItem(QtWidgets.QSpacerItem(30, 30))
        next_btn = QtWidgets.QPushButton("Continue")
        next_btn.clicked.connect(verify_user)
        nav_layout.addWidget(prev_btn)
        nav_layout.addWidget(next_btn)

        layout.addLayout(nav_layout)
        self._update_page(widget)

    # ---------------- PAGE 4: TEST SELECTION ----------------
    def show_test_selection_page(self):
        widget = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(widget)
        layout.setAlignment(QtCore.Qt.AlignCenter)
        try:
             movie = QtGui.QMovie(r"F:\gifs\6smooth_loop.gif")
             background = QtWidgets.QLabel(widget)
             background.setMovie(movie)
             background.setGeometry(0, 0, 800, 800)
             movie.start()
             background.lower()
        except:
            widget.setStyleSheet("background-color: #f0f0f0;")
            
            
        title = QtWidgets.QLabel("Please Choose a Test")
        title.setProperty("role", "title")
        title.setFont(QtGui.QFont("Arial", 18, QtGui.QFont.Bold))
        title.setAlignment(QtCore.Qt.AlignCenter)
        layout.addWidget(title)
        
        n_panel = (
            "<ul>"
            "<li>✅New user: Register your baseline</li>"
            "<li>🔄Existing user: Do random testing</li>"
            "<li>🔃To update baseline: Run baseline test</li>"
            "</ul>"
        )
        n_panel = QtWidgets.QLabel(n_panel)
        n_panel.setProperty("role", "panel")
        n_font = QtGui.QFont()
        n_font.setBold(True)
        n_font.setPointSize(11)
        n_panel.setFont(n_font)
        n_panel.setTextFormat(QtCore.Qt.RichText)
        layout.addWidget(n_panel)
        layout.addSpacerItem(QtWidgets.QSpacerItem(80, 80))

        baseline_btn = QtWidgets.QPushButton("Register your baseline")
        baseline_btn.clicked.connect(lambda: self.open_test_window("baseline"))
        layout.addWidget(baseline_btn, alignment=QtCore.Qt.AlignCenter)
        
        layout.addSpacerItem(QtWidgets.QSpacerItem(80, 80))
        
        testing_btn = QtWidgets.QPushButton("Testing (Random)")
        testing_btn.clicked.connect(lambda: self.open_test_window("testing"))
        layout.addWidget(testing_btn, alignment=QtCore.Qt.AlignCenter)
        
        layout.addSpacerItem(QtWidgets.QSpacerItem(80, 80))

        nav_layout = QtWidgets.QHBoxLayout()
        prev_btn = QtWidgets.QPushButton("Previous")
        layout.addSpacerItem(QtWidgets.QSpacerItem(30, 30))
        prev_btn.clicked.connect(self.show_user_type_page)
        next_btn = QtWidgets.QPushButton("Next")
        next_btn.clicked.connect(lambda: QtWidgets.QMessageBox.information(self, "Next", "Please click on existing options"))
        nav_layout.addWidget(prev_btn)
        nav_layout.addWidget(next_btn)

        layout.addLayout(nav_layout)
        self._update_page(widget)

    # ---------------- PAGE 5: TEST WINDOW (2.15 mins, big Start) ----------------
    def open_test_window(self, mode):
        self.mode = mode
        widget = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(widget)
        layout.setAlignment(QtCore.Qt.AlignCenter)
        try:
             movie = QtGui.QMovie(r"F:\gifs\6smooth_loop.gif") #third transition
             background = QtWidgets.QLabel(widget)
             background.setMovie(movie)
             background.setGeometry(0, 0, 800, 800)
             movie.start()
             background.lower()
        except:
            widget.setStyleSheet("background-color: #f0f0f0;")
        
    
        layout.addSpacerItem(QtWidgets.QSpacerItem(10, 10))
 
        title = QtWidgets.QLabel("Test Window (2.15 minutes)")
        title.setProperty("role", "title")
        title.setFont(QtGui.QFont("Arial", 20, QtGui.QFont.Bold))
        layout.addWidget(title, alignment=QtCore.Qt.AlignCenter)
        layout.addSpacerItem(QtWidgets.QSpacerItem(20, 20))
        try:
            movie_gif = QtGui.QMovie(r"D:\eyeyeeymp4.gif")
            gif_label = QtWidgets.QLabel(widget)
            gif_label.setMovie(movie_gif)
            gif_label.setScaledContents(True)  

            movie_gif.setScaledSize(QtCore.QSize(400, 300))
            gif_label.setMovie(movie_gif)
            movie_gif.start()

            layout.addWidget(gif_label, alignment=QtCore.Qt.AlignCenter)
            
            gif_container = QtWidgets.QWidget()
            gif_layout = QtWidgets.QVBoxLayout(gif_container)
            gif_layout.addWidget(gif_label, alignment=QtCore.Qt.AlignCenter)
            layout.addWidget(gif_container, alignment=QtCore.Qt.AlignCenter)
            
        except Exception as e:
            print("GIF Error:", e)
        
        layout.addSpacerItem(QtWidgets.QSpacerItem(30, 30))
        
        sub_text = """
         <ul>
            <li>✅Click start to start the test</li>
            <li>👨🏻Sit upright and keep the face in frame</li>
            <li>😀Stay neutral as possible</li>
        </ul>
        """
        sub = QtWidgets.QLabel(sub_text)
        sub.setProperty("role", "panel")
        sub_font = QtGui.QFont()
        sub_font.setBold(True)
        sub_font.setPointSize(11)
        sub.setFont(sub_font)
        sub.setTextFormat(QtCore.Qt.RichText)
        sub.setAlignment(QtCore.Qt.AlignCenter)
       # Enables HTML formatting
        sub.setWordWrap(True)
        layout.addWidget(sub)

        layout.addSpacerItem(QtWidgets.QSpacerItem(20, 20))
        start_btn = QtWidgets.QPushButton("START")
        start_btn.setMinimumHeight(120)
        start_btn.setMinimumWidth(300)
        start_btn.setStyleSheet("""
            QPushButton {
                background-color: #FF0000;
                color: white;
                font-size: 36px;
                font-weight: bold;
                border-radius: 12px;
                padding: 20px 40px;
            }
            QPushButton:hover { background-color: #28a745; }
        """)
        layout.addWidget(start_btn, alignment=QtCore.Qt.AlignCenter)
        
        layout.addSpacerItem(QtWidgets.QSpacerItem(50, 50))

        nav_layout = QtWidgets.QHBoxLayout()
        prev_btn = QtWidgets.QPushButton("Previous")
        prev_btn.clicked.connect(self.show_test_selection_page)
        nav_layout.addWidget(prev_btn)
        layout.addLayout(nav_layout)

        def start_test():
            start_btn.setDisabled(True)
            self.run_detection()

        start_btn.clicked.connect(start_test)

        self._update_page(widget)

    # ============== THREAD HANDLERS ==============
    def run_detection(self):
        # Prepare worker and thread
        self.thread = QtCore.QThread()
        self.worker = BlinkTestRunner(mode=self.mode)
        self.worker.moveToThread(self.thread)

        # Connect signals
        self.thread.started.connect(self.worker.run)
        self.worker.finished.connect(self.on_test_finished)
        self.worker.error.connect(self.on_test_error)
        self.worker.finished.connect(self.thread.quit)
        self.worker.error.connect(self.thread.quit)
        self.worker.finished.connect(self.worker.deleteLater)
        self.thread.finished.connect(self.thread.deleteLater)

        self.thread.start()

    def on_test_error(self, msg):
        QtWidgets.QMessageBox.critical(self, "Error", msg)

    def on_test_finished(self, result):
        user_id = None
        if self.user_type == "existing":
        # Get current user id (implement your way to get it)
           user_id = self.current_user_id
        elif self.user_type == "new":
        # For new users, you may have stored the ID in a variable during registration
           user_id = self.current_user_id

        if user_id:
           saved = update_test_results_in_csv(user_id, self.mode, result)
           if not saved:
              print("[!] Warning: Could not update CSV results.")

        summary = []
        summary.append(f"Mode: {self.mode}")
        summary.append(f"Total Blinks: {result.get('total_blinks', 0)}")
        baseline_val = result.get('baseline_per_min', 0.0)
        if baseline_val > 0:
           summary.append(f"Personal Baseline: {baseline_val:.2f}/min")
    
    # Show status in summary interface
        status = result.get("status", "Unknown")
        summary.append(f"Status: {status}")

    # Optional: detailed transitions printed to console
        print("\nEye State Transitions (time since start):")
        for s, t in result.get('eye_states', []):
            print(f"{s} at {t:.2f}")
        QtWidgets.QMessageBox.information(self, "Test Results", "\n".join(summary))
        self.show_conclusion_page(status=status)

    # ---------------- PAGE 6: CONCLUSION ----------------
       
    def show_conclusion_page(self,status=""):
        widget = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(widget)
        layout.setAlignment(QtCore.Qt.AlignCenter)
        try:
             movie = QtGui.QMovie(r"F:\gifs\7smooth_loop.gif") 
             background = QtWidgets.QLabel(widget)
             background.setMovie(movie)
             background.setGeometry(0, 0, 800, 800)
             movie.start()
             background.lower()
        except:
            widget.setStyleSheet("background-color: #f0f0f0;")

        # Logo
        logo_label = QtWidgets.QLabel()
        logo_pixmap = QtGui.QPixmap(r"F:\csir.png").scaled(200, 200, QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation)
        logo_label.setPixmap(logo_pixmap)
        layout.addWidget(logo_label, alignment=QtCore.Qt.AlignCenter)
        layout.addSpacerItem(QtWidgets.QSpacerItem(50, 50))

        # Thank you message
        thank_label = QtWidgets.QLabel("Thank you for your time!")
        thank_label.setFont(QtGui.QFont("Arial", 18, QtGui.QFont.Bold))
        thank_label.setAlignment(QtCore.Qt.AlignCenter)
        layout.addWidget(thank_label, alignment=QtCore.Qt.AlignCenter)
        layout.addSpacerItem(QtWidgets.QSpacerItem(50, 50))
        
        status_label = QtWidgets.QLabel()
        status_label.setFont(QtGui.QFont("Arial", 16, QtGui.QFont.Bold))
        status_label.setAlignment(QtCore.Qt.AlignCenter)
        msg = ""
        if status.lower() == "drowsy":
           msg = "Warning: You might be drowsy, please keep a check."
        elif status.lower() == "alert":
           msg = "Status: Alert - you are attentive."
        elif status.lower() == "neutral":
           msg = "Status: Neutral - keep monitoring yourself."
        else:
           msg = f"Status: {status}"
        status_label.setText(msg)
        layout.addWidget(status_label, alignment=QtCore.Qt.AlignCenter)
        layout.addSpacerItem(QtWidgets.QSpacerItem(20, 20))
        # Start Over button - green
        start_over_btn = QtWidgets.QPushButton("Start Over")
        start_over_btn.setStyleSheet("""
        QPushButton {
            background-color: #28a745;  /* green */
            color: white;
            padding: 12px 40px;
            border: none;
            border-radius: 5px;
            font-weight: bold;
            font-size: 16px;
        }
        QPushButton:hover {
            background-color: #218838;
        }
    """)
        start_over_btn.clicked.connect(self.show_welcome_page)
        layout.addWidget(start_over_btn, alignment=QtCore.Qt.AlignCenter)

        self._update_page(widget)

if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    window = WelcomeApp()
    window.show()
    sys.exit(app.exec_())
