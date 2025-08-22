
from PyQt5 import QtWidgets, QtGui, QtCore

import sys, os, csv, time
import numpy as np
from collections import deque
import cv2
import mediapipe as mp

# ------------------- CSV HELPERS -------------------
CSV_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "users.csv")

def ensure_csv():
    if not os.path.exists(CSV_FILE):
        with open(CSV_FILE, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["serial_no","ID","baseline_blinks","baseline_rate","baseline_avg_dur","baseline_perclos","baseline_status"])

def next_serial_no():
    ensure_csv()
    with open(CSV_FILE, "r", newline="") as f:
        reader = list(csv.reader(f))
        return len(reader) if len(reader)>1 else 1

def read_user_baseline(user_id):
    ensure_csv()
    with open(CSV_FILE, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row["ID"] == user_id and row["baseline_blinks"]:
                # return baseline metrics
                try:
                    return {
                        "baseline_blinks": float(row["baseline_blinks"]),
                        "baseline_rate": float(row["baseline_rate"]),
                        "baseline_avg_dur": float(row["baseline_avg_dur"]),
                        "baseline_perclos": float(row["baseline_perclos"]),
                        "baseline_status": row["baseline_status"],
                    }
                except:
                    return None
    return None

def write_user_baseline(user_id, results):
    # Append new row if ID not present; if present and empty baseline, update first occurrence
    ensure_csv()
    updated = False
    rows = []
    with open(CSV_FILE, "r", newline="") as f:
        rows = list(csv.reader(f))
    if len(rows)==0:
        rows = [["serial_no","ID","baseline_blinks","baseline_rate","baseline_avg_dur","baseline_perclos","baseline_status"]]
    # Try to update existing row with same ID if that row has blank baseline entries
    for i in range(1, len(rows)):
        if len(rows[i]) >= 2 and rows[i][1] == user_id:
            # Update row (overwrite baseline fields)
            if len(rows[i]) < 7:
                rows[i] += [""]*(7-len(rows[i]))
            rows[i][2] = f"{results['TotalBlinks']}"
            rows[i][3] = f"{results['BlinkRate']:.2f}"
            rows[i][4] = f"{results['AvgBlinkDur']:.3f}"
            rows[i][5] = f"{results['PERCLOS']:.2f}"
            rows[i][6] = results["Status"]
            updated = True
            break
    if not updated:
        rows.append([
            f"{next_serial_no()}",
            user_id,
            f"{results['TotalBlinks']}",
            f"{results['BlinkRate']:.2f}",
            f"{results['AvgBlinkDur']:.3f}",
            f"{results['PERCLOS']:.2f}",
            results["Status"]
        ])
    with open(CSV_FILE, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(rows)

def append_user_id_if_new(user_id):
    ensure_csv()
    # Only add ID row if not present at all
    with open(CSV_FILE, "r", newline="") as f:
        reader = list(csv.reader(f))
        # header + rows
        if any((len(r)>=2 and r[1]==user_id) for r in reader[1:]):
            return
    with open(CSV_FILE, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([f"{next_serial_no()}", user_id, "", "", "", "", ""])



###############blink detection algo goes here can be changed however we need#################
# ------------------- QTHREAD WORKER -------------------
  #class BlinkWorker(QtCore.QThread):

# ------------------- MAIN APP -------------------
class WelcomeApp(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("DrowsyDodger")
        self.setGeometry(100, 100, 800, 800)
        self.setFixedSize(800, 800)
        self.current_widget = None
        self.user_name = ""
        self.user_id = ""  # persisted across pages
        try:
            self.setWindowIcon(QtGui.QIcon(r"E:\eyeyey.png"))
        except:
            pass

        # Global theme
        self.setStyleSheet("""
            QMainWindow { background-color: #f0f0f0; }
            QLabel { color: #57316B; }
            QLabel[role="title"] {
                color: #DD137B;
                background-color: beige;
                padding: 10px;
            }
            QLabel[role="panel"] {
                color: #57316B;
                background-color: beige;
                padding: 10px;
            }
            QPushButton {
                background-color: #FF69B4;
                color: white;
                padding: 12px 40px;
                border: none;
                border-radius: 5px;
                font-weight: bold;
            }
            QPushButton:hover { background-color: #C71585; }
            QLineEdit {
                background-color: #FFE4E1;
                color: #8A0253;
                padding: 6px 10px;
                border: 1px solid #C71585;
                border-radius: 4px;
            }
            QTextEdit { background-color: #FFF8F0; color: #57316B; }
        """)
        self.show_welcome_page()

    def _update_page(self, widget):
        if self.current_widget:
            self.current_widget.deleteLater()
        self.current_widget = widget
        self.setCentralWidget(widget)

    # ---------------- PAGE 1 ----------------
    def show_welcome_page(self):
        widget = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(widget)
        layout.setAlignment(QtCore.Qt.AlignCenter)

        try:
            pixmap = QtGui.QPixmap(r"E:\eyebackground.png")
            pixmap = pixmap.scaled(800, 800, QtCore.Qt.KeepAspectRatioByExpanding)
            bg = QtWidgets.QLabel(widget)
            bg.setPixmap(pixmap)
            bg.setGeometry(0,0,800,800)
            bg.setStyleSheet("background-color: rgba(255, 255, 255, 150);")
        except:
            widget.setStyleSheet("background-color: #f0f0f0;")

        try:
            movie = QtGui.QMovie(r"D:\eyevideo.gif")
            gif_label = QtWidgets.QLabel()
            gif_label.setMovie(movie)
            gif_label.setFixedSize(500, 200)
            movie.start()
            layout.addWidget(gif_label, alignment=QtCore.Qt.AlignCenter)
        except:
            try:
                logo = QtGui.QPixmap(r"E:\eyeyey.png").scaled(200, 200, QtCore.Qt.KeepAspectRatio)
                logo_label = QtWidgets.QLabel()
                logo_label.setPixmap(logo)
                layout.addWidget(logo_label, alignment=QtCore.Qt.AlignCenter)
            except:
                pass

        title = QtWidgets.QLabel("Welcome to DrowsyDodger")
        title.setProperty("role", "title")
        title.setFont(QtGui.QFont("Arial", 24, QtGui.QFont.Bold))
        layout.addWidget(title, alignment=QtCore.Qt.AlignCenter)

        desc = QtWidgets.QLabel("This app detects blinks and assesses drowsiness using algorithms like EAR and Perclose via your webcam.")
        desc.setProperty("role", "panel")
        desc.setFont(QtGui.QFont("Arial", 12))
        desc.setWordWrap(True)
        desc.setFixedWidth(500)
        layout.addWidget(desc, alignment=QtCore.Qt.AlignCenter)

        next_btn = QtWidgets.QPushButton("Next")
        next_btn.clicked.connect(self.show_user_type_page)
        layout.addWidget(next_btn, alignment=QtCore.Qt.AlignCenter)

        self._update_page(widget)

    # ---------------- PAGE 2 ----------------
    def show_user_type_page(self):
        widget = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(widget)
        layout.setAlignment(QtCore.Qt.AlignCenter)

        title = QtWidgets.QLabel("HELLO, HOW ARE YOU")
        title.setProperty("role", "title")
        title.setFont(QtGui.QFont("Arial", 20, QtGui.QFont.Bold))
        layout.addWidget(title)

        new_panel = QtWidgets.QLabel("Are you a new user?")
        new_panel.setProperty("role", "panel")
        new_btn = QtWidgets.QPushButton("New User")
        new_btn.clicked.connect(self.show_new_user_page)
        layout.addWidget(new_panel, alignment=QtCore.Qt.AlignCenter)
        layout.addWidget(new_btn, alignment=QtCore.Qt.AlignCenter)

        exist_panel = QtWidgets.QLabel("Continue as an existing user")
        exist_panel.setProperty("role", "panel")
        exist_btn = QtWidgets.QPushButton("Existing User")
        exist_btn.clicked.connect(self.show_existing_user_page)
        layout.addWidget(exist_panel, alignment=QtCore.Qt.AlignCenter)
        layout.addWidget(exist_btn, alignment=QtCore.Qt.AlignCenter)

        nav = QtWidgets.QHBoxLayout()
        prev_btn = QtWidgets.QPushButton("Previous")
        prev_btn.clicked.connect(self.show_welcome_page)
        nav.addWidget(prev_btn)
        layout.addStretch()
        layout.addLayout(nav)

        self._update_page(widget)

    # ---------------- PAGE 3 (NEW USER) ----------------
    def show_new_user_page(self):
        widget = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(widget)
        layout.setAlignment(QtCore.Qt.AlignCenter)

        title = QtWidgets.QLabel("Register as New User")
        title.setProperty("role", "title")
        title.setFont(QtGui.QFont("Arial", 18, QtGui.QFont.Bold))
        layout.addWidget(title)

        name_label = QtWidgets.QLabel("Enter your name:")
        name_label.setProperty("role", "panel")
        name_entry = QtWidgets.QLineEdit()
        name_entry.setPlaceholderText("Your name (e.g., JOHN)")

        id_label = QtWidgets.QLabel("Enter your Unique ID (special chars & uppercase allowed):")
        id_label.setProperty("role", "panel")
        id_entry = QtWidgets.QLineEdit()
        id_entry.setPlaceholderText("Unique ID")
        id_entry.setEchoMode(QtWidgets.QLineEdit.Password)

        layout.addWidget(name_label, alignment=QtCore.Qt.AlignLeft)
        layout.addWidget(name_entry)
        layout.addWidget(id_label, alignment=QtCore.Qt.AlignLeft)
        layout.addWidget(id_entry)

        def save_and_continue():
            name = name_entry.text().strip()
            uid = id_entry.text().strip()
            if not name or not uid:
                QtWidgets.QMessageBox.warning(self, "Error", "Name and ID required!")
                return
            self.user_name = name
            self.user_id = uid
            append_user_id_if_new(uid)  # ensure row exists
            self.show_test_selection_page()

        nav = QtWidgets.QHBoxLayout()
        prev_btn = QtWidgets.QPushButton("Previous")
        prev_btn.clicked.connect(self.show_user_type_page)
        save_btn = QtWidgets.QPushButton("Save and Continue")
        save_btn.clicked.connect(save_and_continue)
        nav.addWidget(prev_btn)
        nav.addWidget(save_btn)

        layout.addLayout(nav)
        self._update_page(widget)

    # ---------------- PAGE 3 (EXISTING USER) ----------------
    def show_existing_user_page(self):
        widget = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(widget)
        layout.setAlignment(QtCore.Qt.AlignCenter)

        title = QtWidgets.QLabel("Existing User Login")
        title.setProperty("role", "title")
        title.setFont(QtGui.QFont("Arial", 18, QtGui.QFont.Bold))
        layout.addWidget(title)

        id_entry = QtWidgets.QLineEdit()
        id_entry.setPlaceholderText("Enter your Unique ID")
        id_entry.setEchoMode(QtWidgets.QLineEdit.Password)
        layout.addWidget(id_entry)

        def continue_login():
            uid = id_entry.text().strip()
            if not uid:
                QtWidgets.QMessageBox.warning(self, "Error", "ID required!")
                return
            self.user_id = uid
            append_user_id_if_new(uid)  # ensure row exists (no overwrite)
            self.show_test_selection_page()

        nav = QtWidgets.QHBoxLayout()
        prev_btn = QtWidgets.QPushButton("Previous")
        prev_btn.clicked.connect(self.show_user_type_page)
        next_btn = QtWidgets.QPushButton("Continue")
        next_btn.clicked.connect(continue_login)
        nav.addWidget(prev_btn)
        nav.addWidget(next_btn)

        layout.addLayout(nav)
        self._update_page(widget)

    # ---------------- PAGE 4 (EMBEDDED TEST) ----------------
    def show_test_selection_page(self):
        widget = QtWidgets.QWidget()
        self.page4_layout = QtWidgets.QVBoxLayout(widget)
        self.page4_layout.setAlignment(QtCore.Qt.AlignCenter)

        title = QtWidgets.QLabel("Choose a Test")
        title.setProperty("role", "title")
        title.setFont(QtGui.QFont("Arial", 18, QtGui.QFont.Bold))
        self.page4_layout.addWidget(title)

        # Mode buttons
        mode_layout = QtWidgets.QHBoxLayout()
        self.btn_baseline = QtWidgets.QPushButton("Test Baseline (Register your baseline)")
        self.btn_testing = QtWidgets.QPushButton("Testing (Random)")
        self.btn_baseline.clicked.connect(self.on_click_baseline)
        self.btn_testing.clicked.connect(self.on_click_testing)
        mode_layout.addWidget(self.btn_baseline)
        mode_layout.addWidget(self.btn_testing)
        self.page4_layout.addLayout(mode_layout)

        # Video preview area
        self.video_label = QtWidgets.QLabel()
        self.video_label.setAlignment(QtCore.Qt.AlignCenter)
        self.video_label.setMinimumSize(800, 450)
        self.video_label.setStyleSheet("background-color: #000;")
        self.page4_layout.addWidget(self.video_label)

        # Start/Stop
        ctrl = QtWidgets.QHBoxLayout()
        self.start_btn = QtWidgets.QPushButton("Start Test")
        self.stop_btn = QtWidgets.QPushButton("Stop")
        self.start_btn.clicked.connect(self.start_test)
        self.stop_btn.clicked.connect(self.stop_test)
        self.stop_btn.setEnabled(False)
        ctrl.addWidget(self.start_btn)
        ctrl.addWidget(self.stop_btn)
        self.page4_layout.addLayout(ctrl)

        # Live status
        self.status_label = QtWidgets.QLabel("Ready")
        self.page4_layout.addWidget(self.status_label)

        # Results box
        self.result_box = QtWidgets.QTextEdit()
        self.result_box.setReadOnly(True)
        self.result_box.setFixedHeight(170)
        self.page4_layout.addWidget(self.result_box)

        # Navigation
        nav = QtWidgets.QHBoxLayout()
        prev_btn = QtWidgets.QPushButton("Previous")
        prev_btn.clicked.connect(self.show_user_type_page)
        next_btn = QtWidgets.QPushButton("Next")
        next_btn.clicked.connect(lambda: QtWidgets.QMessageBox.information(self, "Info", "Implement next page logic"))
        nav.addWidget(prev_btn)
        nav.addWidget(next_btn)
        self.page4_layout.addLayout(nav)

        self.worker = None
        self.current_mode = None  # "baseline" or "testing"
        self._update_page(widget)

    def on_click_baseline(self):
        # If existing user has baseline, prompt
        if self.user_id:
            baseline = read_user_baseline(self.user_id)
            if baseline and any([baseline.get("baseline_rate", 0) > 0]):
                resp = QtWidgets.QMessageBox.question(
                    self, "Baseline Exists",
                    "You are an existing user. Do you want to run baseline again?\nChoose No to go for testing.",
                    QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No
                )
                if resp == QtWidgets.QMessageBox.No:
                    self.current_mode = "testing"
                    self.status_label.setText("Mode: Testing")
                    return
        self.current_mode = "baseline"
        self.status_label.setText("Mode: Baseline")

    def on_click_testing(self):
        self.current_mode = "testing"
        self.status_label.setText("Mode: Testing")

    def start_test(self):
        if not self.current_mode:
            QtWidgets.QMessageBox.warning(self, "Mode Required", "Please choose Baseline or Testing first.")
            return
        if not self.user_id:
            QtWidgets.QMessageBox.warning(self, "ID Required", "Please enter your ID in previous page.")
            return
        self.result_box.clear()
        self.worker = BlinkWorker(mode=self.current_mode, user_id=self.user_id)
        self.worker.frame_signal.connect(self.update_frame)
        self.worker.result_signal.connect(self.show_results)
        self.worker.status_signal.connect(self.update_status)
        self.worker.start()
        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.status_label.setText(f"Running ({'Baseline' if self.current_mode=='baseline' else 'Testing'})...")

    def stop_test(self):
        if self.worker:
            self.worker.stop()
            self.worker.wait()
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.status_label.setText("Stopped")

    @QtCore.pyqtSlot(QtGui.QImage)
    def update_frame(self, qimg):
        pix = QtGui.QPixmap.fromImage(qimg)
        self.video_label.setPixmap(pix.scaled(self.video_label.size(), QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation))

    @QtCore.pyqtSlot(str)
    def update_status(self, text):
        self.status_label.setText(text)

    @QtCore.pyqtSlot(dict)
    def show_results(self, res):
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        status_color = "#00AA00" if res["Status"]=="Alert" else "#D32F2F" if res["Status"]=="Drowsy" else "#1565C0"
        html = f"""
        <b>Results ({'Baseline' if self.current_mode=='baseline' else 'Testing'})</b><br>
        Total Blinks: {res['TotalBlinks']}<br>
        Blink Rate: {res['BlinkRate']:.2f}/min<br>
        Avg Blink Duration: {res['AvgBlinkDur']:.3f}s<br>
        PERCLOS: {res['PERCLOS']:.2f}%<br>
        Baseline Rate Used: {res['BaselineRateUsed']:.2f}/min<br>
        Status: <span style="color:{status_color};">{res['Status']}</span>
        """
        self.result_box.setHtml(html)
        self.status_label.setText("Completed")
        # If baseline mode, it’s already saved by worker

if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    window = WelcomeApp()
    window.show()
    sys.exit(app.exec_())
