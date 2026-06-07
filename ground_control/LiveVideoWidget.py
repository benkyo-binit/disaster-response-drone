# LiveVideoWidget.py (only relevant parts shown)
from PyQt5.QtWidgets import QWidget, QLabel, QVBoxLayout, QPushButton
from PyQt5.QtCore import pyqtSignal, QThread
from PyQt5.QtGui import QImage, QPixmap
import cv2
import numpy as np

class VideoCaptureThread(QThread):
    frame_ready = pyqtSignal(object)  # emits numpy.ndarray

    def __init__(self, source=0, parent=None):
        super().__init__(parent)
        source = "footage/flood.mp4"
        #source = "footage/fire.mp4"
        #source = "footage/collapsed_building.mp4"
        #source = "footage/parking.mp4"
        self.source = source
        self.running = False

    def run(self):
        cap = cv2.VideoCapture(self.source)
        self.running = True
        while self.running:
            ret, frame = cap.read()
            if not ret:
                break
            # emit the BGR numpy frame
            self.frame_ready.emit(frame)
            self.msleep(30)  # ~33fps throttle (adjust as needed)
        cap.release()

    def stop(self):
        self.running = False
        self.wait()

class LiveVideoWidget(QWidget):
    frame_ready = pyqtSignal(object)  # re-exposed for other widgets

    def __init__(self, source=0, parent=None):
        super().__init__(parent)
        self.source = source
        self.image_label = QLabel("No video")
        self.start_btn = QPushButton("Start")
        self.stop_btn = QPushButton("Stop")
        layout = QVBoxLayout()
        layout.addWidget(self.image_label)
        layout.addWidget(self.start_btn)
        layout.addWidget(self.stop_btn)
        self.setLayout(layout)

        self.thread = VideoCaptureThread(source=self.source)
        self.thread.frame_ready.connect(self._on_frame_from_thread)
        self.start_btn.clicked.connect(self.thread.start)
        self.stop_btn.clicked.connect(self.thread.stop)

    def _on_frame_from_thread(self, frame: np.ndarray):
        # display in the widget
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        bytes_per_line = ch * w
        qimg = QImage(rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)
        self.image_label.setPixmap(QPixmap.fromImage(qimg).scaled(
            self.image_label.size(), aspectRatioMode=1))
        # forward the raw frame to any listeners (AI tab)
        self.frame_ready.emit(frame)
