# AIAnalyzerWidget.py
from PyQt5.QtWidgets import QWidget, QLabel, QVBoxLayout, QPushButton, QFileDialog, QHBoxLayout, QApplication
from PyQt5.QtCore import pyqtSignal, pyqtSlot, QObject, QThread
from PyQt5.QtGui import QImage, QPixmap
import cv2
import numpy as np
from ai_adapter import predict
from PyQt5.QtWidgets import QTextEdit
from event_logger import log_event
from MapWidget import MapWidget


# Worker object to run inference in a background thread
class InferenceWorker(QObject):
    result_ready = pyqtSignal(str, float)   # label, confidence
    finished = pyqtSignal()

    @pyqtSlot(object)
    def process_frame(self, frame):
        try:
            label, conf = predict(frame)
            # ensure confidence is float
            conf = float(conf)
            self.result_ready.emit(str(label), conf)
        except Exception as e:
            self.result_ready.emit("error", 0.0)

class AIAnalyzerWidget(QWidget):
    # optional: allow external widgets to connect in
    def __init__(self, map_widget, parent=None):

        super().__init__(parent)
        self.video_label = QLabel("No frame")
        self.pred_label = QLabel("Prediction: -")
        self.pred_label.setMaximumHeight(30)
        self.conf_label = QLabel("Confidence: -")
        self.conf_label.setMaximumHeight(30)

        self.toggle_btn = QPushButton("Enable Inference")
        self.load_video_btn = QPushButton("Load Demo Video")

        self.log_view = QTextEdit()
        self.log_view.setReadOnly(True)
        self.log_view.setPlaceholderText("Event log will appear here...")
        self.log_view.setMaximumHeight(120)
        
        self.map_widget = map_widget

        btn_layout = QHBoxLayout()
        btn_layout.addWidget(self.toggle_btn)
        btn_layout.addWidget(self.load_video_btn)

        layout = QVBoxLayout()
        layout.addWidget(self.video_label)
        layout.addLayout(btn_layout)
        layout.addWidget(self.pred_label)
        layout.addWidget(self.conf_label)
        layout.addWidget(self.log_view)
        self.setLayout(layout)

        # background inference thread setup
        self.worker_thread = QThread()
        self.worker = InferenceWorker()
        self.worker.moveToThread(self.worker_thread)
        self.worker_thread.start()
        self.worker.result_ready.connect(self._on_result)

        self.do_inference = False
        self.toggle_btn.clicked.connect(self._toggle_inference)
        self.load_video_btn.clicked.connect(self._load_demo_video)

        self.demo_cap = None


    def _toggle_inference(self):
        self.do_inference = not self.do_inference
        self.toggle_btn.setText("Disable Inference" if self.do_inference else "Enable Inference")

    @pyqtSlot(object)
    def on_frame(self, frame: np.ndarray):
        """Slot: receives a BGR numpy frame from LiveVideoWidget"""
        # display frame
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        bytes_per_line = ch * w
        qimg = QImage(rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)
        self.video_label.setPixmap(QPixmap.fromImage(qimg).scaled(
            self.video_label.size(), aspectRatioMode=1))

        # if demo video is loaded and playing, we could override frame with demo frame
        if self.do_inference:
            # send frame to worker (non-blocking signal)
            QMeta = getattr(self, 'worker').metaObject  # no-op just to avoid linter
            # Use Qt signal/slot invocation:
            self.worker.process_frame(frame)  # safe because slot is in worker thread (PyQt will queue it)

    @pyqtSlot(str, float)
    def _on_result(self, label, confidence):
        self.pred_label.setText(f"Prediction: {label}")
        self.conf_label.setText(f"Confidence: {confidence:.2f}")
        

        # Threshold-based event logging
        if label != "normal" and confidence >= 0.70:
            log_event(label, confidence, source="Video Stream")

            msg = f"[ALERT] {label.upper()} detected ({confidence:.2f})"
            self.log_view.append(msg)

            lat = 27.698968 
            lon = 85.297184

            event_text = f"{label} ({confidence:.2f})"
            self.map_widget.add_event(lat, lon, event_text)


    def _load_demo_video(self):
        
        path, _ = QFileDialog.getOpenFileName(self, "Open Demo Video", "", "Video Files (*.mp4 *.avi)")
        if not path:
            return
        # open capture and start a small loop feeding frames into on_frame
        if self.demo_cap:
            self.demo_cap.release()
            self.demo_cap = None
        self.demo_cap = cv2.VideoCapture(path)
        # play it in a non-blocking way: start a QThread to read frames and emit to on_frame


    def _load_demo_video(self):
        try:
            path, _ = QFileDialog.getOpenFileName(
                self, "Open Demo Video", "", "Video Files (*.mp4 *.avi)"
            )
            if not path:
                return
            if self.demo_cap:
                self.demo_cap.release()
                self.demo_cap = None
            self.demo_cap = cv2.VideoCapture(path)
        except Exception as e:
            print("Demo video load error:", e)

        def play():
            while self.demo_cap.isOpened():
                ret, frame = self.demo_cap.read()
                if not ret:
                    break
                # call on_frame via Qt event loop
                QApplication.instance().postEvent(self, _FrameEvent(frame))
                QThread.msleep(100)  # adjust for playback speed
            self.demo_cap.release()
        # simple approach: start a thread to play
        t = QThread()
        def runner():
            play()
            t.quit()
        t.run = runner
        t.start()

# Supporting custom event to post frames if desired
from PyQt5.QtCore import QEvent
class _FrameEvent(QEvent):
    EVENT_TYPE = QEvent.Type(QEvent.registerEventType())
    def __init__(self, frame):
        super().__init__(self.EVENT_TYPE)
        self.frame = frame


