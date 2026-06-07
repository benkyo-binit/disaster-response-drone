# MainApp.py
import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QTabWidget
from LiveVideoWidget import LiveVideoWidget
from AIAnalyzerWidget import AIAnalyzerWidget
from MapWidget import MapWidget
from PyQt5.QtCore import QTimer


class MainDashboard(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Drone Ground Control Station")
        self.setGeometry(100, 100, 1000, 700)

        # Create the tab widget
        self.tabs = QTabWidget()
        self.setCentralWidget(self.tabs)
        self.init_tabs()

    def init_tabs(self):
        # Live video feed tab
        self.live_video_tab = LiveVideoWidget(source=0)  # or path to local camera / stream
        self.tabs.addTab(self.live_video_tab, "Live Video")

        self.map_tab = MapWidget()
        self.tabs.addTab(self.map_tab, "Map")

        # AI Classifier tab
        self.ai_tab = AIAnalyzerWidget(self.map_tab)
        self.tabs.addTab(self.ai_tab, "AI Classifier")

        # connect frame signal -> AI tab
        # LiveVideoWidget emits numpy frames via frame_ready signal
        self.live_video_tab.frame_ready.connect(self.ai_tab.on_frame)
        
        #print("Updating location:", lat, lon)
        #self.map_tab.update_location(lat, lon)

        '''
        def test_event():
            print("Sending test event")
            self.map_tab.add_event(27.698968, 85.297184, "Test Event")
        '''

        #QTimer.singleShot(3000, test_event)  # runs after 3 seconds
       
if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainDashboard()
    window.show()
    sys.exit(app.exec_())
