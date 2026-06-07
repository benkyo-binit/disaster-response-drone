import sys
from PyQt5.QtWidgets import QApplication
from MapWidget import MapWidget
from PyQt5.QtCore import QTimer

app = QApplication(sys.argv)
window = MapWidget()
window.show()

# Simulate movement
lat = 27.7172
lon = 85.3240

def move():
    global lat, lon
    lat += 0.0001
    lon += 0.0001
    window.update_location(lat, lon)

timer = QTimer()
timer.timeout.connect(move)
timer.start(1000)

sys.exit(app.exec_())