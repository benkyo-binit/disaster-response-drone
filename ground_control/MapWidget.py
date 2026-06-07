from PyQt5.QtWidgets import QWidget, QVBoxLayout
from PyQt5.QtWebEngineWidgets import QWebEngineView
from PyQt5.QtCore import QUrl

class MapWidget(QWidget):
    def __init__(self):
        super().__init__()

        self.loaded = False 

        layout = QVBoxLayout()
        self.browser = QWebEngineView()

        # Load local HTML file
        self.browser.load(QUrl.fromLocalFile(r"C:\Users\Binit\Documents\Major Project\code\map.html"))

        self.browser.loadFinished.connect(self.on_load)

        layout.addWidget(self.browser)
        self.setLayout(layout)

    def on_load(self):
        print("Map loaded")
        self.loaded = True

    def update_location(self, lat, lon):
        if not self.loaded:
            print("Map not ready yet")
            return
        
        js_code = f"updateLocation({lat}, {lon});"
        self.browser.page().runJavaScript(js_code)

        print("Updating location:", lat, lon)
        self.map_tab.update_location(lat, lon)

    def add_event(self, lat, lon, text):
        if not self.loaded:
            return

        text = text.replace("'", "")  # simple fix
        js_code = f"addEvent({lat}, {lon}, '{text}');"
        self.browser.page().runJavaScript(js_code)