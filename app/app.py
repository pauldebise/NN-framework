import sys
import numpy as np
from PIL import Image, ImageDraw

from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget
from PyQt5.QtGui import QPainter, QPen
from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5 import uic

from core.network import Network


class DrawCanvas(QWidget):
    drawn = pyqtSignal()

    def __init__(self, width=280, height=280, brush_size=12):
        super().__init__()
        self.setFixedSize(width, height)
        self.brush_size = brush_size

        self.pil_image = Image.new("L", (width, height), 0)
        self.draw = ImageDraw.Draw(self.pil_image)

        self.last_point = None
        self.points_to_draw = []

    def clear(self):
        self.pil_image = Image.new("L", (self.width(), self.height()), 0)
        self.draw = ImageDraw.Draw(self.pil_image)
        self.points_to_draw.clear()
        self.update()

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.last_point = event.pos()

    def mouseMoveEvent(self, event):
        if (event.buttons() & Qt.LeftButton) and self.last_point:
            current_point = event.pos()

            x1, y1 = self.last_point.x(), self.last_point.y()
            x2, y2 = current_point.x(), current_point.y()
            r = self.brush_size
            self.draw.ellipse([x2 - r, y2 - r, x2 + r, y2 + r], fill=255)
            self.draw.line([x1, y1, x2, y2], fill=255, width=self.brush_size * 2)

            self.points_to_draw.append((self.last_point, current_point))
            self.last_point = current_point
            self.update()

            if current_point.x() % 3 == 0:
                self.drawn.emit()

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.last_point = None
            self.drawn.emit()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.fillRect(self.rect(), Qt.black)
        pen = QPen(Qt.white, self.brush_size * 2, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin)
        painter.setPen(pen)
        for p1, p2 in self.points_to_draw:
            painter.drawLine(p1, p2)


class App(QMainWindow):

    def __init__(self, model_path="best_model.json"):
        super().__init__()

        uic.loadUi(r"app/interface.ui", self)

        self.canvas = DrawCanvas()
        self.layout_canvas.insertWidget(0, self.canvas)

        self.bars = [
            self.bar_0, self.bar_1, self.bar_2, self.bar_3, self.bar_4,
            self.bar_5, self.bar_6, self.bar_7, self.bar_8, self.bar_9
        ]

        self.btn_clear.clicked.connect(self.clear_ui)
        self.canvas.drawn.connect(self.predict_digit)

        self.model = Network()
        try:
            self.model.load(model_path)
        except Exception as e:
            self.label_prediction.setText("Error")

    def clear_ui(self):
        self.canvas.clear()
        self.label_prediction.setText("?")
        for bar in self.bars:
            bar.setValue(0)

    def predict_digit(self):
        if not self.model.layers:
            return

        img = self.canvas.pil_image
        bbox = img.getbbox()

        if bbox is None:
            return

        cropped = img.crop(bbox)
        w, h = cropped.size
        ratio = 20.0 / max(w, h)
        new_w, new_h = int(w * ratio), int(h * ratio)
        resized = cropped.resize((max(1, new_w), max(1, new_h)), Image.Resampling.LANCZOS)

        arr = np.array(resized)
        total_mass = np.sum(arr)
        if total_mass == 0: return

        y_coords, x_coords = np.indices(arr.shape)
        cy = np.sum(y_coords * arr) / total_mass
        cx = np.sum(x_coords * arr) / total_mass

        paste_x = int(round(14.0 - cx))
        paste_y = int(round(14.0 - cy))

        final_img = Image.new("L", (28, 28), 0)
        final_img.paste(resized, (paste_x, paste_y))

        x_input = np.array(final_img).reshape(1, 784).astype(np.float32) / 255.0

        try:
            predictions = self.model.predict(x_input)
            probs = predictions[0] * 100
            predicted_class = np.argmax(probs)
            self.label_prediction.setText(str(predicted_class))

            for i in range(10):
                val = int(probs[i])
                self.bars[i].setValue(val)

        except Exception as e:
            pass


if __name__ == "__main__":
    app = QApplication(sys.argv)
    model_file = sys.argv[1] if len(sys.argv) > 1 else "../core/model_test.json"

    window = App(model_path=model_file)
    window.show()
    sys.exit(app.exec_())