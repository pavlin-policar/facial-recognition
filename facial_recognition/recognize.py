import sys

import cv2
from PyQt5.QtCore import QSize, QTimer
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import QWidget, QLabel, QPushButton, QVBoxLayout, \
    QApplication


class MainApp(QWidget):
    def __init__(self, parent=None, fps=30):
        super().__init__(parent=parent)

        self.fps = fps

        self.video_size = QSize(640, 480)

        # Setup the UI
        self.image_label = QLabel(self)
        self.image_label.setFixedSize(self.video_size)

        self.quit_button = QPushButton('Quit')
        self.quit_button.clicked.connect(self.close)

        self.main_layout = QVBoxLayout()
        self.main_layout.addWidget(self.image_label)
        self.main_layout.addWidget(self.quit_button)

        self.setLayout(self.main_layout)

        # Setup the camera
        self.capture = cv2.VideoCapture(0)
        self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, self.video_size.width())
        self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self.video_size.height())

        self.timer = QTimer()
        self.timer.timeout.connect(self.display_video_stream)
        self.timer.start(int(1000 / self.fps))

    def display_video_stream(self):
        """Read frame from camera and repaint QLabel widget."""
        _, frame = self.capture.read()
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.flip(frame, 1)

        # Display the image in the image area
        image = QImage(frame, frame.shape[1], frame.shape[0],
                       frame.strides[0], QImage.Format_RGB888)
        self.image_label.setPixmap(QPixmap.fromImage(image))


if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = MainApp()
    win.show()
    sys.exit(app.exec_())
