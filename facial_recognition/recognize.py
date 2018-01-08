import sys
from os import path, mkdir, listdir
from typing import Optional

import cv2
import numpy as np
from PyQt5.QtCore import QSize, QTimer, QStringListModel
from PyQt5.QtGui import QImage, QPixmap, QKeySequence
from PyQt5.QtWidgets import QWidget, QLabel, QApplication, QHBoxLayout, \
    QShortcut, QDialog, QVBoxLayout, QListView, QTextEdit, QPushButton


class NoFacesError(Exception):
    pass


class SpecifyImageLabelDialog(QDialog):
    def __init__(self, image, existing_labels_model, parent=None):
        # type: (np.ndarray, QStringListModel, Optional[QWidget]) -> None
        super().__init__(parent=parent)
        self.label = None

        self.main_layout = QHBoxLayout()
        self.setLayout(self.main_layout)

        self.image = QImage(image, image.shape[1], image.shape[0],
                            image.strides[0], QImage.Format_RGB888)
        self.image_label = QLabel(self)
        self.image_label.setFixedSize(QSize(*image.shape))
        self.image_label.setPixmap(QPixmap.fromImage(self.image))
        self.main_layout.addWidget(self.image_label)

        self.control_layout = QVBoxLayout()
        # Setup existing label view
        self.labels_view = QListView(parent=self)
        self.labels_view.setModel(existing_labels_model)
        self.control_layout.addWidget(self.labels_view)
        self.main_layout.addItem(self.control_layout)

        selection_model = self.labels_view.selectionModel()
        selection_model.selectionChanged.connect(self._label_changed)

        # Setup input for new labels
        self.new_label = QTextEdit(self)
        self.control_layout.addWidget(self.new_label)

        self.add_button = QPushButton('Add', self)
        self.control_layout.addWidget(self.add_button)

    def _label_changed(self, param):
        print(param)


class MainApp(QWidget):
    def __init__(self, fps=30, parent=None):
        # type: (int, Optional[QWidget]) -> None
        super().__init__(parent=parent)

        self.pkg_path = path.dirname(path.dirname(path.abspath(__file__)))
        self.training_data_dir = path.join(self.pkg_path, 'train')
        self.existing_labels = QStringListModel(self.get_existing_labels())

        self.fps = fps
        self.video_size = QSize(640, 480)

        self.gray_image = None
        self.detected_faces = []

        # Setup the UI
        self.image_label = QLabel(self)
        self.image_label.setFixedSize(self.video_size)

        self.main_layout = QHBoxLayout()
        self.main_layout.addWidget(self.image_label)
        self.setLayout(self.main_layout)

        # Add take picture shortcut
        shortcut = QShortcut(QKeySequence('Space'), self, self.take_picture)
        shortcut.setWhatsThis('Take picture to train with.')

        # Add quit shortcut
        shortcut = QShortcut(QKeySequence('Esc'), self, self.close)
        shortcut.setWhatsThis('Quit')

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

        # Use the Viola-Jones face detector to detect faces to classify
        face_cascade = cv2.CascadeClassifier(path.join(
            self.pkg_path, 'resources', 'haarcascade_frontalface_default.xml'))
        self.gray_image = gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        self.detected_faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        for x, y, w, h in self.detected_faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 1)
            cv2.putText(frame, 'Pavlin 74.5%', (x, y + h + 15),
                        cv2.FONT_HERSHEY_TRIPLEX, 0.5, (0, 255, 0))

        # Display the image in the image area
        image = QImage(frame, frame.shape[1], frame.shape[0],
                       frame.strides[0], QImage.Format_RGB888)
        self.image_label.setPixmap(QPixmap.fromImage(image))

    def take_picture(self):
        # Notify the user there were no faces detected
        if self.detected_faces is None:
            raise NoFacesError()

        self.timer.stop()
        for x, y, w, h in self.detected_faces:
            face = self.gray_image[y:y + h, x:x + w]
            face = cv2.resize(face, (100, 100))
            denoised_image = cv2.fastNlMeansDenoising(face)

            dialog = SpecifyImageLabelDialog(denoised_image, self.existing_labels)
            if dialog.exec():
                self.save_image(denoised_image, 'pavlin')

        self.timer.start(int(1000 / self.fps))

    def get_existing_labels(self):
        """Get a list of the currently existing labels"""
        if not path.exists(self.training_data_dir):
            return []

        labels = listdir(self.training_data_dir)
        labels = list(filter(lambda f: '.' not in f, labels))
        return labels

    def save_image(self, image: np.ndarray, label: str) -> None:
        """Save an image to disk in the appropriate directory."""
        if not path.exists(self.training_data_dir):
            mkdir(self.training_data_dir)

        label_path = path.join(self.training_data_dir, label)
        if not path.exists(label_path):
            mkdir(label_path)

        existing_files = listdir(label_path)
        existing_files = map(lambda p: path.splitext(p)[0], existing_files)
        existing_files = list(map(int, existing_files))
        last_fname = sorted(existing_files)[-1] if len(existing_files) else 0

        fname = path.join(label_path, '%03d.png' % (last_fname + 1))
        cv2.imwrite(fname, image)
        
        
if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = MainApp()
    win.show()
    sys.exit(app.exec_())
