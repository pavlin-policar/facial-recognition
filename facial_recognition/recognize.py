import sys
from contextlib import contextmanager
from os import path, mkdir, listdir
from typing import Optional

import cv2
import numpy as np
from PyQt5.QtCore import QSize, QTimer, QStringListModel, Qt, \
    QItemSelectionModel
from PyQt5.QtGui import QImage, QPixmap, QKeySequence
from PyQt5.QtWidgets import QWidget, QLabel, QApplication, QHBoxLayout, \
    QShortcut, QVBoxLayout, QListView, QPushButton, QLineEdit, QGroupBox, \
    QStyledItemDelegate

from facial_recognition import data_provider
from facial_recognition.model import PCALDAClassifier


class NoFacesError(Exception):
    pass


class MultipleFacesError(Exception):
    pass


class CapitalizeDelegate(QStyledItemDelegate):
    def displayText(self, value, locale):
        string = super().displayText(value, locale)
        return string.capitalize()


class MainApp(QWidget):
    STRANGER_DANGER = 350
    IMAGE_SIZE = (100, 100)

    stranger_color = (179, 20, 20)
    recognized_color = (59, 235, 62)

    def __init__(self, fps=30, parent=None):
        # type: (int, Optional[QWidget]) -> None
        super().__init__(parent=parent)

        self.pkg_path = path.dirname(path.dirname(path.abspath(__file__)))
        self.training_data_dir = path.join(self.pkg_path, 'train')
        self.models_dir = path.join(self.pkg_path, 'models')
        self.model_fname = 'fisherfaces.p'

        try:
            self.model = data_provider.load_model(
                path.join(self.models_dir, self.model_fname))
        except AssertionError:
            self.model = None

        self.existing_labels = QStringListModel(self.get_existing_labels())

        self.fps = fps
        self.video_size = QSize(640, 480)

        self.gray_image = None
        self.detected_faces = []

        # Setup the UI
        self.main_layout = QHBoxLayout()
        self.setLayout(self.main_layout)

        self.control_layout = QVBoxLayout()
        self.control_layout.setSpacing(8)
        self.main_layout.addItem(self.control_layout)

        # Setup the existing label view
        self.labels_view = QListView(parent=self)
        self.labels_view.setModel(self.existing_labels)
        self.labels_view.setSelectionMode(QListView.SingleSelection)
        self.labels_view.setItemDelegate(CapitalizeDelegate(self))
        self.control_layout.addWidget(self.labels_view)

        self.new_label_txt = QLineEdit(self)
        self.new_label_txt.returnPressed.connect(self.add_new_label)
        self.new_label_txt.returnPressed.connect(self.new_label_txt.clear)
        self.control_layout.addWidget(self.new_label_txt)

        self.add_button = QPushButton('Add Label', self)
        self.add_button.clicked.connect(self.add_new_label)
        self.control_layout.addWidget(self.add_button)

        # Setup the training area
        train_box = QGroupBox('Train', self)
        train_box_layout = QVBoxLayout()
        train_box.setLayout(train_box_layout)
        self.control_layout.addWidget(train_box)
        self.train_btn = QPushButton('Train', self)
        self.train_btn.clicked.connect(self.train)
        train_box_layout.addWidget(self.train_btn)

        self.control_layout.addStretch(0)

        # Add take picture shortcut
        self.take_picture_btn = QPushButton('Take picture', self)
        self.take_picture_btn.clicked.connect(self.take_picture)
        self.control_layout.addWidget(self.take_picture_btn)
        shortcut = QShortcut(QKeySequence('Space'), self, self.take_picture)
        shortcut.setWhatsThis('Take picture and add to training data.')

        # Add quit shortcut
        shortcut = QShortcut(QKeySequence('Esc'), self, self.close)
        shortcut.setWhatsThis('Quit')

        # Setup the main camera area
        self.image_label = QLabel(self)
        self.image_label.setFixedSize(self.video_size)
        self.main_layout.addWidget(self.image_label)

        # Setup the camera
        self.capture = cv2.VideoCapture(0)
        self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, self.video_size.width())
        self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self.video_size.height())

        self.timer = QTimer()
        self.timer.timeout.connect(self.display_video_stream)
        self.timer.start(int(1000 / self.fps))

    def classify_face(self, image):
        if self.model is None:
            return

        label_idx, distances = self.model.predict(image.ravel(), True)
        label_idx, distance = label_idx[0], distances[0][label_idx]

        labels = self.existing_labels.stringList()
        return labels[label_idx], distance

    def get_training_data(self):
        """Read the images from disk into an n*(w*h) matrix."""
        return data_provider.get_image_data_from_directory(
            self.training_data_dir)

    def train(self):
        X, y, mapping = self.get_training_data()
        # Inspect scree plot to determine appropriate number of PCA components
        classifier = PCALDAClassifier(
            n_components=2, pca_components=200, metric='euclidean',
        ).fit(X, y)

        # Replace the existing running model
        self.model = classifier

        # Save the classifier to file
        data_provider.save_model(
            classifier, path.join(self.models_dir, self.model_fname))

    def add_new_label(self):
        new_label = self.new_label_txt.text()
        new_label = new_label.lower()

        # Prevent empty entries
        if len(new_label) < 3:
            return

        string_list = self.existing_labels.stringList()

        if new_label not in string_list:
            string_list.append(new_label)
            string_list.sort()
            self.existing_labels.setStringList(string_list)

            # Automatically select the added label
            selection_model = self.labels_view.selectionModel()
            index = self.existing_labels.index(string_list.index(new_label))
            selection_model.setCurrentIndex(index, QItemSelectionModel.Select)

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
            # Label the detected face as per the model
            face = gray[y:y + h, x:x + w]
            face = cv2.resize(face, self.IMAGE_SIZE)

            result = self.classify_face(face)
            # If a model is loaded, we can predict
            if result:
                predicted, distance = self.classify_face(face)

                if distance > self.STRANGER_DANGER:
                    predicted = 'Stranger danger!'
                    color = self.stranger_color
                else:
                    predicted = predicted.capitalize()
                    color = self.recognized_color

                # Draw a rectangle around the detected face
                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                text = '%s (%.1f)' % (predicted, distance)
                cv2.putText(frame, text, (x, y + h + 15),
                            cv2.FONT_HERSHEY_TRIPLEX, 0.5, color)
            else:
                # Draw a rectangle around the detected face
                cv2.rectangle(frame, (x, y), (x + w, y + h),
                              self.stranger_color, 2)
                cv2.putText(frame, 'Stranger danger!', (x, y + h + 15),
                            cv2.FONT_HERSHEY_TRIPLEX, 0.5, self.stranger_color)

        # Display the image in the image area
        image = QImage(frame, frame.shape[1], frame.shape[0],
                       frame.strides[0], QImage.Format_RGB888)
        self.image_label.setPixmap(QPixmap.fromImage(image))

    @contextmanager
    def stop_camera_feed(self):
        """Temporarly stop the feed and face detection."""
        try:
            self.timer.stop()
            yield
        finally:
            self.timer.start(int(1000 / self.fps))

    def take_picture(self):
        # Notify the user there were no faces detected
        if self.detected_faces is None or len(self.detected_faces) < 1:
            return
            raise NoFacesError()

        if len(self.detected_faces) > 1:
            return
            raise MultipleFacesError()

        with self.stop_camera_feed():
            x, y, w, h = self.detected_faces[0]

            face = self.gray_image[y:y + h, x:x + w]
            face = cv2.resize(face, self.IMAGE_SIZE)
            denoised_image = cv2.fastNlMeansDenoising(face)

            if not self.selected_label:
                return

            self.save_image(denoised_image, self.selected_label)

    @property
    def selected_label(self):
        index = self.labels_view.selectedIndexes()
        if len(index) < 1:
            return None

        label = self.existing_labels.data(index[0], Qt.DisplayRole)

        return label

    def get_existing_labels(self):
        """Get a list of the currently existing labels"""
        return data_provider.get_folder_names(self.training_data_dir)

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
