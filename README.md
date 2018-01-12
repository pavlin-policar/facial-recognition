# Fisherfaces facial recognition

A simple implementation of Fisherfaces for realtime facial recognition.

## Installation
Clone the repository with `git clone https://github.com/pavlin-policar/facial-recognition.git` and install the requirements listed below. The recommended installation method is `conda`.

### Requirements
- `opencv` (for webcam interaction and face detection)
- `pyqt` (gui)
- `numpy` (PCA, LDA)
- `scikit-learn` (model cross-validation)
- `fire` - pip only (convenient diagnostics)

## Usage
When you first start the program, the face detector should find and mark your face in the display area. In order for the fisherfaces to recognize anybody, it has to be trained first.

Training is made very easy. Firstly, add your name to the input box above the button `Add label`. After adding, your name should appear in the list above. Next, the classifier requires some training data. To add images, selet the desired name from the list (so it's highlighted in blue) and press the `Take picture` button or simply press Space. Make sure the face detector only detects the single face corresponding to the selected label. Lastly, we must train the model. We do this by pressing the `Train` button. The model will be reloaded automatically.

## Known bugs
- When a new label is added and does is not alphabetically last i.e. appears in the middle of the labels, the display will use the label list instead of the labels that the model was trained with, so the label on the display area will be wrong. Retrain the model to fix this with the new training data.
