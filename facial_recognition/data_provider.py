from os import path, listdir, mkdir
import pickle

import cv2
import numpy as np


def get_folder_names(directory):
    if not path.exists(directory):
        return []

    labels = listdir(directory)
    labels = list(filter(lambda f: '.' not in f, labels))
    labels = sorted(labels)

    return labels


def get_image_data_from_directory(directory):
    # type: (str) -> Tuple[np.ndarray, np.ndarray, Dict[int, str]]
    """Read the images from disk into an n*(w*h) matrix."""
    labels = get_folder_names(directory)

    matrices, ys, label_mapping = [], [], {}

    for i, label in enumerate(labels):
        label_mapping[i] = label

        label = label
        dir_path = path.join(directory, label)
        files = listdir(dir_path)

        ys.append(i * np.ones(len(files), dtype=int))

        im_matrix = None

        for j, im_file in enumerate(files):
            im_path = path.join(dir_path, im_file)
            image = cv2.imread(im_path, cv2.IMREAD_GRAYSCALE)

            if im_matrix is None:
                im_matrix = np.zeros((len(files), np.prod(image.shape)))

            im_matrix[j, :] = np.ravel(image)

        matrices.append(im_matrix)

    return np.vstack(matrices), np.hstack(ys), label_mapping


def load_model(fname):
    """Load the trained model from disk."""
    assert path.exists(fname), 'Model does not exist. Train a model first'

    with open(fname, 'rb') as file_handle:
        return pickle.load(file_handle)


def save_model(model, fname):
    """Save the trained model to disk."""
    model_directory = path.dirname(fname)
    if not path.exists(model_directory):
        mkdir(model_directory)

    with open(fname, 'wb') as file_handle:
        pickle.dump(model, file_handle)
