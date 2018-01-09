from os import path, listdir

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
