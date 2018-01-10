from typing import Optional, Dict

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from facial_recognition.model import PCA

sns.set('paper', 'darkgrid')


def explained_variance(pca):
    # type: (PCA) -> None
    x = list(range(len(pca.eigenvalues)))

    explained_var = np.cumsum(pca.eigenvalues)
    sum_eigs = explained_var[-1]
    explained_var /= explained_var[-1]

    plt.title('PCA explained variance')
    plt.plot(x, pca.eigenvalues / sum_eigs, label='Eigenvalues')
    plt.plot(x, explained_var, label='Explained variance')
    plt.legend()
    plt.show()


def scatter(X, y, mapping=None):
    # type: (np.ndarray, np.ndarray, Optional[Dict[int, str]]) -> None
    """Plot the data points according to their class label."""
    assert X.ndim == 2, 'X can not have more than 2 dimensions'
    assert X.shape[0] == y.shape[0], 'X and y do not match in dim 0'

    n_classes = np.max(y) + 1

    for i in range(n_classes):
        class_samples = X[y == i]
        label = mapping[i] if mapping else i
        plt.plot(class_samples[:, 0], class_samples[:, 1], 'o', label=label)

    plt.legend()
    plt.show()


def faces(vecs, start=0, cols=4, rows=4):
    for fig_idx, idx in enumerate(range(start, start + rows * cols)):
        ax = plt.subplot(rows, cols, fig_idx + 1)
        image_size = int(np.sqrt(vecs.shape[0]))
        image = np.reshape(vecs[:, idx], (image_size, image_size))

        ax.set_title('Face %d' % idx)
        ax.imshow(image, cmap='gray')
        ax.grid(False), ax.set_xticklabels([]), ax.set_yticklabels([])

    plt.tight_layout()
    plt.show()
