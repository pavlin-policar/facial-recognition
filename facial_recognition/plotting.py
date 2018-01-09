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
