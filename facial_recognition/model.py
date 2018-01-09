import numpy as np


class Projection:
    def __init__(self, n_components=2):
        self.n_components = n_components

    def fit(self, X, y):
        """Fit the projection onto the training data."""

    def project(self, X):
        """Project the new data using the fitted projection matrices."""

    def reconstruct(self, X):
        """Reconstruct the projected data back into the original space."""


class PCA(Projection):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.X_mean = None
        self.U = None
        self.P = None

    def fit(self, X, y=None):
        assert X.ndim == 2, 'X can only be a 2-d matrix'

        # Center the data
        self.X_mean = np.mean(X, axis=0)
        X -= self.X_mean

        # If the d >> n then we should use dual PCA for efficiency
        use_dual_pca = X.shape[1] > X.shape[0]

        if use_dual_pca:
            X = X.T

        # Estimate the covariance matrix
        C = np.dot(X.T, X) / (X.shape[0] - 1)

        U, S, V = np.linalg.svd(C)

        if use_dual_pca:
            U = X.dot(U).dot(np.diag(1 / np.sqrt(S * (X.shape[0] - 1))))

        self.U = U
        self.P = U[:, :self.n_components]

        return self

    def project(self, X):
        assert self.X_mean is not None, \
            'You must fit PCA before you can project'

        X -= self.X_mean
        return np.dot(X, self.P)

    def reconstruct(self, X):
        return np.dot(X, self.P.T) + self.X_mean
