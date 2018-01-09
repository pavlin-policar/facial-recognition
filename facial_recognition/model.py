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
        self.subspace_basis = None
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

        self.subspace_basis = U
        self.P = U[:, :self.n_components]

        return self

    def project(self, X):
        assert self.P is not None, \
            'You must fit PCA before you can project'

        X -= self.X_mean
        return np.dot(X, self.P)

    def reconstruct(self, X):
        return np.dot(X, self.P.T) + self.X_mean


class LDA(Projection):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.subspace_basis = None
        self.P = None
        self.class_means = None

    def fit(self, X, y):
        assert X.shape[0] == y.shape[0], 'X and y dimensions do not match.'

        n_classes = np.max(y) + 1
        n_samples, n_features = X.shape

        # Compute the class means
        class_means = np.zeros((n_classes, n_features))
        for idx in range(n_classes):
            class_means[idx, :] = np.mean(X[y == idx], axis=0)

        mean = np.mean(class_means, axis=0)

        Sw = Sb = 0
        for i in range(n_classes):
            for j in X[y == i]:
                val = np.atleast_2d(j - class_means[i])
                Sw += np.dot(val.T, val)

            val = np.atleast_2d(class_means[i] - mean)
            Sb += n_samples * np.dot(val.T, val)

        eigvals, eigvecs = np.linalg.eigh(np.linalg.inv(Sw) * Sb)

        self.subspace_basis = eigvecs
        self.P = eigvecs[:, :self.n_components]

        self.class_means = np.dot(class_means, self.P)

        return self

    def project(self, X):
        assert self.P is not None, \
            'You must fit LDA before you can project'

        return np.dot(X, self.P)

    def reconstruct(self, X):
        np.dot(X, self.P.T)
