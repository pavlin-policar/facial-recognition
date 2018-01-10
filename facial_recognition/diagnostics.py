import fire
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold

from facial_recognition import data_provider, plotting
from facial_recognition.model import PCALDA, PCA, PCALDAClassifier
from facial_recognition.recognize import MainApp


def show_scatter(model_fname, images_dir):
    model = data_provider.load_model(model_fname)
    X, y, mapping = data_provider.get_image_data_from_directory(images_dir)

    projection = model.pca_lda.project(X)
    plotting.scatter(projection, y, mapping)


def train_pca_lda(images_dir):
    X, y, mapping = data_provider.get_image_data_from_directory(images_dir)
    projector = PCALDA(n_components=2, pca_components=500).fit(X, y)

    projection = projector.project(X)
    plotting.scatter(projection, y, mapping)


def train_pca(images_dir):
    X, y, mapping = data_provider.get_image_data_from_directory(images_dir)
    projector = PCA(n_components=2).fit(X, y)

    projection = projector.project(X)
    plotting.scatter(projection, y, mapping)
    plotting.explained_variance(projector)


def show_pca_eigv(model_fname, start=0, rows=3, cols=4):
    model = data_provider.load_model(model_fname)
    eigvecs = model.pca_lda.pca.subspace_basis
    plotting.faces(eigvecs, start=start, rows=rows, cols=cols)


def cross_validate(images_dir, k_splits=5):
    X, y, mapping = data_provider.get_image_data_from_directory(images_dir)
    clf = PCALDAClassifier(
        n_components=2, pca_components=200, metric='euclidean',
    ).fit(X, y)

    scores = []
    kfold = StratifiedKFold(n_splits=k_splits, shuffle=True)
    for train_idx, val_idx in kfold.split(X, y):
        X_train, y_train = X[train_idx], y[train_idx]
        X_val, y_val = X[val_idx], y[val_idx]

        clf.fit(X_train, y_train)
        predictions = clf.predict(X_val)

        accuracy = accuracy_score(y_val, predictions)
        scores.append(accuracy)

    print('CA: %.4f' % np.mean(scores))


if __name__ == '__main__':
    fire.Fire()
