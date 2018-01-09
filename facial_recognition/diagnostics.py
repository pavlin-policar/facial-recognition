import fire

from facial_recognition import data_provider, plotting
from facial_recognition.model import PCALDAClassifier, PCALDA, PCA


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


if __name__ == '__main__':
    fire.Fire()
