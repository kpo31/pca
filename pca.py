import matplotlib
import scipy
from scipy.linalg import eigh
import numpy as np
import matplotlib.pyplot as plt


def load_and_center_dataset(filename):
    file = np.load(filename)
    file = np.reshape(file, (2000, 784))
    x = np.array(file)
    np.mean(x, axis=0)
    x = x - np.mean(x, axis=0)
    return x


def get_covariance(dataset):
    n = len(dataset)
    x = np.array(dataset)
    y = np.transpose(x)
    w = np.dot(y, x)
    return w / (n - 1)


def get_eig(S, m):
    eigvals, eig_vectors = scipy.linalg.eigh(S, eigvals=(len(S) - m, len(S) - 1))
    return np.diag(np.flipud(eigvals)), np.fliplr(eig_vectors)


def get_eig_perc(S, perc):
    eigvals, eig_vectors = scipy.linalg.eigh(S)
    eigvals = np.flipud(eigvals)
    eig_vectors = np.fliplr(eig_vectors)
    eig_vectors_return = np.zeros(np.shape(eig_vectors))
    eigvals1 = np.array([])
    for i in range(len(eigvals)):
        if eigvals[i] / np.sum(eigvals) > perc:
            eigvals1 = np.append(eigvals1, eigvals[i])
            eig_vectors_return[i] = eig_vectors[:, i]

    return np.diag(eigvals1), eig_vectors_return


def project_image(image, U):
    projection = np.zeros([len(U), 1])
    for i in range(len(U[0])):
        vector = []
        for j in range(len(U)):
            vector.append(U[j, i])
        value = np.dot(np.transpose(vector), np.transpose(image))
        projection[:, 0] = projection[:, 0] + value * U[:, i]
    projection = np.transpose(projection)
    return projection[0]


def display_image(orig, proj):
    original = np.reshape(orig, (28, 28))
    projection = np.reshape(proj, (28, 28))
    figure, axs = matplotlib.pyplot.subplots(1, 2, figsize=(9, 3))

    image1 = axs[0].imshow(original, aspect='equal', cmap='gray')
    figure.colorbar(image1, ax=axs[0])
    image2 = axs[1].imshow(projection, aspect='equal', cmap='gray')
    figure.colorbar(image2, ax=axs[1])
    axs[0].set_title('Original')
    axs[1].set_title('Projection')
    matplotlib.pyplot.show()

