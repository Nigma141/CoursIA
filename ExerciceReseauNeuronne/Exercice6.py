from random import uniform
import numpy as np
import matplotlib.pyplot as plt
from numpy import linalg as LA
import matplotlib.image as mpimg


def distance(P, Q):
    Q = np.dot(-1, Q)
    R = np.add(P, Q)
    return LA.norm(R, 2)


def get_classes(P, Q):
    n = len(P)
    target = np.zeros(n)
    # TO DO
    return target


def update(P, target, K):
    n = len(P)
    c = np.zeros((K, 3))
    p = np.zeros(K)
    # update the centroids
    for i in range(n):
    # TO DO

    return q


def split(img, Q, K):
    n, p, _ = img.shape
    res = np.zeros((n, p), dtype=np.float32)
    # TO DO
    return res


def segment(img, K, itermax=30, eps=0.001):
    n, p, _ = img.shape
    print(" dim = ", n, " ", p)
    # TO DO
    return Q


# Load a color image
im1 = mpimg.imread("nao2.png")
plt.imshow(im1)
plt.show()


K = 3
centroides = segment(im1, K)
# print(centroides)
im2 = np.asarray(split(im1, centroides, K), dtype=np.float32)
plt.gray()
plt.imshow(im2)
plt.savefig('nao_seg.png')
plt.show()