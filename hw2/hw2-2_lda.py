import matplotlib
matplotlib.use("TkAgg")

from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import sys
import cv2
import os
from sklearn.manifold import TSNE
from argparse import ArgumentParser as parser

parser = parser(description='Computer Vision 2018 Homework 2 (lda)')
parser.add_argument('dataset_path', 
    nargs='?',
    default='./hw2-2_data', 
    help = 'where to find dataset, default: ./hw2-2_data')
parser.add_argument('output_fisherface', 
    nargs='?',
    default='./output_fisher.png',
    help = 'path of the output fisherface, default: ./output_fisher.png')
args = parser.parse_args()

faces_dir = args.dataset_path
output_face = args.output_fisherface

def pca(x):
    x = x - np.mean(x, axis=0)
    u, _, v = np.linalg.svd(x)
    return v

def lda(x, y):
    mean_vectors = []
    mean_vectors_all = []
    dim = x.shape[1]
    for i in range(40):
        mean_for_1_class = np.mean(x[i*7:i*7+7,:], axis = 0)
        mean_vectors.append(mean_for_1_class)
        for j in range(7):
            mean_vectors_all.append(mean_for_1_class)

    # Sw
    s_ = x - mean_vectors_all
    Sw = np.zeros((dim, dim))
    for xSubmean in s_:
        xSubmean = xSubmean.reshape(-1, 1)
        Sw += np.dot(xSubmean, xSubmean.T)

    # Sb
    overall_mean = np.mean(x, axis = 0)
    overall_mean = overall_mean.reshape(-1, 1)
    Sb = np.zeros((dim, dim))
    for mv in mean_vectors:
        mv = mv.reshape(-1, 1)
        Sb += np.dot((mv - overall_mean), (mv - overall_mean).T)

    cov_mat = np.dot(np.linalg.pinv(Sw), Sb)
    u, _, _ = np.linalg.svd(cov_mat)

    return u

def reconstruct(a, m, eigen_v):
    projected_a = np.dot(a - m, eigen_v.T)
    reconstructed_a = m + np.dot(projected_a, eigen_v)
    return reconstructed_a

training_faces = []
testing_faces = []
img_file_name = os.listdir(faces_dir)
for name in img_file_name:
    face_img = io.imread(os.path.join(faces_dir, name))
    r, c = np.shape(face_img)
    face = np.array(face_img).flatten()
    if int(name[name.find('_')+1:name.find('.')]) > 7:
        testing_faces.append(np.append(np.array(int(name[:name.find('_')])), face))
    else:
        training_faces.append(np.append(np.array(int(name[:name.find('_')])), face))

training_faces = sorted(training_faces, reverse = False, key = lambda x : x[0])
training_faces = np.array(training_faces)
training_label = training_faces[:, 0]
training_faces = training_faces[:, 1:]
average_face = np.mean(training_faces, axis=0)
x = training_faces
y = training_label
N = x.shape[0]
C = 40

eigenfaces = pca(training_faces)
eigenvectors = eigenfaces[:N-C].T
transformed_faces = np.dot(training_faces - average_face, eigenvectors)
W = lda(transformed_faces, y)
W = W[:, :39]
fisherfaces = np.dot(eigenvectors, W)
firstfish = fisherfaces.T[0]
# print(firstfish)
# print(np.shape(fisherfaces.T[0]))
plt.figure(figsize = (c/100, r/100))
plt.axis('off')
plt.figimage(firstfish.reshape(r,c), cmap = 'gray')
plt.savefig(output_face)
plt.close()
# cv2.imwrite(output_face, firstfish.astype(np.uint8).reshape(r, c))

# -------------- for hw2-2(b1) --------------
'''
plt.figure()
for i in range(5):
    plt.subplot(2, 3, i + 1)
    plt.axis("off")
    plt.imshow(fisherfaces.T[i].reshape(r, c), cmap="gray")
plt.subplots_adjust(wspace=0.2, hspace=0.2)
plt.suptitle("fisherfaces")
plt.savefig("fisherfaces.png")
plt.close()
'''

# -------------- for hw2-2(b2) --------------
'''
testing_faces = sorted(testing_faces, reverse = False, key = lambda x : x[0])
testing_faces = np.array(testing_faces)
testing_label = testing_faces[:, 0]
testing_faces = testing_faces[:, 1:]
testing_mean = np.mean(testing_faces, axis = 0)
x = testing_faces
y = testing_label

transformed_faces = np.dot(testing_faces - testing_mean, eigenfaces[:N-C].T)
new_data = np.dot(transformed_faces, W)[:, :30]
# print(np.shape(new_data))
tsne = TSNE(n_components=2)
reduced_vec = tsne.fit_transform(new_data)
group = np.arange(40)
ys = [i+group+(i*group)**2 for i in range(40)]
colors = cm.rainbow(np.linspace(0, 1, len(ys)))
plt.figure()
j = 0
for g in np.unique(group):
    for i in range(3):
        plt.scatter(reduced_vec[i+j][0], reduced_vec[i+j][1], c = colors[g], label = g)
    j += 3
plt.savefig('tsnelda.png')
plt.close()
'''
