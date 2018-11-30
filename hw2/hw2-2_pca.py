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

parser = parser(description='Computer Vision 2018 Homework 2 (pca)')
parser.add_argument('dataset_path', 
    nargs='?',
    default='./hw2-2_data', 
    help = 'where to find dataset, default: ./hw2-2_data')
parser.add_argument('input_test_image', 
    nargs='?',
    default='./hw2-2_data/8_9.png',
    help = 'path of the input testing image, default: ./hw2-2_data/8_9.png')
parser.add_argument('output_reconstructed_image', 
    nargs='?',
    default='./reconstructed_testing.png',
    help = 'path of the output reconstructed image, default: ./reconstructed_testing.png')
args = parser.parse_args()

faces_dir = args.dataset_path
testing_img = args.input_test_image
output_face = args.output_reconstructed_image

def pca(x):
    x = x - np.mean(x, axis=0)
    u, _, v = np.linalg.svd(x)
    return v

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
        training_faces.append(face)
# print('training: {}, testing: {}'.format(np.shape(training_faces), np.shape(testing_faces)))
testing_faces = np.array(testing_faces)
testing_faces = sorted(testing_faces, reverse = False, key = lambda x : x[0])
testing_faces = np.array(testing_faces)
testing_faces = testing_faces[:, 1:]
testing_face = np.array(io.imread(testing_img)).flatten()
training_faces = np.array(training_faces)
average_face = np.mean(training_faces, axis=0)
# io.imsave(output_dir + 'average_face.png', average_face.astype(np.uint8).reshape(r, c))

eigenfaces = pca(training_faces)
reconstruct_face = reconstruct(testing_face, average_face, eigenfaces[:r*c])
cv2.imwrite(output_face, reconstruct_face.reshape(r,c))


# -------------- for hw2-2(a1) --------------
'''
plt.figure()
for i in range(5):
    plt.subplot(2, 3, i + 1)
    plt.axis("off")
    plt.imshow(eigenfaces[i].reshape(r, c), cmap="gray")
plt.subplots_adjust(wspace=0.2, hspace=0.2)
plt.suptitle("eigenfaces")
plt.savefig("eigenfaces.png")
plt.close()
'''

# -------------- for hw2-2(a2) --------------
'''
n = [5, 50, 150, np.shape(eigenfaces)[0]]
name = '8_6.png'
face_img = io.imread(os.path.join(faces_dir, name))
face = np.array(face_img).flatten()
for i in n:
    reconstruct_face = reconstruct(face, average_face, eigenfaces[:i])
    plt.figure()
    plt.axis('off')
    plt.imshow(reconstruct_face.reshape(r, c), cmap = 'gray')
    plt.suptitle('MSE = {:0.2f}'.format( np.mean(( (face - reconstruct_face) ** 2 )) ) )
    plt.savefig('reconstructed_with{}.png'.format(i))
    plt.close()
'''


# -------------- for hw2-2(a3) --------------
'''
testing_mean = np.mean(testing_faces, axis = 0)

transformed_faces = np.dot(testing_faces - testing_mean, eigenfaces[:100].T)
tsne = TSNE(n_components=2)
reduced_vec = tsne.fit_transform(transformed_faces)
group = np.arange(40)
ys = [i+group+(i*group)**2 for i in range(40)]
colors = cm.rainbow(np.linspace(0, 1, len(ys)))
plt.figure()
j = 0
for g in np.unique(group):
    for i in range(3):
        plt.scatter(reduced_vec[i+j][0], reduced_vec[i+j][1], c = colors[g], label = g)
    j += 3
plt.savefig('tsne.png')
plt.close()
'''
