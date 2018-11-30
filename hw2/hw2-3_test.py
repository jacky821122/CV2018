import numpy as np
import os
import pickle
import matplotlib.pyplot as plt
from argparse import ArgumentParser as parser
from skimage import io

parser = parser(description='Computer Vision 2018 Homework 2 (cnn)')
parser.add_argument('dataset_path', 
    nargs='?',
    default='./hw2-3_data/valid/class_0', 
    help = 'where to find dataset, default: ./hw2-3_data/valid/class_0')
parser.add_argument('output_file', 
    nargs='?',
    default='./output.csv', 
    help = 'output file name, default: ./output.csv')
args = parser.parse_args()

img_rows, img_cols = 28, 28
model_file_name = 'model.model'

data_dir = args.dataset_path
x = []
y = []
for i in os.listdir(data_dir):
  data_dir_n = data_dir + '/'
  data = io.imread(os.path.join(data_dir, i))
  x.append(np.array(data))
# print(np.shape(x))
x = np.array(x)

from keras.models import load_model

x = x.reshape(x.shape[0], img_rows, img_cols, 1)
x = x.astype('float32')
x /= 255

model = load_model(model_file_name)
model.summary()

y = model.predict_classes(x)

with open(args.output_file, "w") as output_file:
  output_file.write("id,label\n")
  for i, n in enumerate(y):
    output_file.write("%d,%d\n" % (i, n))

'''
from keras.models import Model
from sklearn.manifold import TSNE
import random
import sys

x = []
for i in range(10):
  data_dir = './hw2-3_data/valid/class_' + str(i) + '/'
  count = 0
  for j in os.listdir(data_dir):
    if count == 100:
      break
    data = io.imread(os.path.join(data_dir, j))
    count += 1
    x.append(np.array(data))
x = np.array(x)
print(np.shape(x))

x = x.reshape(x.shape[0], img_rows, img_cols, 1)
x = x.astype('float32')
x /= 255

features = model.predict(x)
model_extractfeatures = Model(input=model.input, output=model.get_layer('conv2d_1').output)
conv2d_1_features = model_extractfeatures.predict(x)
model_extractfeatures = Model(input=model.input, output=model.get_layer('conv2d_2').output)
conv2d_2_features = model_extractfeatures.predict(x)
feature_maps_low = []
feature_maps_high = []
for i in conv2d_1_features:
  feature_maps_low.append(i.flatten())
for i in conv2d_2_features:
  feature_maps_high.append(i.flatten())
feature_maps_low = np.array(feature_maps_low)
feature_maps_high = np.array(feature_maps_high)
tsne_low = TSNE(n_components=2)
tsne_high = TSNE(n_components=2)
reduced_vec_low = tsne_low.fit_transform(feature_maps_low)
reduced_vec_high = tsne_high.fit_transform(feature_maps_high)
xval_low = reduced_vec_low[:,0]
yval_low = reduced_vec_low[:,1]
plt.figure()
plt.scatter(xval_low, yval_low)
plt.savefig('tsnecnn_low.png')
plt.close()
xval_high = reduced_vec_high[:,0]
yval_high = reduced_vec_high[:,1]
plt.figure()
plt.scatter(xval_high, yval_high)
plt.savefig('tsnecnn_high.png')
plt.close()
'''
