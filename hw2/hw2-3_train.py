import numpy as np
import os
import pickle
import matplotlib.pyplot as plt
from argparse import ArgumentParser as parser
from skimage import io

parser = parser(description='Computer Vision 2018 Homework 2 (cnn)')
parser.add_argument('dataset_path', 
    nargs='?',
    default='./hw2-3_data', 
    help = 'where to find dataset, default: ./hw2-3_data')
args = parser.parse_args()

model_file_name = 'model.model'

data_dir = args.dataset_path + '/train'
x_train = []
y_train = []
count = 0
for i in os.listdir(data_dir):
    data_dir_n = data_dir + '/' + i
    for j in os.listdir(data_dir_n):
        data = io.imread(os.path.join(data_dir_n, j))
        x_train.append(np.array(data))
        y_train.append(np.array(int(i[6])))
        count += 1
x_train = np.array(x_train)
y_train = np.array(y_train)

data_dir = args.dataset_path + '/valid'
x_test = []
y_test = []
count = 0
for i in os.listdir(data_dir):
    data_dir_n = data_dir + '/' + i
    for j in os.listdir(data_dir_n):
        data = io.imread(os.path.join(data_dir_n, j))
        x_test.append(np.array(data))
        y_test.append(np.array(int(i[6])))
        count += 1
x_test = np.array(x_test)
y_test = np.array(y_test)

import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K

batch_size = 128
num_classes = 10
epochs = 12

img_rows, img_cols = 28, 28

if K.image_data_format() == 'channels_first':
    x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
    x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
    input_shape = (1, img_rows, img_cols)
else:
    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3),
                 activation='relu',
                 input_shape=input_shape))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))

model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])

model.summary()

# keras.utils.plot_model(model, to_file='model.png')
history = model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_data=(x_test, y_test))

score = model.evaluate(x_train, y_train, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

model.save(model_file_name)

dump_file_name = 'model.h5'
with open(dump_file_name, "wb") as dump_file:
    pickle.dump({"history": history.history}, dump_file)

'''
dump_file_name = 'model.h5'
with open(dump_file_name, "rb") as dump_file:
    m = pickle.load(dump_file)
    history = m["history"]
'''

'''
plt.figure()
plt.plot(history['acc'])
plt.title('Learning Curve (Accuracy)')
plt.ylabel('Accuracy')
plt.xlabel('# of epoch')
plt.savefig('LC_acc.png')
plt.close()
plt.figure()
plt.plot(history['loss'])
plt.title('Learning Curve (Loss)')
plt.ylabel('Loss')
plt.xlabel('# of epoch')
plt.savefig('LC_loss.png')
plt.close()
'''
