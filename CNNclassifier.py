import datetime
import dicom
from keras import backend as K
from keras import backend as K
from keras.constraints import maxnorm
from keras.datasets import cifar10
from keras.layers import Activation
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Embedding
from keras.layers import Flatten
from keras.layers import GRU
from keras.layers import LSTM
from keras.layers import Permute
from keras.layers import Reshape
from keras.layers import SimpleRNN
from keras.layers.convolutional import Convolution2D
from keras.layers.convolutional import MaxPooling1D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.convolutional import ZeroPadding2D
from keras.models import load_model
from keras.models import Sequential
from keras.optimizers import SGD
from keras.utils import np_utils
from matplotlib import pyplot, cm
import numpy as np
import os
import pandas
import pickle
import progressbar
import scipy.misc
from scipy.misc import toimage
import time
import tensorflow
K.set_image_dim_ordering('th')

'''
Parameters the user can set:
'''
parentdirectory="C:\\Users\\bermads1\\Pictures"
negdir="apt"
posdir="physics"
os.chdir(parentdirectory)
posfiles=[os.path.join(posdir,x) for x in os.listdir(os.path.join(os.getcwd(),posdir))]
negfiles=[os.path.join(negdir,x) for x in os.listdir(os.path.join(os.getcwd(),negdir))]
posdict=pandas.DataFrame({"filename":posfiles,"class":1})
negdict=pandas.DataFrame({"filename":negfiles,"class":0})
allfiles=pandas.concat([posdict,negdict])
allfiles=allfiles.sample(frac=1)


X_data=np.zeros((allfiles.shape[0],3,100,100))
Y_data=np.zeros((allfiles.shape[0]))
pbar = progressbar.ProgressBar(maxval=1).start()
for direc,cl,i in zip(allfiles['filename'],allfiles['class'],range(allfiles.shape[0])):
	image=scipy.misc.imread(direc)
	image=scipy.misc.imresize(image,(100,100))
	image=image/255.0
	for j in range(image.shape[2]):
		X_data[i][j]=image[:,:,j]
	Y_data[i]=cl
	
	pbar.update(i/allfiles.shape[0])
pbar.finish()

model = Sequential()
model.add(Convolution2D(32, 3, 3, input_shape=(3,100, 100), activation='relu', bias=True, W_constraint=maxnorm(3)))
model.add(Dropout(0.2))
model.add(Convolution2D(32, 3, 3, activation='relu', bias=True, W_constraint=maxnorm(3)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Convolution2D(64, 3, 3, activation='relu', bias=True, W_constraint=maxnorm(3)))
model.add(Dropout(0.2))
model.add(Convolution2D(64, 3, 3, activation='relu', bias=True, W_constraint=maxnorm(3)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Convolution2D(128, 3, 3, activation='relu', bias=True, W_constraint=maxnorm(3)))
model.add(Dropout(0.2))
model.add(Convolution2D(128, 3, 3, activation='relu', bias=True, W_constraint=maxnorm(3)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dropout(0.2))
model.add(Dense(1024, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(1, activation='sigmoid'))
# Compile model
epochs = 25
lrate = 0.01
decay = lrate/epochs
sgd = SGD(lr=lrate, momentum=0.9, decay=decay, nesterov=False)
model.compile(loss='binary_crossentropy', optimizer=sgd, metrics=['accuracy'])

model.fit(X_data, Y_data, nb_epoch=epochs)

