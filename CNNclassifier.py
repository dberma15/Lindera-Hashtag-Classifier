'''
This needs to be done: type activate tensorflow-gpu
http://www.heatonresearch.com/2017/01/01/tensorflow-windows-gpu.html
'''
import datetime
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
from keras.preprocessing.image import ImageDataGenerator
from matplotlib import pyplot, cm
import numpy as np
import os
import pandas
from pandas_confusion import BinaryConfusionMatrix
import pickle
import progressbar
import scipy.misc
from scipy.misc import toimage
from sklearn import svm
from sklearn.svm import SVC
from sklearn import ensemble
from sklearn.metrics import confusion_matrix
import time
import tensorflow
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
K.set_image_dim_ordering('th')

# '''
# Parameters the user can set:
# '''


parentdirectory="C:\\Users\\daniel\\Documents\\Machine Learning And Statistics Projects\\Lindera HAshtag Classifier"
onDanielsComputer=os.path.isdir(parentdirectory)

if onDanielsComputer:
	os.chdir(parentdirectory)
	positivedirectory="wildfoodlove"
	negativedirectory="gym"
	print('loading in the data')
	positivevalues=[os.path.join(positivedirectory,x) for x in os.listdir(os.path.join(parentdirectory,positivedirectory))]
	negativevalues=[os.path.join(negativedirectory,x) for x in os.listdir(os.path.join(parentdirectory,negativedirectory))]
	neg={"files": pandas.Series(negativevalues), "class": pandas.Series(np.zeros(len(negativevalues)))}
	pos={"files": pandas.Series(positivevalues), "class": pandas.Series(np.ones(len(positivevalues)))}
	data=pandas.concat([pandas.DataFrame(neg),pandas.DataFrame(pos)])
	data=data.sample(frac=1)

	X_training=np.zeros((data.shape[0],3,32,32))
	Y_training=np.zeros((data.shape[0]))

	pbar = progressbar.ProgressBar(maxval=1).start()
	for file,cl,i in zip(data['files'],data['class'],range(data.shape[0])):
		image=scipy.misc.imread(file)
		image=scipy.misc.imresize(image,(32,32,3))
		# image=image/255.0
		for j in range(image.shape[2]):
			X_training[i][j]=image[:,:,j]
		Y_training[i]=int(cl)
		pbar.update(i/data.shape[0])
	pbar.finish()
	np.save("X_data_wildfoodlovevsgym.npz",X_training)
	np.save("Y_data_wildfoodlovevsgym.npz",Y_training)
else:
	print("loading data")
	X_training=np.load("X_data_wildfoodlovevsgym.npz.npy")
	Y_training=np.load("Y_data_wildfoodlovegym.npz.npy")

print("Create test and training sets")
msk=np.random.rand(len(X_training))<.75
#Y_training=np_utils.to_categorical(Y_training)
x_train=X_training[msk]
y_train=(Y_training[msk])
x_test=X_training[~msk]
y_test=(Y_training[~msk])

(X_train, Y_train), (X_test, Y_test) = cifar10.load_data()
is9or6=[(y[0]==6 or y[0]==9) for y in Y_train]
X_train=X_train[is9or6][:,:,:]
Y_train=Y_train[is9or6]
Y_train=[int(y==6) for y in Y_train]


X_test=X_train[300:400]
Y_test=Y_train[300:400]
X_train=X_train[0:300]
Y_train=Y_train[0:300]

# Create the model
model=Sequential()
model.add(Convolution2D(32, 3, 3, input_shape=(3, 32, 32), activation='relu'))
model.add(Dropout(0.2))
model.add(Convolution2D(32, 3, 3, activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Convolution2D(64, 3, 3, activation='relu'))
model.add(Dropout(0.2))
model.add(Convolution2D(64, 3, 3, activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Convolution2D(128, 3, 3, activation='relu'))
model.add(Dropout(0.2))
model.add(Convolution2D(128, 2,2, activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dropout(0.2))
model.add(Dense(1024, activation='relu', W_constraint=maxnorm(3)))
model.add(Dropout(0.2))
model.add(Dense(512, activation='relu', W_constraint=maxnorm(3)))
model.add(Dropout(0.2))
model.add(Dense(1, activation='sigmoid'))

epochs = 100
lrate = 0.0001
decay = lrate/epochs
sgd = SGD(lr=lrate, momentum=0.9, decay=decay, nesterov=False)
print('compiling model')
model.compile(loss='binary_crossentropy', optimizer=sgd, metrics=['accuracy'])

# datagen = ImageDataGenerator(
    # featurewise_center=True,
    # featurewise_std_normalization=True,
    # rotation_range=20,
    # width_shift_range=0.2,
    # height_shift_range=0.2,
    # horizontal_flip=False)
	
# print("fitting model")
# model.fit_generator(datagen.flow(x_train, y_train,batch_size=32), samples_per_epoch=len(x_train),  nb_epoch=epochs, validation_data=(x_train,y_train) )
model.fit(x_train, y_train,batch_size=32, nb_epoch=epochs, validation_data=(x_test,y_test) )


modelsavename="LinderaCNN"

ts=time.time()
st=datetime.datetime.fromtimestamp(ts).strftime("%Y%m%d %H%M%S")
filename=modelsavename+st+'.h5'


model.save(filename)
predictions=model.predict(x_test)

print(max(predictions))
print(min(predictions))
predictions=predictions.reshape(predictions.shape[0])
comparison={'predicted':predictions, 'actual':pandas.Series(y_test)}
#print(comparison)
comparison=pandas.DataFrame(comparison)
comparison.to_csv(modelsavename+st+"_"+positivedirectory+"_"+negativedirectory+"predictions.csv")
print(comparison)
