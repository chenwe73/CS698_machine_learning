# Wei Tao Chen (wt6chen)
# python 2.7

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from PIL import ImageFilter

import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.optimizers import SGD
from keras.utils import plot_model
from keras.datasets import mnist
#from keras.callbacks import EarlyStopping
#from keras import backend as K
from keras.models import load_model
from keras import regularizers

INPUT_SIZE = 28
batch_size = 256
OUT_PATH = 'out/'
RESULT_PATH = 'result/'

def init():
	(x_train, y_train), (x_test, y_test) = mnist.load_data()
	print (x_train.shape, y_train.shape), (x_test.shape, y_test.shape)
	
	x_train = np.reshape(x_train, x_train.shape + (1,))
	x_test = np.reshape(x_test, x_test.shape + (1,))
	y_train = keras.utils.to_categorical(y_train, num_classes=10)
	y_test = keras.utils.to_categorical(y_test, num_classes=10)
	
	img = Image.fromarray(x_test[1,:,:,0], 'L')
	img.save(OUT_PATH + 'data.png')
	
	n = 60000
	x_train = x_train[0:n]
	y_train = y_train[0:n]
	
	#x_train = x_train.astype('float32')
	#x_test = x_test.astype('float32')
	#x_train /= 255
	#x_test /= 255
	
	np.savetxt(OUT_PATH + 'data.out', x_train[0,:,:,0], fmt='%3d')
	print (x_train.shape, y_train.shape), (x_test.shape, y_test.shape)
	return (x_train, y_train), (x_test, y_test)

def plot(history):
	# accuracy
	plt.gcf().clear()
	plt.plot(history.history['acc'])
	plt.plot(history.history['val_acc'])
	plt.title('model accuracy')
	plt.ylabel('accuracy')
	plt.xlabel('epoch')
	plt.legend(['train', 'test'], loc='best')
	plt.savefig(RESULT_PATH + 'accuracy.png')
	
	# loss
	plt.gcf().clear()
	plt.plot(history.history['loss'])
	plt.plot(history.history['val_loss'])
	plt.title('model loss')
	plt.ylabel('loss')
	plt.xlabel('epoch')
	plt.legend(['train', 'test'], loc='best')
	plt.savefig(RESULT_PATH + 'loss.png')


# conv layer: 3x3 field, stride 1, padding preserve size
# max-pooling: 2x2 field, stride 2
# ReLU activation
def CNN_model():
	model = Sequential()

	model.add(keras.layers.ZeroPadding2D(padding=(2, 2),  input_shape=(INPUT_SIZE, INPUT_SIZE, 1)))
	model.add(Conv2D(64, (3, 3), activation='relu', padding = 'same'))
	model.add(MaxPooling2D(pool_size=(2, 2)))

	model.add(Conv2D(128, (3, 3), activation='relu', padding = 'same'))
	model.add(MaxPooling2D(pool_size=(2, 2)))

	model.add(Conv2D(256, (3, 3), activation='relu', padding = 'same'))
	model.add(Conv2D(256, (3, 3), activation='relu', padding = 'same'))
	model.add(MaxPooling2D(pool_size=(2, 2)))

	model.add(Conv2D(512, (3, 3), activation='relu', padding = 'same'))
	model.add(Conv2D(512, (3, 3), activation='relu', padding = 'same'))
	model.add(MaxPooling2D(pool_size=(2, 2)))

	model.add(Conv2D(512, (3, 3), activation='relu', padding = 'same'))
	model.add(Conv2D(512, (3, 3), activation='relu', padding = 'same'))
	model.add(MaxPooling2D(pool_size=(2, 2)))

	model.add(Flatten())
	model.add(Dense(4096, activation='relu'))
	#model.add(Dropout(0.5)) # need to reduce batch_size
	model.add(Dense(4096, activation='relu'))
	#model.add(Dropout(0.5))
	model.add(Dense(10, activation='softmax'))
	
	plot_model(model, to_file=OUT_PATH + 'model.png', show_shapes = True)
	print model.count_params()
	return model

def CNN_model_reg():
	reg = regularizers.l2(0.01)
	model = Sequential()

	model.add(keras.layers.ZeroPadding2D(padding=(2, 2),  input_shape=(INPUT_SIZE, INPUT_SIZE, 1)))
	model.add(Conv2D(64, (3, 3), activation='relu', padding = 'same', kernel_regularizer=reg ))
	model.add(MaxPooling2D(pool_size=(2, 2)))

	model.add(Conv2D(128, (3, 3), activation='relu', padding = 'same', kernel_regularizer=reg ))
	model.add(MaxPooling2D(pool_size=(2, 2)))

	model.add(Conv2D(256, (3, 3), activation='relu', padding = 'same', kernel_regularizer=reg ))
	model.add(Conv2D(256, (3, 3), activation='relu', padding = 'same', kernel_regularizer=reg ))
	model.add(MaxPooling2D(pool_size=(2, 2)))

	model.add(Conv2D(512, (3, 3), activation='relu', padding = 'same', kernel_regularizer=reg ))
	model.add(Conv2D(512, (3, 3), activation='relu', padding = 'same', kernel_regularizer=reg ))
	model.add(MaxPooling2D(pool_size=(2, 2)))

	model.add(Conv2D(512, (3, 3), activation='relu', padding = 'same', kernel_regularizer=reg ))
	model.add(Conv2D(512, (3, 3), activation='relu', padding = 'same', kernel_regularizer=reg ))
	model.add(MaxPooling2D(pool_size=(2, 2)))

	model.add(Flatten())
	model.add(Dense(4096, activation='relu', kernel_regularizer=reg ))
	#model.add(Dropout(0.5)) # need to reduce batch_size
	model.add(Dense(4096, activation='relu', kernel_regularizer=reg ))
	#model.add(Dropout(0.5))
	model.add(Dense(10, activation='softmax', kernel_regularizer=reg ))
	
	plot_model(model, to_file=OUT_PATH + 'model.png', show_shapes = True)
	print model.count_params()
	return model

def train(model, x_train, y_train, x_test, y_test, epochs):
	# compile
	sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9)
	model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

	# train
	history = model.fit(x_train, y_train, validation_data=(x_test, y_test), batch_size=batch_size, epochs = epochs)
	
	plot(history)
	model.save('model.h5')
	
	score = model.evaluate(x_test, y_test, batch_size=batch_size)
	print "score = ", score
	return model

def rotate(X, deg):
	N = X.shape[0]
	Y = np.array(X)
	for i in range(N):
		img = Image.fromarray(X[i,:,:,0], 'L')
		img = img.rotate(deg)
		Y[i,:,:,0] = np.array(img)
		img.close()
	return Y

def testRotate(model, x_test, y_test):
	n = 19
	deg = np.zeros(n)
	score = np.zeros((n,2))

	for i in range(0, n):
		deg[i] = 5*i - 45
		x_test_Rot = rotate(x_test, deg[i])
		#img = Image.fromarray(x_test_Rot[1,:,:,0], 'L')
		#img.show()
		score[i] = model.evaluate(x_test_Rot, y_test, batch_size=batch_size)
		print 'rotate ', deg[i], score[i]

	plt.gcf().clear()
	plt.plot(deg, score[:,0])
	plt.savefig(RESULT_PATH + 'rotate_loss.png')

	plt.gcf().clear()
	plt.plot(deg, score[:,1])
	plt.savefig(RESULT_PATH + 'rotate_accu.png')

def blur(X, radius):
	N = X.shape[0]
	Y = np.array(X)
	for i in range(N):
		img = Image.fromarray(X[i,:,:,0], 'L')
		img = img.filter(ImageFilter.GaussianBlur(radius))
		Y[i,:,:,0] = np.array(img)
		img.close()
	return Y

def testBlur(model, x_test, y_test):
	n = 7
	radius = np.zeros(n)
	score = np.zeros((n,2))

	for i in range(0, n):
		x_test_Blur = blur(x_test, i)
		#img = Image.fromarray(x_test_Blur[1,:,:,0], 'L')
		#img.show()
		score[i] = model.evaluate(x_test_Blur, y_test, batch_size=batch_size)
		print 'blur ', i, score[i]

	plt.gcf().clear()
	plt.plot(range(0, n), score[:,0])
	plt.savefig(RESULT_PATH + 'blur_loss.png')

	plt.gcf().clear()
	plt.plot(range(0, n), score[:,1])
	plt.savefig(RESULT_PATH + 'blur_accu.png')

def dataAugment(x_train, y_train):
	print 'augmenting data...'
	
	deg = [-45, 45]
	rad = [6]
	n = x_train.shape[0]
	k = 2
	x_train_aug = np.array(x_train)
	y_train_aug = np.array(y_train)
	
	for i in deg:
		x_train_rot = rotate(x_train, i)
		x_train_aug = np.concatenate((x_train_aug, x_train_rot))
		y_train_aug = np.concatenate((y_train_aug, y_train))
		#k = k + n
		#img = Image.fromarray(x_train_aug[k,:,:,0], 'L')
		#img.show()#save(OUT_PATH + 'aug_rot' + str(i) +'.png')
	
	for i in rad:
		radius = 1+ 2*i
		x_train_blur = blur(x_train, i)
		x_train_aug = np.concatenate((x_train_aug, x_train_blur))
		y_train_aug = np.concatenate((y_train_aug, y_train))
		#k = k + n
		#img = Image.fromarray(x_train_aug[k,:,:,0], 'L')
		#img.show()#save(OUT_PATH + 'aug_blur' + str(i) +'.png')
	
	print x_train_aug.shape, y_train_aug.shape
	return x_train_aug, y_train_aug



def main():
	epochs = 2
	isaug = 0
	isreg = 0
	istrain = 1
	istest = 0
	
	(x_train, y_train), (x_test, y_test) = init()
	
	if (isaug):
		x_train, y_train = dataAugment(x_train, y_train)
	
	if (isreg):
		model = CNN_model_reg()
	else:
		model = CNN_model()
	
	if (istrain):
		model = train(model, x_train, y_train, x_test, y_test, epochs)
	else:
		model = load_model('model.h5')
		score = model.evaluate(x_test, y_test, batch_size=batch_size)
		print "score = ", score
	
	if (istest):
		testRotate(model, x_test, y_test)
		testBlur(model, x_test, y_test)
	
	print model.predict(x_test[1:1])
	
main()






