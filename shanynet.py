import os
from os import path
from PIL import Image
import numpy as np
from keras.models import Sequential, model_from_json
from keras.layers import Dense, Conv2D, MaxPooling2D, Dropout, Flatten
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import History
from keras.optimizers import Adam
from sklearn.preprocessing import LabelBinarizer
import matplotlib.pyplot as plt
import pickle

def pil_loader(path):
	# open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
	with open(path, 'rb') as f:
		img = Image.open(f)
		img.load()
		img = img.convert('RGB')
	return img

def loadClassNames(dirpath):
	testLabels = []
	labels = set()
	for subdir, dirs, files in os.walk(dirpath):
		for filename in files:
			testLabels.append(subdir.replace(dirpath+'/', ''))
			labels.add(subdir.replace(dirpath+'/', ''))

	return [labels,testLabels]

class ImageLoader(object):
	def __init__(self, dirpath, transform=None, loader=pil_loader):
		self.filepaths_in_dir = []
		self.transform = transform
		self.loader = loader
		for subdir, dirs, files in os.walk(dirpath):
			for filename in files:
				filepath = path.join(subdir, filename)
				if not (filename.lower().endswith('.png') or filename.lower().endswith('.jpg')):
					continue

				img = pil_loader(filepath)
				img.convert('RGB')

				self.filepaths_in_dir.append([np.array(img), subdir.replace(dirpath+'/', '')])

				print(filename + " loaded...")

	def __len__(self):
		return len(self.filepaths_in_dir)


def ShanyNet2():
	""" use of 6 conv layers, 1 fully connected """
	model = Sequential()
	model.add(Conv2D(32, (3, 3), padding='same', activation='relu', input_shape=(224, 224, 3)))  # 32 filters, size of 3x3
	model.add(Conv2D(32, (3, 3), activation='relu'))
	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Dropout(0.25))

	model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))  # 64 filters, size 3x3
	model.add(Conv2D(64, (3, 3), activation='relu'))
	model.add(MaxPooling2D(pool_size=(2, 2)))  # 2x2 max pooling
	model.add(Dropout(0.25)) # technique used to tackle Overfitting

	model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
	model.add(Conv2D(64, (3, 3), activation='relu'))
	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Dropout(0.25))

	model.add(Flatten())  # flat
	model.add(Dense(512, activation='relu', name="layer_1"))
	model.add(Dropout(0.5))
	model.add(Dense(100, activation='softmax', name="layer_2"))  # 100 classes

	model.summary()
	return model


def train_val_test(model=None, train_val_path='', batch=16, epochs=100):
	if model is None:
		return

	if train_val_path == '':
		return

	enable_multithreading = True
	threads = 5

	train_datagen = ImageDataGenerator(rescale=1./255, shear_range=0.2, zoom_range=0.2, horizontal_flip=True)
	valid_datagen = ImageDataGenerator(rescale=1./255, shear_range=0.2, zoom_range=0.2, horizontal_flip=True)

	train_generator = train_datagen.flow_from_directory(
		train_val_path,
		target_size=(224, 224),
		batch_size=batch,
		class_mode='binary')

	val_generator = valid_datagen.flow_from_directory(
		train_val_path,
		target_size=(224, 224),
		batch_size=batch,
		class_mode='binary')



	# calculate best steps for training
	STEP_SIZE_TRAIN=train_generator.n/train_generator.batch_size
	STEP_SIZE_VALID=val_generator.n/val_generator.batch_size

	# store all training information while training the dataset
	history = History()

	print("Training network...")
	model.fit_generator(
		train_generator,
		steps_per_epoch=STEP_SIZE_TRAIN,
		epochs=epochs,
		validation_data=val_generator,
		validation_steps=batch,
		use_multiprocessing=enable_multithreading,
		workers=threads,
		callbacks=[history])

	# save model and weights
	model_json = model.to_json()
	with open("shanynet2.json", "w") as json_file:
		json_file.write(model_json)
	# serialize weights to HDF5
	model.save_weights("shanynet2.h5")
	print("Model saved to disk.")

	print(history.history)
	print("Saving history...")
	with open("shanynet2.history", "wb") as f:
		pickle.dump(history.history, f)

	# model.evaluate_generator(generator=val_generator, steps=STEP_SIZE_VALID, use_multiprocessing=True, workers=6)



def test_model(model=None, test_path=''):
	if model is None:
		return

	if test_path == '':
		return

	enable_multithreading = True
	threads = 5

	test_datagen = ImageDataGenerator(rescale=1./255, shear_range=0.2, zoom_range=0.2, horizontal_flip=True)

	test_generator = test_datagen.flow_from_directory(
		directory=test_path,
		target_size=(224, 224),
		color_mode="rgb",
		batch_size=batch,
		class_mode=None,
		shuffle=False
	)

	STEP_SIZE_TEST = test_generator.n/test_generator.batch_size

	print("Testing network...")
	test_generator.reset()
	prediction=model.predict_generator(test_generator, steps=STEP_SIZE_TEST, use_multiprocessing=enable_multithreading, workers=threads, verbose=1)
	predIdxs=np.argmax(prediction,axis=1)

	print(prediction)

	return predIdxs

def load_model():
	json_file = open('shanynet2.json', 'r')
	model_json = json_file.read()
	model = model_from_json(model_json)
	model.load_weights("shanynet2.h5")
	model._make_predict_function()

	return model

def plotModelInformation(model = None):
	if model is None:
		return

	print(history.history.keys())
	# summarize history for accuracy
	plt.plot(history.history['acc'])
	plt.plot(history.history['val_acc'])
	plt.title('model accuracy')
	plt.ylabel('accuracy')
	plt.xlabel('epoch')
	plt.legend(['train', 'test'], loc='upper left')
	plt.show()
	# summarize history for loss
	plt.plot(history.history['loss'])
	plt.plot(history.history['val_loss'])
	plt.title('model loss')
	plt.ylabel('loss')
	plt.xlabel('epoch')
	plt.legend(['train', 'test'], loc='upper left')
	plt.show()


def main():
	train_val_data_path = '/ib/junk/junk/shany_ds/shany_proj/dataset/train'
	test_data_path = '/ib/junk/junk/shany_ds/shany_proj/dataset/test'
	save_model_path = '/ib/junk/junk/shany_ds/shany_proj/model/model2.h5'

	data = loadClassNames(train_val_data_path)
	lb = LabelBinarizer()
	lb.fit(list(data[0]))
	classes = lb.transform(data[1])
	# from tensorflow.python.client import device_lib
	# print(device_lib.list_local_devices())

	#load model
	#model = load_model()
	#return

	# conf
	batch = 16
	epochs = 10  # number of epochs



	# train from new model
	model = ShanyNet2()
	# opt = 'rmsprop'
	# opt = 'sgd'
	# opt = optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
	opt = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
	model.compile(optimizer=opt, loss='sparse_categorical_crossentropy', metrics=['accuracy'])

	train_val_test(model=model, train_val_path=train_val_data_path, test_path=test_data_path, batch=batch, epochs=epochs)

main()