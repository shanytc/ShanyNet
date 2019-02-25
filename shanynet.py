import os
from os import path
from PIL import Image
import numpy as np
from keras.models import Sequential, model_from_json
from keras.layers import Dense, Conv2D, MaxPooling2D, Dropout, Flatten
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import History
import matplotlib.pyplot as plt
import pickle

def pil_loader(path):
	# open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
	with open(path, 'rb') as f:
		img = Image.open(f)
		img.load()
		img = img.convert('RGB')
	return img


class ShanyNet:
	def __init__(self, cfg=None):
		self.config = cfg

		if cfg is None:
			self.config = CFG()

		self.model = None
		self.predictions = None
		self.classes = None

	def simple(self):

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
		model.add(Dense(512, activation='relu', name="layer_512"))
		model.add(Dropout(0.5))
		model.add(Dense(100, activation='softmax', name="layer_100"))  # 100 classes

		model.summary()
		self.model = model

		return self

	def create(self):
		self.model = None
		self.model = Sequential()
		return self

	def add_2d(self, filters=32, kernel=(3,3), **kwargs):
		self.model.add(Conv2D(filters, kernel, **kwargs))  # 32 filters, size of 3x3
		return self

	def add_max_pooling(self, size=(2, 2)):
		self.model.add(MaxPooling2D(pool_size=size))  # 2x2 max pooling
		return self

	def add_dropout(self, dropout=0.25):
		self.model.add(Dropout(dropout))
		return self

	def add_flatten(self):
		self.model.add(Flatten())
		return self

	def add_dense(self, size=512, **kwargs):
		self.model.add(Dense(size, **kwargs))
		return self

	def add_basic_block(self):
		self.add_2d(filters=64, kernel=(3, 3), padding='same', activation='relu') \
			.add_2d(filters=64, kernel=(3, 3), activation='relu') \
			.add_max_pooling() \
			.add_dropout()

		return self

	def load_model(self):
		json_file = open(self.config.get_model_name() + '.json', 'r')
		model_json = json_file.read()
		model = model_from_json(model_json)
		model.load_weights(self.config.get_model_name() + ".h5",by_name=True)

		self.model = model

		return self

	def show_model_summary(self):
		self.model.summary()
		return self

	def get_model(self):
		return self.model

	def load_classes(self, dirpath=''):
		if dirpath == '':
			self.classes = []
			return self

		self.classes = []
		for subdir, dirs, files in os.walk(dirpath):
			for filename in files:
				self.classes.append(subdir.replace(dirpath+'/', ''))

		return self

	def compile(self):
		if self.model is None:
			return self

		self.model.compile(optimizer=self.config.get_optimizer(), loss=self.config.get_loss_function(), metrics=self.config.get_compile_metrics())

		return self

	def train(self):
		if self.model is None:
			return self

		if self.config.get_train_val_path() == '':
			return self

		train_datagen = ImageDataGenerator(rescale=1./255)
		valid_datagen = ImageDataGenerator(rescale=1./255, shear_range=0.2, zoom_range=0.2, horizontal_flip=True)

		train_generator = train_datagen.flow_from_directory(
			self.config.get_train_val_path(),
			target_size=(224, 224),
			batch_size=self.config.get_batch_size(),
			class_mode='binary')

		val_generator = valid_datagen.flow_from_directory(
			self.config.get_train_val_path(),
			target_size=(224, 224),
			batch_size=self.config.get_batch_size(),
			class_mode='binary')

		# steps for training
		steps = train_generator.n/train_generator.batch_size

		# store all training information while training the dataset
		history = History()

		print("Training network...")
		self.model.fit_generator(
			train_generator,
			steps_per_epoch=steps,
			epochs=self.config.get_num_epochs(),
			validation_data=val_generator,
			validation_steps=self.config.get_batch_size(),
			use_multiprocessing=self.config.get_multithreading_status(),
			workers=self.config.get_num_threads(),
			callbacks=[history])

		if self.config.enable_saving is True:
			if self.config.get_should_save_model() is True and self.config.get_model_name() != '':
				# save model and weights
				model_json = self.model.to_json()
				with open(self.config.get_model_name() + ".json", "w") as json_file:
					json_file.write(model_json)

				print("Model saved to disk.")

			if self.config.get_should_save_weights() is True and self.config.get_model_name() != '':
				# serialize weights to HDF5
				self.model.save_weights(self.config.get_model_name() + ".h5")
				print("Model's weights saved to disk.")

			if self.config.get_should_save_history() is True and self.config.get_model_name() != '':
				with open(self.config.get_model_name() + ".history", "wb") as f:
					pickle.dump(history.history, f)
				print("Model's history saved.")

		# Evaluate model
		# model.evaluate_generator(generator=val_generator, steps=STEP_SIZE_VALID, use_multiprocessing=True, workers=6)

		return self

	def test(self):
		if self.model is None:
			return self

		if self.config.get_test_path() == '':
			return self

		test_datagen = ImageDataGenerator()

		test_generator = test_datagen.flow_from_directory(
			directory=self.config.get_test_path(),
			target_size=(224, 224),
			color_mode="rgb",
			batch_size=self.config.get_batch_size(),
			class_mode=None,
			shuffle=False
		)

		steps = test_generator.n/test_generator.batch_size

		print("Testing network...")
		test_generator.reset()
		self.predictions = self.model.predict_generator(test_generator, steps=steps, use_multiprocessing=self.config.get_multithreading_status(), workers=self.config.get_num_threads(), verbose=1)

		return self

	def infer(self, image_path=None):
		if self.model is None:
			return self

		if self.config.get_test_path() == '':
			return self

		test_datagen = ImageDataGenerator()

		test_generator = test_datagen.flow_from_directory(
			directory=image_path,
			target_size=(224, 224),
			color_mode="rgb",
			batch_size=self.config.get_batch_size(),
			class_mode=None,
			shuffle=False
		)

		steps = test_generator.n/test_generator.batch_size

		print("Testing network...")
		test_generator.reset()
		predictions = self.model.predict_generator(test_generator, steps=steps, use_multiprocessing=self.config.get_multithreading_status(), workers=self.config.get_num_threads(), verbose=1)

		# print embeddings
		print(predictions)

		return self


	def get_predictions(self):
		return self.predictions

	def get_predictions_indexes(self):
		return np.argmax(self.predictions, axis=1)

class CFG:
	def __init__(self, batch=10,
			epochs=100,
			enable_multithreading=True,
			threads=10,
			train_val_path='',
			test_path='',
			model_output_path='',
			model_name='',
			optimizer='sgd',
			loss_function='sparse_categorical_crossentropy',
			compile_metrics=['metrics'],
			enable_saving = False,
			save_model=False,
			save_weights=False,
			save_history=False):
		self.batch = batch
		self.epochs = epochs
		self.enable_multithreading = enable_multithreading
		self.threads = threads
		self.train_val_path = train_val_path
		self.test_path = test_path
		self.model_output_path = model_output_path
		self.model_name = model_name
		self.optimizer = optimizer
		self.loss_function = loss_function
		self.compile_metrics = compile_metrics
		self.save_model = save_model
		self.save_weights = save_weights
		self.save_history = save_history
		self.enable_saving = enable_saving

	def set_optimizer(self, optimizer=None):
		if optimizer is None:
			return self

		self.optimizer = optimizer

		return self

	def get_model_name(self):
		return self.model_name

	def get_model_output_path(self):
		if self.model_output_path[-1] == '/':
			return self.model_output_path + self.model_name

		return self.model_output_path + '/' + self.model_name

	def get_should_save_history(self):
		return self.save_history

	def get_should_save_weights(self):
		return self.save_weights

	def get_should_save_model(self):
		return self.save_model

	def get_train_val_path(self):
		return self.train_val_path

	def get_test_path(self):
		return self.test_path

	def get_num_epochs(self):
		return self.epochs

	def get_batch_size(self):
		return self.batch

	def get_num_threads(self):
		return self.threads

	def get_multithreading_status(self):
		return self.enable_multithreading

	def get_optimizer(self):
		return self.optimizer

	def get_loss_function(self):
		return self.loss_function

	def get_compile_metrics(self):
		return self.compile_metrics


def main():
	# Activations
	# ============
	# 'softmax'
	# 'relu'
	# 'sigmoid'
	# 'elu'
	# 'selu'
	# 'softplus'
	# 'softsign'
	# 'tanh'
	# 'hard_sigmoid'
	# 'exponential'
	# 'linear'

	# Optimizers:
	# =============
	# 'RMSprop'
	# 'Adagrad'
	# 'Adadelta'
	# 'Adam'
	# 'Adamax'
	# 'Nadam'
	# 'SGD' / optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
	# 'adam' / optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)

	# Loss Functions
	# ==============
	# 'mean_squared_error'
	# 'mean_absolute_error'
	# 'mean_absolute_percentage_error'
	# 'mean_squared_logarithmic_error'
	# 'squared_hinge'
	# 'hinge'
	# 'categorical_hinge'
	# 'logcosh'
	# 'categorical_crossentropy'
	# 'sparse_categorical_crossentropy'
	# 'binary_crossentropy'
	# 'kullback_leibler_divergence'
	# 'poisson'
	# 'cosine_proximity'

	cfg = CFG(batch=16,
			epochs=10,
			enable_multithreading=True,
			threads=5,
			train_val_path='/ib/junk/junk/shany_ds/shany_proj/dataset/train/',
			test_path='/ib/junk/junk/shany_ds/shany_proj/dataset/test/',
			model_output_path='/ib/junk/junk/shany_ds/shany_proj/model/',
			model_name='shanynet3',
			optimizer='rmsprop',
			loss_function='sparse_categorical_crossentropy',
			compile_metrics=['accuracy'],
			enable_saving=True,
			save_model=True,
			save_weights=True,
			save_history=True)

	Net = ShanyNet(cfg=cfg)
	Net.create()\
		.add_2d(filters=32, kernel=(3, 3), activation="relu", padding='same', input_shape=(224, 224, 3))\
		.add_2d(filters=32, kernel=(3, 3), activation='relu')\
		.add_max_pooling()\
		.add_dropout()\
		.add_basic_block()\
		.add_basic_block()\
		.add_flatten()\
		.add_dense(size=512, activation='relu', name="layer_512")\
		.add_dense(size=100, activation='softmax', name="layer_100")\
		.show_model_summary()\
		.compile()\
		.train()\
		.test()

	# model.add(Flatten())  # flat
	# model.add(Dense(512, activation='relu', name="layer_512"))
	# model.add(Dropout(0.5))
	# model.add(Dense(100, activation='softmax', name="layer_100"))  # 100 classes


	#Net.simple().compile().train().test()  # create simple network
	#Net.load_model().infer('/ib/junk/junk/shany_ds/shany_proj/test_image')  # load model

main()