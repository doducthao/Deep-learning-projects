from tensorflow.keras import layers, Model
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Dense, Dropout, Flatten, MaxPooling2D
import tensorflow as tf 
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
import os
import argparse
import cv2
import argparse

def create_model_transfer(load_weights_inceptionv3=False):
    pretrained_model = tf.keras.applications.InceptionV3(
        include_top=False,
        weights=None,
        input_shape=(75,75,3), # 75 is least required size of imagenet
        classifier_activation='softmax')

    if load_weights_inceptionv3:
    	local_weights_file = 'models/inception_v3_weights_tf_dim_ordering_tf_kernels_notop'
    	pretrained_model.load_weights(local_weights_file)

    for layer in pretrained_model.layers:
        layer.trainable=False

    last_layer = pretrained_model.get_layer('mixed7')
    last_output = last_layer.output

    x = layers.Flatten()(last_output)
    x = layers.Dense(1024, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(10, activation='softmax')(x)

    model = Model(inputs=pretrained_model.input, outputs=x)
    model.compile(optimizer=tf.keras.optimizers.Adam(lr=1e-3),
                loss = 'categorical_crossentropy',
                metrics=['accuracy'])
    return model

def transform(imgs,size):
    imgs = np.array([cv2.resize(img, (size)) for img in imgs], dtype=np.float32)
    imgs = np.expand_dims(imgs, axis=-1)
    imgs = np.repeat(imgs,3,-1)
    imgs = imgs/255.
    return imgs

def preprocess_data_npy(size):
    (train_mnist, labels_train_mnist), (test_mnist, labels_test_mnist)= tf.keras.datasets.mnist.load_data()

    train_data = np.load('data/train_data.npy')
    train_labels = np.load('data/train_labels.npy')
    validation_data = np.load('data/validation_data.npy')
    validation_labels = np.load('data/validation_labels.npy')

    train_x = np.concatenate((train_mnist, train_data), axis=0)
    train_y = np.concatenate((labels_train_mnist, train_labels), axis=0)
    test_x = np.concatenate((test_mnist, validation_data), axis=0)
    test_y = np.concatenate((labels_test_mnist, validation_labels), axis=0)

    train_x = transform(train_x,size)
    test_x = transform(test_x,size)

    train_y = tf.keras.utils.to_categorical(train_y, 10)
    test_y = tf.keras.utils.to_categorical(test_y, 10)

    return (train_x, train_y), (test_x, test_y)
 

class StopTraining(tf.keras.callbacks.Callback):
	def __init__(self, desired_acc):
		self.desired_acc = desired_acc

	def on_epoch_end(self, epoch, logs={}):
		if logs.get('val_accuracy') > self.desired_acc:
			print('\n Reached {} accuracy so cancelling training!'.format(self.desired_acc))
			self.model.stop_training = True

def train(batch_size, epochs, desired_acc, model):
	checkpoint_path = "training/sudoku_transfer/cp-{epoch:04d}.ckpt"
	checkpoint_dir = os.path.dirname(checkpoint_path)

	my_callbacks = tf.keras.callbacks.ModelCheckpoint(
		filepath=checkpoint_path,
		save_weights_only=True,
		save_best_only=True, 
		verbose=1)

	early_stopping = tf.keras.callbacks.EarlyStopping(monitor='loss',patience=10)
	reduce_lr_on_plateau = tf.keras.callbacks.ReduceLROnPlateau(
		monitor='val_loss',
		factor=0.1, 
		patience=10)
	
	stop_training = StopTraining(desired_acc)
	history = model.fit(train_x,
						train_y,
						validation_data=(test_x, test_y),
						batch_size=batch_size,
						epochs=epochs,
						verbose=1)
	model.save_weights('models/sudoku_weights.h5')

def load_best_model(path_model):
	model = create_model_transfer()
	model.load_weights(path_model)
	return model


if __name__ == '__main__':
	parse = argparse.ArgumentParser(description='train model')
	parse.add_argument("-m", "--model", help="model training")
	parse.add_argument('-e', '--epochs', type=int, default=100, help='number of epochs to train')
	parse.add_argument('-d', '--desired_acc', type=float, default=0.99, help='desired accuracy')
	parse.add_argument('-b', '--batch_size', type=int, default=128, help='Size of data batch on each epoch')
	parse.add_argument("-tr", "--train", dest='train', action='store_true', help="boolean, True to train, False to not")
	parse.set_defaults(feature=True)

	args = parse.parse_args()
	
	model=create_model_transfer()
	if args.train:
		train(
			epochs=args.epochs,
			desired_acc=args.desired_acc,
			batch_size=args.batch_size,
			path_model=args.model)


	










 





