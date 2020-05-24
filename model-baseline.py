import h5py
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.losses import binary_crossentropy	
from tensorflow.keras.layers import InputLayer, Conv2D, AveragePooling2D, Dense, Conv2DTranspose
from tensorflow.keras.layers import LayerNormalization, concatenate, Flatten, Dropout, Reshape
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint, ReduceLROnPlateau


TRAIN_DATASET_PATH = 'uci_ml_hackathon_fire_dataset_2012-05-09_2013-01-01_10k_train.hdf5'
TEST_DATASET_PATH = 'uci_ml_hackathon_fire_dataset_2013-01-01_2014-01-01_5k_test.hdf5'

def getData():
	with h5py.File(TRAIN_DATASET_PATH, 'r') as f:
		train_data = {}
		for k in list(f):
			train_data[k] = f[k][:]

	with h5py.File(TEST_DATASET_PATH, 'r') as f:
		test_data = {}
		for k in list(f):
			test_data[k] = f[k][:]
	return (train_data,test_data)

def standardizeData(data):
	np.nan_to_num(data,copy=False)
	print("New Set: ",np.mean(data),np.std(data),np.min(data),np.max(data))
	for i in range(data.shape[-1]):
		dim = data[:,:,:,i]
		mean = np.mean(dim)
		std = np.std(dim)
		if int(mean)>0 and int(std)>0:
			print(i,mean,std)
			dim = dim-mean
			dim = dim/std
			data[:,:,:,i] = dim

	print(np.mean(data),np.std(data),np.min(data),np.max(data))
	print("----------------------------------\n\n")
	return data

def preprocessData(data):
	geographical_data = data['land_cover']
	geographical_data = np.moveaxis(geographical_data,1,3)
	geographical_data = standardizeData(geographical_data)
	weather_data = data['meteorology'][:,0,:,:,:]
	weather_data = np.moveaxis(weather_data,1,3)
	weather_data = standardizeData(weather_data)
	image_data = data['observed']
	image_data = np.moveaxis(image_data,1,3)
	image_data = standardizeData(image_data)
	training_data = [image_data,geographical_data,weather_data]
	ground_truth = data['target'][:,0,:,:]
	return (training_data,ground_truth)

def prepareDataset(task_mode="classification",timestep="12"):
	train_data,test_data = getData()
	print(test_data.keys())
	if task_mode=="classification":
		if timestep=="12":
			X_train,Y_train = preprocessData(train_data)
			X_test,Y_test = preprocessData(test_data)

			return (X_train,Y_train,X_test,Y_test)

def getWeights(Y_train):
	imp_idx = [1170, 1171, 1172, 1173, 1174, 1175, 1176, 1177, 1178, 1179, 1180,1181, 1182, 1183, 1184, 1185, 1186, 1187, 1188, 
	1189, 1190, 1191,1192, 1193, 1194, 1195, 1196, 1197, 1198, 1199, 1200, 1201, 1202,1203, 1205, 1206, 1207, 1208, 1209, 1210, 1211, 
	1212, 1213, 1214,1215, 1216, 1217, 1218, 1220, 1221, 1222, 1223, 1224, 1225, 1226,1227, 1228, 1229, 1230, 1231, 1232, 1646, 1675, 
	6512, 7062, 7063,7120, 7123, 7124, 7126, 7128, 7375, 7377, 7381, 7382, 7469, 7578,7585, 7586, 7588, 7594, 7597, 7628, 7682, 7688, 
	7706, 7711, 7712,7714, 7717, 7718, 7719, 7721, 7723, 7754, 7755, 7756, 7757, 7759,7760, 7763, 7785, 7786, 7787, 7788, 7789, 7790, 
	7791, 7792, 7793,7794]
	sample_weights = np.ones((Y_train.shape[0]))
	sample_weights[imp_idx] = 3
	return sample_weights


def comboLoss():
	def totalLoss(y_true, y_pred):
		y_true = tf.cast(y_true,dtype=tf.float32)
		numerator = 2 * tf.reduce_sum(y_true * y_pred, axis=(1,2,3))
		denominator = tf.reduce_sum(y_true + y_pred, axis=(1,2,3))
		l2 = tf.reshape(1 - numerator / denominator, (-1, 1, 1))
		return tf.add(binary_crossentropy(y_true, y_pred),l2)

	return totalLoss


def getModel():

	# Input head of the model
	Input1 = InputLayer(input_shape=(30,30,5,))
	Input2 = InputLayer(input_shape=(30,30,17,))
	Input3 = InputLayer(input_shape=(30,30,5,))

	# Convolution segment for VIIRS input
	Part1Conv1 = Conv2D(64,3,activation='relu',data_format='channels_last')(Input1.output)
	Part1Conv1 = AveragePooling2D()(Part1Conv1)
	Part1Conv2 = Conv2D(64,3,activation='relu',data_format='channels_last')(Part1Conv1)
	Part1Flat = Flatten()(Part1Conv2)

	# Convolution segment for Geographical input
	Part2Conv1 = Conv2D(64,3,activation='relu',data_format='channels_last')(Input2.output)
	Part2Conv1 = AveragePooling2D()(Part2Conv1)
	Part2Conv2 = Conv2D(128,3,activation='relu',data_format='channels_last')(Part2Conv1)
	Part2Flat = Flatten()(Part2Conv2)

	# Convolution segment for Weather input
	Part3Conv1 = Conv2D(64,3,activation='relu',data_format='channels_last')(Input3.output)
	Part3Conv1 = AveragePooling2D()(Part3Conv1)
	Part3Conv2 = Conv2D(64,3,activation='relu',data_format='channels_last')(Part3Conv1)
	Part3Flat = Flatten()(Part3Conv2)

	# Projecting all features into a common embedding space
	merge_layer = concatenate([Part1Flat,Part2Flat,Part3Flat])
	Dense1 = Dense(units=2304, activation='relu')(merge_layer)
	Dense1 = Dropout(rate=0.2)(Dense1)
	Dense2 = Dense(units=900, activation='relu')(Dense1)
	Dense2 = Reshape((30,30,1))(Dense2)
	Conv3 = Conv2D(32,1,activation='relu')(Dense2)
	Conv4 = Conv2D(1,1,activation='relu')(Conv3)


	model = Model(inputs=[Input1.input,Input2.input,Input3.input],outputs=Conv4)

	model.summary()
	return model

def createCallbacks():
	tb = TensorBoard(log_dir='./comboloss/logs')
	reduceLR = ReduceLROnPlateau(monitor='val_loss',factor=0.15,min_lr=0.0001)
	ckpt_loss = ModelCheckpoint("./comboloss/checkpoints/weights.{epoch:02d}-{val_loss:.2f}-{val_accuracy:.2f}.h5",monitor="val_loss",verbose=1)
	ckpt_acc = ModelCheckpoint("./comboloss/checkpoints/weights.{epoch:02d}-{val_loss:.2f}-{val_accuracy:.2f}.h5",monitor='val_acc',verbose=1)

	return [tb,reduceLR,ckpt_loss,ckpt_acc]



if __name__ == '__main__':
	model = getModel()
	(X_train,Y_train,X_test,Y_test) = prepareDataset()
	#sample_weights = getWeights(Y_train)
	model.compile(optimizer='adam',loss=comboLoss(), metrics=['accuracy'])
	callbacks = createCallbacks()
	model.fit(X_train,Y_train,batch_size=64,epochs=10,validation_data=(X_test,Y_test),
		steps_per_epoch=10000/64, callbacks=callbacks)


