import TextToArray as TTA
import plotting
import matplotlib.pyplot as plt
import numpy as np
import cPickle
import sys
from copy import deepcopy
import os

from keras.preprocessing import sequence
from keras.optimizers import SGD, RMSprop, Adagrad
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Merge, Flatten, TimeDistributedDense, Masking, Lambda
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM, GRU
from keras.models import model_from_json

from CustomFunctions import MaskingHack, MaskingHack_output_shape

import plottingUtils
import json
from optparse import OptionParser

sys.setrecursionlimit(40000)

# TODO List:
# https://docs.google.com/spreadsheets/d/1nL6EDw3ALPQpDNQL3V-2lKSN6kzroPqHFQDA6MR_TUg/edit#gid=0

dataset_storage = None
model_storage = None
history_storage = None

p = OptionParser()
p.add_option('--Var', type = "string", default = 'IP3D',   dest = 'Variables', help = 'Variables to be included in model')
p.add_option('--Mode', type = "string", default = 'M',  dest = 'Mode', help = 'Type of Study: building model [M] or check ROC/results [C] ')
p.add_option('--nEpoch', type = "string", default = '50', dest = 'nEpoch', help = 'number of epochs ')
p.add_option('--nEvents', type = "string", default = '10000', dest = 'nEvents', help = 'number of events ')
p.add_option('--nMaxTrack', type ="string", default= '15', dest="nMaxTrack", help="Maximum number of tracks")
p.add_option('--nTrackCut', type ="string", default= '0', dest="nTrackCut", help="Cut on jets with exact n tracks")
p.add_option('--doBatch', type ="string", default= 'n', dest="doBatch", help="Whether run batch job")
p.add_option('--doTrainC', type ="string", default= 'y', dest="doTrainC", help="Whether include C jets in trainning sample")
p.add_option('--TrackOrder', type ="string", default= 'Sd0', dest="TrackOrder", help="Track Ordering [Sd0], [pT] more to be added")
p.add_option('--padding', type = "string", default = 'pre', dest="padding", help="padding order, pre or post")
p.add_option('--Model', type = "string", default = 'LSTM', dest="Model", help="Model type: LSTM, DenseIP3D")
p.add_option('--AddJetpT', type = "string", default = 'n', dest="AddJetpT", help="if add jet pT to the model of RNN+")
p.add_option('--nLSTMNodes', type = "string", default = '25', dest="nLSTMNodes", help="number of hidden nodes for the LSTM algorithm")
p.add_option('--nLSTMClass', type = "string", default = '2', dest="nLSTMClass", help="the number of output classes")


(o,a) = p.parse_args()

nb_epoch = int(o.nEpoch)
max_len = int(o.nMaxTrack)
batch_size = 128
n_events = int(o.nEvents)
trainFraction = 0.8
max_embed_features = 16
embed_size = 2
ntrk_cut = int(o.nTrackCut)

print 'number of epoch ', nb_epoch, 'number of events ', n_events

def makeData( Variables = "IP3D", max_len=max_len, padding= o.padding, nLSTMClass = o.nLSTMClass):
	print "Getting Data ..."

	f = None
	if o.TrackOrder == "Sd0" and Variables == "phi":
		f = file('MakeData/Dataset_V47_IP3D_pTFrac_dphi_deta_5m.pkl','r')
	if o.TrackOrder == "Sd0" and Variables == "dtheta":
		f = file('MakeData/Dataset_V47_IP3D_pTFrac_dphi_dtheta_5m.pkl','r')
	if o.TrackOrder == "Sd0" and Variables == "d0z0":
		f = file('MakeData/Dataset_V47_IP3D_pTFrac_d0_z0_5m.pkl','r')

	if o.TrackOrder == "Sd0" and Variables == "dR":
		f = file('MakeData/Dataset_V47_IP3D_pTFrac_dR_5m.pkl','r')
	if o.TrackOrder == "Sd0" and Variables == "IP3D":
		f = file('MakeData/Dataset_V47_IP3D_pTFrac_dR_5m.pkl','r')

	if o.TrackOrder == "SL0":
		f = file('MakeData/Dataset_IP3D_pTFrac_dR_5m_CMix_MV2C20_SL0Sort.pkl','r')
	if o.TrackOrder == "pT":
		f = file('MakeData/Dataset_IP3D_pTFrac_dR_5m_CMix_pTSort.pkl','r')

	trk_arr_all = cPickle.load(f)
	labels_all = cPickle.load(f)
	f.close()

	###########
	
	# input variables
	print "Getting Input Variables"
	X_all = None
	X = None
	
	if Variables == "IP3D":
		X = TTA.MakePaddedSequenceTensorFromListArray( trk_arr_all[:,0:2], doWhitening=False, maxlen=max_len, padding = padding)	

	if Variables == "pTFrac":
		X = TTA.MakePaddedSequenceTensorFromListArray( trk_arr_all[:,0:3], doWhitening=False, maxlen=max_len, padding = padding)	

	if Variables == "dR":
		X = TTA.MakePaddedSequenceTensorFromListArray( trk_arr_all[:,0:4], doWhitening=False, maxlen=max_len, padding = padding)	

	if Variables == "phi":
		X = TTA.MakePaddedSequenceTensorFromListArray( trk_arr_all[:,0:5], doWhitening=False, maxlen=max_len, padding = padding)	

	if Variables == "d0z0":
		X = TTA.MakePaddedSequenceTensorFromListArray( trk_arr_all[:,0:5], doWhitening=False, maxlen=max_len, padding = padding)	

	if Variables == "dtheta":
		X = TTA.MakePaddedSequenceTensorFromListArray( trk_arr_all[:,0:5], doWhitening=False, maxlen=max_len, padding = padding)	

	trk_grd = TTA.convertSequencesFromListArray( trk_arr_all[:,5], dopad=True, pad_value=-1, maxlen=max_len )

	if Variables == "dR" or Variables == "IP3D":
		trk_grd = TTA.convertSequencesFromListArray( trk_arr_all[:,4], dopad=True, pad_value=-1, maxlen=max_len )

	print "Getting Labels"
	print "padding ", padding

	X_all = np.dstack( (X, trk_grd+1) )
	X = X_all[:n_events]
	labels = labels_all[:n_events]
	y = (labels[:,0] ==5)

	if int(nLSTMClass) == 4 and o.Model == "LSTM":
		y = np.ndarray(shape =(labels.shape[0],4), dtype=float)
		y[:, 0] = (labels[:,0] ==5)
		y[:, 1] = (labels[:,0] ==4)
		y[:, 2] = (labels[:,0] ==0)
		y[:, 3] = (labels[:,0] ==15)

	if ntrk_cut != 0:
		print ' cutting on number of tracks to be exactly ', ntrk_cut
		X = X[ labels[:, 7] == ntrk_cut]
		y = y[ labels[:, 7] == ntrk_cut]
		labels = labels[ labels[:, 7] == ntrk_cut]

	if o.doTrainC != 'y':
		print ' not training on C jets'
		X = X[ labels[:,0]!=4]
		y = y[ labels[:,0]!=4]
		labels = labels[ labels[:,0]!=4]
	
	## apply jvt cut
	JVTcuts =  np.logical_or(labels[:,11]>0.52 ,labels[:,1]>60000)
	X = X[ JVTcuts]
	y = y[ JVTcuts]
	labels = labels[JVTcuts] 
		
	X_train, X_test = np.split( X, [ int(trainFraction*X.shape[0]) ] )
	y_train, y_test = np.split( y, [ int(trainFraction*y.shape[0]) ] )
	labels_train, labels_test = np.split( labels, [ int(trainFraction*labels.shape[0]) ] )
	ip3d_test = labels_test[:,3]


	print("data shape",X.shape)
	print y_train.shape, y_test.shape

	dataset = {
	  "X": X,
	  "X_train": X_train,
	  "X_test": X_test,
	  "labels": labels,
	  "labels_train": labels_train,
	  "labels_test": labels_test,
	  "y": y,
	  "y_train": y_train,
	  "y_test": y_test
	}

	return dataset


def buildModel_1hidden(dataset, useAdam=False):

	print "Building Model ..."

	#################
	# Configuration #
	#################

	X_train = dataset['X_train']
	X_test = dataset['X_test']
	y_train = dataset['y_train']
	print y_train

	# split by "continuous variable" and "categorization variable"
	X_train_vec = [X_train[:,:,0:-1],  X_train[:,:,-1] ]

	n_cont_vars = X_train_vec[0].shape[2]
	print ' number of continuous input variables ', n_cont_vars
	
	##################
	print "shape ", X_train_vec[0].shape,  X_train_vec[1].shape

	left = Sequential()
	left.add( Activation('linear', input_shape=(max_len, n_cont_vars)) )
	#left.add( TimeDistributedPassThrough( input_shape=(max_len, n_cont_vars) ) )

	right = Sequential()
	right.add(Embedding(max_embed_features, embed_size, mask_zero=False, input_length=max_len))

	model = Sequential()
	model.add( Merge([left, right],mode='concat') )

	model.add(Lambda(MaskingHack, output_shape = MaskingHack_output_shape))

	model.add( Masking( mask_value=0.) )

	lstm_layer = LSTM( int(o.nLSTMNodes), return_sequences=False)
	model.add(lstm_layer)
	model.add(Dropout(0.2))

	if int(o.nLSTMClass) ==2:
		model.add(Dense(1))
		model.add(Activation('sigmoid'))
	if int(o.nLSTMClass) ==4:
		model.add(Dense(4))
		model.add(Activation('softmax'))

	# try using different optimizers and different optimizer configs
	print "Compiling ..."
	if useAdam:
		if int(o.nLSTMClass)==2:
			model.compile(loss='binary_crossentropy', optimizer='adam', metrics=["accuracy"])
		else:
			model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=["accuracy"])
	else:
		if int(o.nLSTMClass)==2:
			model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=["accuracy"])
		else:
			model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=["accuracy"])

	print "Finish Compilation"

	print("Train...")
	history = model.fit( X_train_vec , y_train, batch_size=batch_size, nb_epoch=nb_epoch, validation_split=0.1, shuffle = True)
	print "Finish Training"

	return (model, history)


def buildModel_SimpleDense(dataset, useAdam=True):

	print "Building Model Dense IP3D..."

	#################
	# Configuration #
	#################

	X_train = dataset['X_train']
	y_train = dataset['y_train']
	labels_train = dataset['labels_train']
	labels_test  = dataset['labels_test']

	# split by "continuous variable" and "categorization variable"

	X_train_vec =   [X_train[:, 0:ntrk_cut, 0:2], X_train[:, 0:ntrk_cut,-1]]

	left = Sequential()
	left.add( Activation('linear', input_shape=(ntrk_cut, 2) ) )

	right = Sequential()
	right.add(Embedding(max_embed_features, embed_size, mask_zero=False, input_length=ntrk_cut))

	model = Sequential()
	model.add( Merge([left, right],mode='concat') )
	model.add(Flatten())

	model.add(Dense(128) )
	model.add(Activation('relu'))
	model.add(Dropout(0.2))

	model.add(Dense(64))
	model.add(Activation('relu'))
	model.add(Dropout(0.2))

	model.add(Dense(1))
	model.add(Activation('sigmoid'))

	# try using different optimizers and different optimizer configs
	print "Compiling ..."
	if useAdam:
		model.compile(loss='binary_crossentropy', optimizer='adam', metrics=["accuracy"])
	else:
		model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=["accuracy"]) ## sgd
	print "Finish Compilation"

	print("Train...")
	history = model.fit( X_train_vec , y_train, batch_size=batch_size, nb_epoch=nb_epoch, validation_split=0.2, shuffle = True)
	print "Finish Training"

	return (model, history)


def buildModel_RNNPlus(dataset, useAdam=True):

	print "Building Model RNN plus MV2"

	#################
	# Configuration #
	#################
	X_train = dataset['X_train']
	y_train = dataset['y_train']
	labels_train = dataset['labels_train']

	# split by "continuous variable" and "categorization variable"
	X_train_vec_dR     = [X_train[:,:,0:4], X_train[:,:,-1]]
	X_concat = None
	pred = None

	if int(o.nLSTMClass) ==2:
		RNNmodel = model_from_json(open( 'V47_LSTM_dR_40epoch_5000kEvts_0nTrackCut_15nMaxTrack_2nLSTMClass_25nLSTMNodes_CMix_architecture.json').read())
		RNNmodel.load_weights( 'V47_LSTM_dR_40epoch_5000kEvts_0nTrackCut_15nMaxTrack_2nLSTMClass_25nLSTMNodes_CMix_model_weights.h5' )
		RNNmodel.compile(loss='binary_crossentropy', optimizer='adam', metrics=["accuracy"])

		pred = RNNmodel.predict( X_train_vec_dR, batch_size)
		X_concat = np.ndarray(shape=( pred[:,0].shape[0], 2))

	if int(o.nLSTMClass) ==4:
		RNNmodel = model_from_json(open( 'V47_LSTM_dR_40epoch_5000kEvts_0nTrackCut_15nMaxTrack_4nLSTMClass_25nLSTMNodes_CMix_architecture.json').read())
		RNNmodel.load_weights( 'V47_LSTM_dR_40epoch_5000kEvts_0nTrackCut_15nMaxTrack_4nLSTMClass_25nLSTMNodes_CMix_model_weights.h5' )
		RNNmodel.compile(loss='binary_crossentropy', optimizer='adam', metrics=["accuracy"])

		pred = RNNmodel.predict( X_train_vec_dR, batch_size)

		X_concat = np.ndarray(shape=( pred[:, 0].shape[0], 5))

	for i in range(X_concat.shape[0]):
		if o.Model == "RNNPlusMV2":
			X_concat[i][0] = labels_train[i, 10]
		if o.Model == "RNNPlusSV1":
			X_concat[i][0] = labels_train[i, 9]

		for j in range(pred.shape[1]):
			X_concat[i][j+1] = pred[i, j]


	model = Sequential()
	if (int(o.nLSTMClass)==2):
		model.add(Dense(10, input_dim=(2)) )
	if (int(o.nLSTMClass)==4):
		model.add(Dense(10, input_dim=(5)) )

	model.add(Activation('relu'))
	model.add(Dropout(0.2))

	model.add(Dense(1))
	model.add(Activation('sigmoid'))

	# try using different optimizers and different optimizer configs
	print "Compiling ..."
	if useAdam:
		model.compile(loss='binary_crossentropy', optimizer='adam', metrics=["accuracy"])
	else:
		model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=["accuracy"])
	print "Finish Compilation"

	print("Train...")
	history = model.fit( X_concat , y_train, batch_size=batch_size, nb_epoch=nb_epoch, validation_split=0.2, show_accuracy=True, shuffle = True)
	print "Finish Training"

	return (model, history)


def generateOutput():
	dataset = makeData (Variables = "dR")
	labels = dataset['labels']
	labels = labels[labels[:,7]>=2]

	outfile = file("MV2C20_arr.pkl", 'wb')
	cPickle.dump(labels[:,8], outfile, protocol=cPickle.HIGHEST_PROTOCOL)

	print labels.shape
	

def saveModel(fileNameBase, model, history = None):

	json_string = model.to_json()
	print 'base ', fileNameBase
	open(fileNameBase+"_architecture.json", 'w').write(json_string)
	model.save_weights(fileNameBase + '_model_weights.h5', overwrite=True)


def evalModel(dataset, model, modelname):
	#################
	# Configuration #
	#################

	#################

	X_test = dataset['X_test']
	y_test = dataset['y_test']

	labels_test = dataset['labels_test']

	# split by "continuous variable" and "categorization variable"
	X_test_vec  = [X_test[:,:,0:-1],   X_test[:,:, -1]]

	if o.Model == "DenseIP3D":
		X_test_vec  = [  X_test [:, 0:ntrk_cut, 0:2], X_test [:, 0:ntrk_cut,-1]]

	if o.Model == "RNNPlusMV2" or o.Model == "RNNPlusSV1":

		X_test_vec_dR = [X_test[:,:,0:4], X_test[:,:,-1]]
		X_test_vec = None
		pred = None

		if int(o.nLSTMClass) ==2:
			RNNmodel = model_from_json(open( 'V47_LSTM_dR_40epoch_5000kEvts_0nTrackCut_15nMaxTrack_2nLSTMClass_25nLSTMNodes_CMix_architecture.json').read())
			RNNmodel.load_weights( 'V47_LSTM_dR_40epoch_5000kEvts_0nTrackCut_15nMaxTrack_2nLSTMClass_25nLSTMNodes_CMix_model_weights.h5' )
			RNNmodel.compile(loss='binary_crossentropy', optimizer='adam', metrics=["accuracy"])

			pred = RNNmodel.predict( X_test_vec_dR, batch_size)
			X_test_vec = np.ndarray(shape=( pred[:,0].shape[0], 2))

		if int(o.nLSTMClass) ==4:
			RNNmodel = model_from_json(open( 'V47_LSTM_dR_40epoch_5000kEvts_0nTrackCut_15nMaxTrack_4nLSTMClass_25nLSTMNodes_CMix_architecture.json').read())
			RNNmodel.load_weights( 'V47_LSTM_dR_40epoch_5000kEvts_0nTrackCut_15nMaxTrack_4nLSTMClass_25nLSTMNodes_CMix_model_weights.h5' )
			RNNmodel.compile(loss='binary_crossentropy', optimizer='adam', metrics=["accuracy"])

			pred = RNNmodel.predict( X_test_vec_dR, batch_size)
			X_test_vec = np.ndarray(shape=( pred[:, 0].shape[0], 5))


		for i in range(X_test_vec.shape[0]):
			if o.Model == "RNNPlusMV2":
				X_test_vec[i][0] = labels_test[i, 10]
			if o.Model == "RNNPlusSV1":
				X_test_vec[i][0] = labels_test[i, 9]

			for j in range(pred.shape[1]):
				X_test_vec[i][j+1] = pred[i, j]


	score = model.evaluate(X_test_vec, y_test, batch_size=batch_size)
	print('Test score:', score)

	classes = model.predict_classes(X_test_vec, batch_size=batch_size)
	acc = np_utils.accuracy(classes, y_test)
	print('Test accuracy:', acc)

	acc = np_utils.accuracy(classes[labels_test[:,0]==5], y_test[labels_test[:,0]==5])
	print('Test b accuracy:', acc)

	acc = np_utils.accuracy(classes[labels_test[:,0]==0], y_test[labels_test[:,0]==0])
	print('Test l accuracy:', acc)

	pred = model.predict(X_test_vec, batch_size=batch_size)
	return model


def BuildModel():

	#global dataset_storage,model_storage,history_storage

	dataset = makeData (Variables = o.Variables)
	dataset_storage = dataset

	model = None
	history = None
	modelname = "" 
	print o.Model
	if o.Model == "LSTM":
		model, history = buildModel_1hidden(dataset, True)
	if o.Model == "DenseIP3D":
		model, history = buildModel_SimpleDense(dataset, False)
	print ' ------------------------------------------'
	print o.Model
	if o.Model == "RNNPlusMV2" or o.Model == "RNNPlusSV1":
		model, history = buildModel_RNNPlus(dataset, useAdam=True)


	modelname = "V47_" + o.Model + "_"+ o.Variables + "_" + o.nEpoch + "epoch_" + str( n_events/1000) + 'kEvts_' + str( o.nTrackCut) + 'nTrackCut_' +  o.nMaxTrack + "nMaxTrack_" + o.nLSTMClass +"nLSTMClass_" + o.nLSTMNodes +"nLSTMNodes"

	model = evalModel(dataset, model, o.Model)
	
	if o.TrackOrder == 'pT':
		modelname += "_SortpT"
	if o.doTrainC == 'y':
		modelname += "_CMix"
	if o.AddJetpT == 'y':
		modelname += '_AddJetpT'

	#modelname = "test"
	
	saveModel(modelname, model, history)


def compareROC():

	dataset_phi = makeData( Variables = "phi", padding = "pre", nLSTMClass=2)
	dataset_dR = makeData( Variables = "dR", padding = "pre" , nLSTMClass=2)
	dataset_d0z0 = makeData( Variables = "d0z0", padding = "pre" , nLSTMClass=2)

	############################
	labels_test_phi = dataset_phi['labels_test']
	ip3d_test = labels_test_phi[:,3]
	ntrk_test = labels_test_phi[:,7]
	MV2_test = labels_test_phi[:,10]
	SV1_test = labels_test_phi[:,9]
	pt_test = labels_test_phi[:,1]

	X_test_phi = dataset_phi['X_test']
	y_test_phi = dataset_phi['y_test']
	X_test_vec_phi     = [X_test_phi[:,:,0:5], X_test_phi[:,:,5]]

	X_test_d0z0 = dataset_d0z0['X_test']
	y_test_d0z0 = dataset_d0z0['y_test']
	X_test_vec_d0z0     = [X_test_d0z0[:,:,0:5], X_test_d0z0[:,:,5]]
	labels_test_d0z0 = dataset_d0z0['labels_test']

	X_test_dR = dataset_dR['X_test']
	y_test_dR = dataset_dR['y_test']
	X_test_vec_dR     = [X_test_dR[:,:,0:4], X_test_dR[:,:,4]]
	X_test_vec_IP3D     = [X_test_dR[:,:,0:2], X_test_dR[:,:,4]]
	labels_test_dR = dataset_dR['labels_test']

	RNNmodel = model_from_json(open( 'V47_LSTM_IP3D_40epoch_5000kEvts_0nTrackCut_15nMaxTrack_2nLSTMClass_25nLSTMNodes_CMix_architecture.json').read())
	RNNmodel.load_weights( 'V47_LSTM_IP3D_40epoch_5000kEvts_0nTrackCut_15nMaxTrack_2nLSTMClass_25nLSTMNodes_CMix_model_weights.h5' )
	RNNmodel.compile(loss='binary_crossentropy', optimizer='adam', metrics=["accuracy"])
	RNNpred = RNNmodel.predict( X_test_vec_IP3D, batch_size)

	RNNmodel_dR = model_from_json(open( 'V47_LSTM_dR_40epoch_5000kEvts_0nTrackCut_15nMaxTrack_2nLSTMClass_25nLSTMNodes_CMix_architecture.json').read())
	RNNmodel_dR.load_weights( 'V47_LSTM_dR_40epoch_5000kEvts_0nTrackCut_15nMaxTrack_2nLSTMClass_25nLSTMNodes_CMix_model_weights.h5' )
	RNNmodel_dR.compile(loss='binary_crossentropy', optimizer='adam', metrics=["accuracy"])
	RNNpred_dR = RNNmodel_dR.predict( X_test_vec_dR, batch_size)

	RNNmodel_d0z0 = model_from_json(open( 'V47_LSTM_d0z0_40epoch_5000kEvts_0nTrackCut_15nMaxTrack_2nLSTMClass_25nLSTMNodes_CMix_architecture.json').read())
	RNNmodel_d0z0.load_weights( 'V47_LSTM_d0z0_40epoch_5000kEvts_0nTrackCut_15nMaxTrack_2nLSTMClass_25nLSTMNodes_CMix_model_weights.h5' )
	RNNmodel_d0z0.compile(loss='binary_crossentropy', optimizer='adam', metrics=["accuracy"])
	RNNpred_d0z0 = RNNmodel_d0z0.predict( X_test_vec_d0z0, batch_size)

	RNNmodel_phi = model_from_json(open( 'V47_LSTM_phi_40epoch_5000kEvts_0nTrackCut_15nMaxTrack_2nLSTMClass_25nLSTMNodes_CMix_architecture.json').read())
	RNNmodel_phi.load_weights( 'V47_LSTM_phi_40epoch_5000kEvts_0nTrackCut_15nMaxTrack_2nLSTMClass_25nLSTMNodes_CMix_model_weights.h5' )
	RNNmodel_phi.compile(loss='binary_crossentropy', optimizer='adam', metrics=["accuracy"])
	RNNpred_phi = RNNmodel_phi.predict( X_test_vec_phi, batch_size)

	plottingUtils.getROC( [ip3d_test[labels_test_phi[:,0]==5], SV1_test[labels_test_phi[:,0]==5], MV2_test[labels_test_phi[:,0]==5], 
			       RNNpred[labels_test_dR[:,0]==5, 0], RNNpred_dR[labels_test_dR[:,0]==5, 0],   RNNpred_phi[labels_test_phi[:,0]==5, 0], RNNpred_d0z0[labels_test_d0z0[:,0]==5, 0]],
			      [ip3d_test[labels_test_phi[:,0]==0], SV1_test[labels_test_phi[:,0]==0], MV2_test[labels_test_phi[:,0]==0], 
			       RNNpred[labels_test_dR[:,0]==0, 0], RNNpred_dR[labels_test_dR[:,0]==0, 0],   RNNpred_phi[labels_test_phi[:,0]==0, 0], RNNpred_d0z0[labels_test_d0z0[:,0]==0, 0]],
			      label=["IP3D", "SV1", "MV2", "RNN", "RNN (w/ pTFrac and dR)", "RNN (w/ pTFrac, dphi and deta)", "RNN (w/ pTFrac, d0 and z0)"],
			      outputName="ROC_RNN_BL.root", Rejection ="l" )

	plottingUtils.getROC( [ip3d_test[labels_test_phi[:,0]==5], SV1_test[labels_test_phi[:,0]==5], MV2_test[labels_test_phi[:,0]==5], 
			       RNNpred[labels_test_dR[:,0]==5, 0], RNNpred_dR[labels_test_dR[:,0]==5, 0],   RNNpred_phi[labels_test_phi[:,0]==5, 0], RNNpred_d0z0[labels_test_d0z0[:,0]==5, 0]],
			      [ip3d_test[labels_test_phi[:,0]==4], SV1_test[labels_test_phi[:,0]==4], MV2_test[labels_test_phi[:,0]==4], 
			       RNNpred[labels_test_dR[:,0]==4, 0], RNNpred_dR[labels_test_dR[:,0]==4, 0],   RNNpred_phi[labels_test_phi[:,0]==4, 0], RNNpred_d0z0[labels_test_d0z0[:,0]==4, 0]],
			      label=["IP3D", "SV1", "MV2", "RNN", "RNN (w/ pTFrac and dR)", "RNN (w/ pTFrac, dphi and deta)", "RNN (w/ pTFrac, d0 and z0)"],
			      outputName="ROC_RNN_BC.root", Rejection ="c" )

	RNNmodel_mc = model_from_json(open( 'V47_LSTM_IP3D_40epoch_5000kEvts_0nTrackCut_15nMaxTrack_4nLSTMClass_25nLSTMNodes_CMix_architecture.json').read())
	RNNmodel_mc.load_weights( 'V47_LSTM_IP3D_40epoch_5000kEvts_0nTrackCut_15nMaxTrack_4nLSTMClass_25nLSTMNodes_CMix_model_weights.h5' )
	RNNmodel_mc.compile(loss='categorical_crossentropy', optimizer='adam', metrics=["accuracy"])
	RNNpred_mc = RNNmodel_mc.predict( X_test_vec_IP3D, batch_size)

	RNNmodel_dR_mc = model_from_json(open( 'V47_LSTM_dR_40epoch_5000kEvts_0nTrackCut_15nMaxTrack_4nLSTMClass_25nLSTMNodes_CMix_architecture.json').read())
	RNNmodel_dR_mc.load_weights( 'V47_LSTM_dR_40epoch_5000kEvts_0nTrackCut_15nMaxTrack_4nLSTMClass_25nLSTMNodes_CMix_model_weights.h5' )
	RNNmodel_dR_mc.compile(loss='categorical_crossentropy', optimizer='adam', metrics=["accuracy"])
	RNNpred_dR_mc = RNNmodel_dR_mc.predict( X_test_vec_dR, batch_size)

	RNNmodel_d0z0_mc = model_from_json(open( 'V47_LSTM_d0z0_40epoch_5000kEvts_0nTrackCut_15nMaxTrack_4nLSTMClass_25nLSTMNodes_CMix_architecture.json').read())
	RNNmodel_d0z0_mc.load_weights( 'V47_LSTM_d0z0_40epoch_5000kEvts_0nTrackCut_15nMaxTrack_4nLSTMClass_25nLSTMNodes_CMix_model_weights.h5' )
	RNNmodel_d0z0_mc.compile(loss='categorical_crossentropy', optimizer='adam', metrics=["accuracy"])
	RNNpred_d0z0_mc = RNNmodel_d0z0_mc.predict( X_test_vec_d0z0, batch_size)

	RNNmodel_phi_mc = model_from_json(open( 'V47_LSTM_phi_40epoch_5000kEvts_0nTrackCut_15nMaxTrack_4nLSTMClass_25nLSTMNodes_CMix_architecture.json').read())
	RNNmodel_phi_mc.load_weights( 'V47_LSTM_phi_40epoch_5000kEvts_0nTrackCut_15nMaxTrack_4nLSTMClass_25nLSTMNodes_CMix_model_weights.h5' )
	RNNmodel_phi_mc.compile(loss='categorical_crossentropy', optimizer='adam', metrics=["accuracy"])
	RNNpred_phi_mc = RNNmodel_phi_mc.predict( X_test_vec_phi, batch_size)

	RNNpred_dR_llh   = np.log(RNNpred_dR_mc[:,0]/RNNpred_dR_mc[:,2])
	RNNpred_d0z0_llh = np.log(RNNpred_d0z0_mc[:,0]/RNNpred_d0z0_mc[:,2])
	RNNpred_phi_llh  = np.log(RNNpred_phi_mc[:,0]/RNNpred_phi_mc[:,2])
	RNNpred_llh  = np.log(RNNpred_mc[:,0]/RNNpred_mc[:,2])

	RNNpred_dR_llh_c   = np.log(RNNpred_dR_mc[:,0]/ (RNNpred_dR_mc[:,2]*0.55+RNNpred_dR_mc[:,1]*0.1))
	RNNpred_d0z0_llh_c = np.log(RNNpred_d0z0_mc[:,0]/(RNNpred_d0z0_mc[:,2]*0.55+RNNpred_d0z0_mc[:,1]*0.1))
	RNNpred_phi_llh_c  = np.log(RNNpred_phi_mc[:,0]/(RNNpred_phi_mc[:,2]*0.55+RNNpred_phi_mc[:,1]*0.1))
	RNNpred_llh_c  = np.log(RNNpred_mc[:,0]/(RNNpred_mc[:,2]*0.55+RNNpred_mc[:,1]*0.1))

	plottingUtils.getROC( [ ip3d_test[labels_test_phi[:,0]==5], SV1_test[labels_test_phi[:,0]==5], MV2_test[labels_test_phi[:,0]==5], 
				RNNpred_llh[labels_test_dR[:,0]==5], RNNpred_dR_llh[labels_test_dR[:,0]==5],   RNNpred_d0z0_llh[labels_test_phi[:,0]==5], RNNpred_phi_llh[labels_test_d0z0[:,0]==5]],
			      [ ip3d_test[labels_test_phi[:,0]==0], SV1_test[labels_test_phi[:,0]==0], MV2_test[labels_test_phi[:,0]==0], 
				RNNpred_llh[labels_test_dR[:,0]==0], RNNpred_dR_llh[labels_test_dR[:,0]==0],   RNNpred_d0z0_llh[labels_test_phi[:,0]==0], RNNpred_phi_llh[labels_test_d0z0[:,0]==0]],
			      label=["IP3D", "SV1", "MV2", "RNN", "RNN (w/ pTFrac and dR)", "RNN (w/ pTFrac, dphi and deta)", "RNN (w/ pTFrac, d0 and z0)"],
			      outputName="ROC_RNN_BL_MultiClass_2ClassLLH.root", Rejection ="l" )

	plottingUtils.getROC( [ ip3d_test[labels_test_phi[:,0]==5], SV1_test[labels_test_phi[:,0]==5], MV2_test[labels_test_phi[:,0]==5], 
				RNNpred_llh_c[labels_test_dR[:,0]==5], RNNpred_dR_llh_c[labels_test_dR[:,0]==5],   RNNpred_d0z0_llh_c[labels_test_phi[:,0]==5], RNNpred_phi_llh_c[labels_test_d0z0[:,0]==5]],
			      [ ip3d_test[labels_test_phi[:,0]==0], SV1_test[labels_test_phi[:,0]==0], MV2_test[labels_test_phi[:,0]==0], 
				RNNpred_llh_c[labels_test_dR[:,0]==0], RNNpred_dR_llh_c[labels_test_dR[:,0]==0],   RNNpred_d0z0_llh_c[labels_test_phi[:,0]==0], RNNpred_phi_llh_c[labels_test_d0z0[:,0]==0]],
			      label=["IP3D", "SV1", "MV2", "RNN", "RNN (w/ pTFrac and dR)", "RNN (w/ pTFrac, dphi and deta)", "RNN (w/ pTFrac, d0 and z0)"],
			      outputName="ROC_RNN_BL_MultiClass_3ClassLLH.root", Rejection ="l" ) 

	plottingUtils.getROC( [ ip3d_test[labels_test_phi[:,0]==5], SV1_test[labels_test_phi[:,0]==5], MV2_test[labels_test_phi[:,0]==5], 
				RNNpred_llh[labels_test_dR[:,0]==5], RNNpred_dR_llh[labels_test_dR[:,0]==5],   RNNpred_d0z0_llh[labels_test_phi[:,0]==5], RNNpred_phi_llh[labels_test_d0z0[:,0]==5]],
			      [ ip3d_test[labels_test_phi[:,0]==4], SV1_test[labels_test_phi[:,0]==4], MV2_test[labels_test_phi[:,0]==4], 
				RNNpred_llh[labels_test_dR[:,0]==4], RNNpred_dR_llh[labels_test_dR[:,0]==4],   RNNpred_d0z0_llh[labels_test_phi[:,0]==4], RNNpred_phi_llh[labels_test_d0z0[:,0]==4]],
			      label=["IP3D", "SV1", "MV2", "RNN", "RNN (w/ pTFrac and dR)", "RNN (w/ pTFrac, dphi and deta)", "RNN (w/ pTFrac, d0 and z0)"],
			      outputName="ROC_RNN_BC_MultiClass_2ClassLLH.root", Rejection ="c" )

	plottingUtils.getROC( [ ip3d_test[labels_test_phi[:,0]==5], SV1_test[labels_test_phi[:,0]==5], MV2_test[labels_test_phi[:,0]==5], 
				RNNpred_llh_c[labels_test_dR[:,0]==5], RNNpred_dR_llh_c[labels_test_dR[:,0]==5],   RNNpred_d0z0_llh_c[labels_test_phi[:,0]==5], RNNpred_phi_llh_c[labels_test_d0z0[:,0]==5]],
			      [ ip3d_test[labels_test_phi[:,0]==4], SV1_test[labels_test_phi[:,0]==4], MV2_test[labels_test_phi[:,0]==4], 
				RNNpred_llh_c[labels_test_dR[:,0]==4], RNNpred_dR_llh_c[labels_test_dR[:,0]==4],   RNNpred_d0z0_llh_c[labels_test_phi[:,0]==4], RNNpred_phi_llh_c[labels_test_d0z0[:,0]==4]],
			      label=["IP3D", "SV1", "MV2", "RNN", "RNN (w/ pTFrac and dR)", "RNN (w/ pTFrac, dphi and deta)", "RNN (w/ pTFrac, d0 and z0)"],
			      outputName="ROC_RNN_BC_MultiClass_3ClassLLH.root", Rejection ="c" ) 


	X_test_RNNMV2 = np.ndarray(shape=( RNNpred_mc.shape[0], 5))
	X_test_RNNSV1 = np.ndarray(shape=( RNNpred_mc.shape[0], 5))

	for i in range(RNNpred_mc.shape[0]):
		X_test_RNNMV2[i][0] = labels_test_phi[i, 10]
		X_test_RNNSV1[i][0] = labels_test_phi[i, 9]

		for j in range(RNNpred_mc.shape[1]):
			X_test_RNNMV2[i][j+1] = RNNpred_mc[i, j]
			X_test_RNNSV1[i][j+1] = RNNpred_mc[i, j]


	#############################
	RNNPlusSV1model = model_from_json(open( 'V47_RNNPlusSV1_dR_20epoch_5000kEvts_0nTrackCut_15nMaxTrack_4nLSTMClass_25nLSTMNodes_CMix_architecture.json').read())
	RNNPlusSV1model.load_weights( 'V47_RNNPlusSV1_dR_20epoch_5000kEvts_0nTrackCut_15nMaxTrack_4nLSTMClass_25nLSTMNodes_CMix_model_weights.h5')
	RNNPlusSV1model.compile(loss='binary_crossentropy', optimizer='adam', metrics=["accuracy"])

	RNNPlusSV1pred = RNNPlusSV1model.predict( X_test_RNNSV1, batch_size)

	RNNPlusMV2model = model_from_json(open( 'V47_RNNPlusMV2_dR_20epoch_5000kEvts_0nTrackCut_15nMaxTrack_4nLSTMClass_25nLSTMNodes_CMix_architecture.json').read())
	RNNPlusMV2model.load_weights( 'V47_RNNPlusMV2_dR_20epoch_5000kEvts_0nTrackCut_15nMaxTrack_4nLSTMClass_25nLSTMNodes_CMix_model_weights.h5')
	RNNPlusMV2model.compile(loss='binary_crossentropy', optimizer='adam', metrics=["accuracy"])

	RNNPlusMV2pred = RNNPlusMV2model.predict( X_test_RNNMV2, batch_size)


	plottingUtils.getROC( [ ip3d_test[labels_test_phi[:,0]==5], SV1_test[labels_test_phi[:,0]==5], MV2_test[labels_test_phi[:,0]==5],
				RNNpred_dR_llh[labels_test_phi[:,0]==5],   RNNPlusSV1pred[labels_test_phi[:,0]==5, 0], RNNPlusMV2pred[labels_test_phi[:,0]==5, 0]],
			      [ ip3d_test[labels_test_phi[:,0]==0], SV1_test[labels_test_phi[:,0]==0], MV2_test[labels_test_phi[:,0]==0],
				RNNpred_dR_llh[labels_test_phi[:,0]==0],   RNNPlusSV1pred[labels_test_phi[:,0]==0, 0], RNNPlusMV2pred[labels_test_phi[:,0]==0, 0]],

			      label=["IP3D", "SV1", "MV2", "RNN (w/ pTFrac and dR)", "RNN (w/ pTFrac and dR)+SV1", "RNN (w/ pTFrac and dR)+MV2"],
			      outputName="ROC_BL_RNNPlus.root", Rejection ="l" )


#        print "Making b jet efficiency v.s. pT curve ... "
#        plottingUtils.MultipleEffCurve(                                 
#		outputName = "BEffCurveCompare_pT.root",                                                                                                                                    
#		approachList = [                                 
#			(ip3d_test[labels_test[:,0]==5], pt_test[labels_test[:,0]==5], ("EffCurve_IP3D", "IP3D Signal Efficiency")),                                         
#			(SV1_test[labels_test[:,0]==5], pt_test[labels_test[:,0]==5], ("EffCurve_SV1", "SV1 Signal Efficiency")),                                         
#			(MV2_test[labels_test[:,0]==5], pt_test[labels_test[:,0]==5], ("EffCurve_MV2", "MV2 Signal Efficiency")),                                         
#			(RNNpred[labels_test[:,0]==5,0], pt_test[labels_test[:,0]==5], ("EffCurve_RNN1HiddenLSTM", "RNN-1HiddenLSTM")),                      
#			(RNNPlusSV1pred[labels_test[:,0]==5,0], pt_test[labels_test[:,0]==5], ("EffCurve_RNN1HiddenLSTMSV1", "RNN-1HiddenLSTM+SV1")),                      
#			(RNNPlusMV2pred[labels_test[:,0]==5,0], pt_test[labels_test[:,0]==5], ("EffCurve_RNN1HiddenLSTMMV2", "RNN-1HiddenLSTM+MV2")), ],
#		bins = [20, 50, 80, 120, 200, 300, 500, 800],
#		eff_target = 0.7,)


#        print "Making b jet efficiency v.s. ntrk curve ... "
#        plottingUtils.MultipleEffCurve(                                 
#		outputName = "BEffCurveCompare_ntrk.root",                                                                                                                                    
#		approachList = [                                 
#			(ip3d_test[labels_test[:,0]==5], ntrk_test[labels_test[:,0]==5], ("EffCurve_IP3D", "IP3D Signal Efficiency")),                                         
#			(pred1[labels_test[:,0]==5,0], ntrk_test[labels_test[:,0]==5], ("EffCurve_RNN1HiddenLSTM", "RNN-1HiddenLSTM")),                      
#			(pred2[labels_test[:,0]==5,0], ntrk_test[labels_test[:,0]==5], ("EffCurve_RNN1HiddenLSTMpTFrac", "RNN-1HiddenLSTM (w/ pT frac)")),                      
#			(pred3[labels_test[:,0]==5,0], ntrk_test[labels_test[:,0]==5], ("EffCurve_RNN1HiddenLSTMpTFracdR", "RNN-1HiddenLSTM (w/ pT frac and dR)")), ],
#		bins = range(1, 42),
#		eff_target = 0.7,)



#	bins = [20, 50, 80, 120, 200, 300, 500, 800]
#        def getScoreCutList(scoreList):
#                return plottingUtils.getFixEffCurve(
#                                                     scoreList = scoreList,
#                                                     varList = pt_test[labels_test[:,0]==5]/1000.0,
#                                                     label = "IdontCare",
#                                                     bins = bins,
#                                                     fix_eff_target = 0.7,
#                                                     onlyReturnCutList = True
#                                                    )
#
#
#        scoreDB_signal = [ ip3d_test[labels_test[:,0]==5], SV1_test[labels_test[:,0]==5], MV2_test[labels_test[:,0]==5], RNNpred[labels_test[:,0]==5,0], RNNPlusSV1pred[labels_test[:,0]==5,0], RNNPlusMV2pred[labels_test[:,0]==5,0]]
#        scoreDB_bkg = [ ip3d_test[labels_test[:,0]==0], SV1_test[labels_test[:,0]==0], MV2_test[labels_test[:,0]==0], RNNpred[labels_test[:,0]==0,0], RNNPlusSV1pred[labels_test[:,0]==0,0], RNNPlusMV2pred[labels_test[:,0]==0,0]]
#
#	print "score db signal", scoreDB_signal
#
#        def DrawFlatEfficiencyCurves(flavorCode):
#                if flavorCode == 5:
#                        flavorLabel = "Signal Efficiency"
#                        scoreDB = scoreDB_signal
#                elif flavorCode == 0:
#                        flavorLabel = "Bkg Rejection"
#                        scoreDB = scoreDB_bkg
#                else:
#                        flavorLabel = "Unknown"
#                        scoreDB = None
#
#                varList = pt_test[labels_test[:,0]==flavorCode]
#		varList = varList/1000.
#
#                plottingUtils.MultipleFlatEffCurve(
#                                                   outputName = "ZihaoTest_FlatEffCurveCompare_Flavor%s.root" % (flavorCode),
#                                                   approachList = [
#                                                                    (scoreDB[0], varList, ("EffCurvePt_IP3D"     , "IP3D " + flavorLabel)       , getScoreCutList(scoreDB_signal[0])),
#                                                                    (scoreDB[1], varList, ("EffCurvePt_SV1"      , "SV1 " + flavorLabel)        , getScoreCutList(scoreDB_signal[1])),
#                                                                    (scoreDB[2], varList, ("EffCurvePt_MV2"      , "MV2 " + flavorLabel)        , getScoreCutList(scoreDB_signal[2])),
#                                                                    (scoreDB[3], varList, ("EffCurvePt_RNN"      , "RNN " + flavorLabel)          , getScoreCutList(scoreDB_signal[3])),
#                                                                    (scoreDB[4], varList, ("EffCurvePt_RNNSV1"   , "RNN+SV1 " + flavorLabel)          , getScoreCutList(scoreDB_signal[4])),
#                                                                    (scoreDB[5], varList, ("EffCurvePt_RNNMV2"   , "RNN+MV2 " + flavorLabel)          , getScoreCutList(scoreDB_signal[5])),
#                                                                  ],
#                                                   bins = bins,
#                                                  )
#	DrawFlatEfficiencyCurves(5)
#	DrawFlatEfficiencyCurves(0)


	
########################################

if __name__ == "__main__":
	if o.doBatch == "y":
		currentPWD = os.getcwd()

		modelname = "V47_" + o.Model + "_"+ o.Variables + "_" + o.nEpoch + "epoch_" + str( n_events/1000) + 'kEvts_' + str( o.nTrackCut) + 'nTrackCut_' +  o.nMaxTrack + "nMaxTrack_" + o.nLSTMClass +"nLSTMClass_" + o.nLSTMNodes +"nLSTMNodes"
		if o.TrackOrder == 'pT':
			modelname += "_SortpT"
		if o.doTrainC == 'y':
			modelname += '_CMix'
		if o.AddJetpT == 'y':
			modelname += '_AddJetpT'
		
		cmd = "bsub -q atlas-t3 -W 80:00 -o 'output/" + modelname + "' THEANO_FLAGS='base_compiledir=" + currentPWD + "/BatchCompileDir/0/' python lstmUtils.py --nEpoch " + o.nEpoch + " --Mode " + o.Mode + " --Var " + o.Variables + " --nEvents " + o.nEvents + " --doTrainC " + o.doTrainC + " --nMaxTrack " + o.nMaxTrack + " --TrackOrder " + o.TrackOrder + " --padding " + o.padding + " --Model " + o.Model + " --nTrackCut " + o.nTrackCut + " --AddJetpT " + o.AddJetpT + " --nLSTMClass " + o.nLSTMClass + " --nLSTMNodes " + o.nLSTMNodes

		print cmd
		os.system(cmd)

	else:
		if o.Mode == "M":
			BuildModel()
		if o.Mode == "C":
			compareROC()
		if o.Mode == "P":
			generateOutput()

