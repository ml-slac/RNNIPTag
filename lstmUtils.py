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
import random
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
p.add_option('--doLessC', type ="string", default= 'n', dest="doLessC", help="do less c")
p.add_option('--TrackOrder', type ="string", default= 'Sd0', dest="TrackOrder", help="Track Ordering [Sd0], [pT] more to be added")
p.add_option('--padding', type = "string", default = 'pre', dest="padding", help="padding order, pre or post")
p.add_option('--Model', type = "string", default = 'LSTM', dest="Model", help="Model type: LSTM, DenseIP3D")
p.add_option('--AddJetpT', type = "string", default = 'n', dest="AddJetpT", help="if add jet pT to the model of RNN+")
p.add_option('--nLSTMNodes', type = "string", default = '25', dest="nLSTMNodes", help="number of hidden nodes for the LSTM algorithm")
p.add_option('--nLSTMClass', type = "string", default = '2', dest="nLSTMClass", help="the number of output classes")
p.add_option('--nLayers', type = "string", default = '1', dest="nLayers", help="number of hidden layers")
p.add_option('--Filebase', type = "string", default = 'None', dest="filebase", help="filebase of trained model")
p.add_option('--EmbedSize', type = "string", default = '2', dest="EmbedSize", help="embedding size")
p.add_option('--doJetpTReweight', type = "string", default = 'n', dest="doJetpTReweight", help="reweight jet pT")
p.add_option('--Version', type = "string", default = 'V51', dest="Version", help="version of input file")

(o,a) = p.parse_args()

nb_epoch = int(o.nEpoch)
max_len = int(o.nMaxTrack)
batch_size = 128
n_events = int(o.nEvents)
trainFraction = 0.8
max_embed_features = 16
embed_size = int(o.EmbedSize)
ntrk_cut = int(o.nTrackCut)

SavedModels ={}

class Models:
	def __init__(self, filebase, pred, label, val_loss,   loss):
		self.filebase = filebase
		self.pred = pred
		self.val_loss = val_loss
		self.loss = loss
		self.label = label


def LoadModel(filebase, testvec, label, simpleBuild=False ,loss = "binary_crossentropy"):
	if simpleBuild:
		model = model_from_json(open( filebase+'_architecture.json').read())
		model.load_weights(filebase + '_model_weights.h5')
	else:
        #NOTE only for models coming from build_1hidden
		#print "testvec shape", testvec.shape
		n_cont_vars = 0
		if "dR" in filebase:
			n_cont_vars = testvec[0].shape[2]
		if "hits" in filebase:
			n_cont_vars = testvec.shape[1]
		model = _buildModel_1hidden(n_cont_vars)
		model.load_weights(filebase + '_model_weights.h5', by_name=True)
        
	model.compile(loss =loss , optimizer= 'adam', metrics=["accuracy"])
    
	pred = model.predict( testvec, batch_size)

	if "4n" in filebase:
		pred = np.log(pred[:,0]/(0.9*pred[:,2] + 0.1*pred[:,1]))
	    
	f = open(filebase+"_history.json", "r")
	history = cPickle.load(f)
	train_hist = history["loss"]
	test_hist = history["val_loss"]

	SavedModels[filebase] = Models(filebase, pred, label, test_hist, train_hist)


def makeData( Variables = "IP3D", max_len=max_len, padding= o.padding, nLSTMClass = o.nLSTMClass, TrackOrder = o.TrackOrder): 
	print "Getting Data ..."

	folder_name = '/u/at/zihaoj/nfs2/RNNIPTag/MakeData/'
        
	f = None
	if TrackOrder == "Sd0" and Variables == "phi":
		f = file(folder_name+'Dataset_'+o.Version+'_IP3D_pTFrac_dphi_deta_5m.pkl','r')
	if TrackOrder == "Sd0" and Variables == "dtheta":
		f = file(folder_name+'Dataset_'+o.Version+'_IP3D_pTFrac_dphi_dtheta_5m.pkl','r')
	if TrackOrder == "Sd0" and Variables == "d0z0":
		f = file(folder_name+'Dataset_'+o.Version+'_IP3D_pTFrac_d0_z0_5m.pkl','r')

	if TrackOrder == "Sd0" and Variables == "dR":
		#f = file(folder_name+'Dataset_'+o.Version+'_IP3D_pTFrac_dR_5m.pkl','r')
		f = file(folder_name+'Dataset_'+o.Version+'_IP3D_pTFrac_dR_hits_3m.pkl', 'r')

	if TrackOrder == "Sd0" and Variables == "Hits":
		f = file(folder_name+'Dataset_'+o.Version+'_IP3D_pTFrac_dR_hits_3m.pkl','r')

	if TrackOrder == "Reverse" and Variables == "dR":
		f = file(folder_name+'Dataset_'+o.Version+'_IP3D_pTFrac_dR_reverse_sd0order_5m.pkl','r')

	if TrackOrder == "Sd0" and Variables == "IP3D":
		f = file(folder_name+'Dataset_'+o.Version+'_IP3D_pTFrac_dR_5m.pkl','r')

	if TrackOrder == "SL0":
		f = file(folder_name+'Dataset_'+o.Version+'_IP3D_pTFrac_dR_sl0order_hits_3m.pkl','r')

	if TrackOrder == "pT":
		f = file(folder_name+'Dataset_'+o.Version+'_IP3D_pTFrac_dR_pt_hits_3m.pkl','r')


        trk_arr_all = cPickle.load(f)
	labels_all = cPickle.load(f)
        #sv1_all = cPickle.load(f)

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

	if Variables == "Hits":
		X = TTA.MakePaddedSequenceTensorFromListArray( trk_arr_all[:,[0, 1, 2, 3, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17]], doWhitening=False, maxlen=max_len, padding = padding)	

	trk_grd = TTA.convertSequencesFromListArray( trk_arr_all[:,5], dopad=True, pad_value=-1, maxlen=max_len )

	if Variables == "dR" or Variables == "IP3D":
		trk_grd = TTA.convertSequencesFromListArray( trk_arr_all[:,4], dopad=True, pad_value=-1, maxlen=max_len )

	print "Getting Labels"
	print "padding ", padding

	if Variables == "Hits":
		X_all = X
	else:
		X_all = np.dstack( (X, trk_grd+1) )
		
	X = X_all[:n_events]
	labels = labels_all[:n_events]
	y = (labels[:,0] ==5)
	#sv1 = sv1_all[:n_events]

	if int(nLSTMClass) == 4 and ("LSTM" in o.Model or "GRU" in o.Model or "RNNSV1"==o.Model):
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
	
	## apply jet sellection
	# JVT >0.59 for jets with pT<60GeV and |eta|<2.4
	JVTCuts =  np.logical_or(labels[:,11]>0.59 ,np.absolute(labels[:,2])>2.4)
	JVTCuts =  np.logical_or(JVTCuts ,labels[:,1]>60000)
	# Jet pT cuts > 20GeV
	JetpTCut = labels[:,1]>20000
	# Jet |eta|<2.5
	JetEtaCut = np.absolute(labels[:,2])<2.5
	# Jet alive after OR
	JetEleVetoCut = (labels[:,12]==1)

	JetCuts = np.logical_and(JVTCuts, JetpTCut)
	JetCuts = np.logical_and(JetCuts, JetEtaCut)
	JetCuts = np.logical_and(JetCuts, JetEleVetoCut)

	X = X[ JetCuts]
	y = y[ JetCuts]
	labels = labels[JetCuts] 
	#sv1 = sv1[JetCuts]

	if o.doLessC == "y":
		X_firsthalf = X[0:int(n_events/2.0)]
		y_firsthalf = y[0:int(n_events/2.0)]
		labels_firsthalf  = labels[0:int(n_events/2.0)]
		#sv1_firsthalf     = sv1[0:int(n_events/2.0)]

		X_second = X[int(n_events/2.0):n_events]
		y_second = y[int(n_events/2.0):n_events]
		labels_second  = labels[int(n_events/2.0):n_events]
		#sv1_second     = sv1[int(n_events/2.0):n_events]

		X_second = X_second[labels_second[:,0]!=4]
		y_second = y_second[labels_second[:,0]!=4]
		labels_second  = labels_second[labels_second[:,0]!=4]
		#sv1_second     = sv1_second[labels_second[:,0]!=4]

		X = np.vstack((X_firsthalf, X_second))

		labels = np.vstack((labels_firsthalf, labels_second))
		if int(nLSTMClass) == 4:
			y = np.vstack((y_firsthalf, y_second))
		else :
			y = (labels[:,0]==5)
		#sv1 = np.vstack((sv1_firsthalf, sv1_second))


	weights = np.ones( X.shape[0])
	if o.doJetpTReweight == "y":
		
		upbound = 700
		step =10
		if o.Version =="V56" or o.Version == "V52":
			upbound = 2000
			step =50
		
		pt = labels[:,1]/1000.0
		print "max pt", max(pt)

		pt_b = pt[labels[:,0]==5]
		pt_c = pt[labels[:,0]==4]
		pt_l = pt[labels[:,0]==0]

		hist_b = np.histogram(pt_b, 3000/step, (0,3000))[0]
		hist_b = hist_b/float(np.sum(hist_b))
		hist_b += 0.00000001


		hist_c = np.histogram(pt_c, 3000/step, (0,3000))[0]
		hist_c = hist_c/float(np.sum(hist_c))
		hist_c += 0.00000001


		hist_l = np.histogram(pt_l, 3000/step, (0,3000))[0]
		hist_l = hist_l/float(np.sum(hist_l))
		hist_l += 0.00000001


		weight_b = hist_l/hist_b
		weight_c = hist_l/hist_c

		weight_b[upbound/10: weight_b.shape[0]-1] = 1
		weight_c[upbound/10: weight_c.shape[0]-1] = 1

		pt_bin = np.floor(pt/(float(step)))
		pt_bin.astype(int)

		for ijet in range(weights.shape[0]):
			if labels[ijet,0] ==0:
				continue
			if labels[ijet,0] ==5:
				weights[ijet] = weight_b[pt_bin[ijet]]
			if labels[ijet,0] ==4:
				weights[ijet] = weight_c[pt_bin[ijet]]

		print labels[:,0]
		print pt
		print weights
		print weight_b
		print weight_c
		
		
	X_train, X_test = np.split( X, [ int(trainFraction*X.shape[0]) ] )
	#sv1_train, sv1_test = np.split ( sv1, [ int(trainFraction*sv1.shape[0]) ] )
	y_train, y_test = np.split( y, [ int(trainFraction*y.shape[0]) ] )
	labels_train, labels_test = np.split( labels, [ int(trainFraction*labels.shape[0]) ] )
	weights_train, weights_test = np.split( weights, [ int(trainFraction*labels.shape[0]) ] )
	ip3d_test = labels_test[:,3]


	print("data shape",X.shape)
	print y_train.shape, y_test.shape


	dataset = {
	  "X": X,
	  "X_train": X_train,
	  "X_test": X_test,
	  #"sv1_train": sv1_train,
	  #"sv1_test": sv1_test,
	  "labels": labels,
	  "labels_train": labels_train,
	  "labels_test": labels_test,
	  "y": y,
	  "y_train": y_train,
	  "y_test": y_test,
	  "weights_train": weights_train,
	  "weights_test": weights_test
	}

	return dataset



def _buildModel_1hidden(n_cont_vars):
        
	_m = Masking( mask_value=0, input_shape = (max_len, n_cont_vars))
	left = Sequential()
	left.add(_m)
	#left.add( Activation('linear', input_shape=(max_len, n_cont_vars)) )
	#left.add( Masking( mask_value=0, input_shape = (max_len, n_cont_vars) ))
	#left.add( TimeDistributedPassThrough( input_shape=(max_len, n_cont_vars) ) )

	_e = Embedding(max_embed_features, embed_size, mask_zero=True, input_length=max_len, name="embedding_1")
	right = Sequential()
	right.add(_e)
	#right.add(Embedding(max_embed_features, embed_size, mask_zero=True, input_length=max_len))
	#right.add(Embedding(max_embed_features, embed_size, mask_zero=False, input_length=max_len))


	model = Sequential()
	
	if o.Variables != "Hits":
		#model.add( Merge([left, right],mode='concat') )
		model.add( Merge([_m, _e],mode='concat') )
		#model.add(Lambda(MaskingHack, output_shape = MaskingHack_output_shape))
	else:
		model.add( Masking( mask_value=0, input_shape = (max_len, n_cont_vars))  )
		#model.add(Lambda(MaskingHack, output_shape = MaskingHack_output_shape))
		
	#model.add( Masking( mask_value=0.) )


	if "LSTM" in o.Model:

		lstm_layer = LSTM( int(o.nLSTMNodes), return_sequences=False, name="lstm_1")
		lstm_layer_return_sequence = LSTM( int(o.nLSTMNodes), return_sequences=True, name="lstm_2")

		if o.nLayers == "1":
			model.add(lstm_layer)
			model.add(Dropout(0.2))

		if o.nLayers == "2":
			model.add(lstm_layer_return_sequence)
			model.add(Dropout(0.2))
			model.add(lstm_layer)
			model.add(Dropout(0.2))

		if "MoreDense" in o.Model:
			model.add(Dense(25,name="dense_2"))
			model.add(Activation('relu'))
			model.add(Dropout(0.2))

	if "GRU" in o.Model:

		gru_layer = GRU( int(o.nLSTMNodes), return_sequences=False,name="gru_1")
		gru_layer_return_sequence = GRU( int(o.nLSTMNodes), return_sequences=True,name="gru_2")

		if o.nLayers == "1":
			model.add(gru_layer)
			model.add(Dropout(0.2))

		if o.nLayers == "2":
			model.add(gru_layer_return_sequence)
			model.add(Dropout(0.2))
			model.add(gru_layer)
			model.add(Dropout(0.2))

		if "MoreDense" in o.Model:
			model.add(Dense(25,name="dense_2"))
			model.add(Activation('relu'))
			model.add(Dropout(0.2))

	if int(o.nLSTMClass) ==2:
		model.add(Dense(1,name="dense_1"))
		model.add(Activation('sigmoid'))

	if int(o.nLSTMClass) ==4:
		model.add(Dense(4,name="dense_1"))
		model.add(Activation('softmax'))
		
	return model


def buildModel_1hidden(dataset, useAdam=False):

	print "Building Model ..."

	#################
	# Configuration #
	#################

	X_train = dataset['X_train']
	y_train = dataset['y_train']

	sample_weight = dataset['weights_train']

	# split by "continuous variable" and "categorization variable"
	X_train_vec = [X_train[:,:,0:-1],  X_train[:,:,-1] ]
	if o.Variables == "Hits":
		X_train_vec = X_train

	if o.Variables == "Hits":
		n_cont_vars = X_train_vec.shape[2]
	else:
		n_cont_vars = X_train_vec[0].shape[2]

	print ' number of continuous input variables ', n_cont_vars
	
	##################
	print "shape ", X_train_vec[0].shape,  X_train_vec[1].shape

	model = _buildModel_1hidden( n_cont_vars = n_cont_vars )

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

	if o.Mode == "R":
		model = model_from_json(open( o.filebase+'_architecture.json').read())
		model.load_weights(o.filebase + '_model_weights.h5')
		if int(o.nLSTMClass)==2:
			model.compile(loss='binary_crossentropy', optimizer='adam', metrics=["accuracy"])
		else:
			model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=["accuracy"])

	history = model.fit( X_train_vec , y_train, batch_size=batch_size, nb_epoch=nb_epoch, validation_split=0.1, shuffle = True, sample_weight= sample_weight)
	print "Finish Training"

	return (model, history)


def buildModel_RNNSV1(dataset, useAdam=False):

	print "Building Model ..."

	#################
	# Configuration #
	#################

	X_train = dataset['X_train']
	sv1_train = dataset['sv1_train']
	y_train = dataset['y_train']
	sample_weight = dataset['weights_train']
	# split by "continuous variable" and "categorization variable"
	X_train_vec = [X_train[:,:,0:-1],  X_train[:,:,-1] ]

	n_cont_vars = X_train_vec[0].shape[2]
	print ' number of continuous input variables ', n_cont_vars
	print "shape ", X_train_vec[0].shape,  X_train_vec[1].shape

	X_train_vec.append( sv1_train)

	left = Sequential()
	#left.add( Activation('linear', input_shape=(max_len, n_cont_vars)) )
	left.add( Masking( mask_value=0, input_shape = (max_len, n_cont_vars) ))
	#left.add( TimeDistributedPassThrough( input_shape=(max_len, n_cont_vars) ) )

	right = Sequential()
	#right.add(Embedding(max_embed_features, embed_size, mask_zero=False, input_length=max_len))
	right.add(Embedding(max_embed_features, embed_size, mask_zero=True, input_length=max_len))

	intermediate = Sequential()
	intermediate.add( Merge([left, right],mode='concat') )

	#intermediate.add(Lambda(MaskingHack, output_shape = MaskingHack_output_shape))
	#intermediate.add( Masking( mask_value=0.) )

	model = Sequential()

	lstm_layer = LSTM( int(o.nLSTMNodes), return_sequences=False)
	lstm_layer_return_sequence = LSTM( int(o.nLSTMNodes), return_sequences=True)

	if o.nLayers == "1":
		intermediate.add(lstm_layer)
		intermediate.add(Dropout(0.2))

	if o.nLayers == "2":
		intermediate.add(lstm_layer_return_sequence)
		intermediate.add(Dropout(0.2))
		intermediate.add(lstm_layer)
		intermediate.add(Dropout(0.2))
	
	more = Sequential()
	more.add(Dense(32, input_dim =9))
	more.add(Activation('relu'))
	more.add(Dropout(0.2))

	model.add( Merge([intermediate, more], mode = 'concat'))

	model.add( Dense(64))
	model.add( Activation('relu'))
	model.add( Dropout(0.2))

	model.add( Dense(32))
	model.add( Activation('relu'))
	model.add( Dropout(0.2))

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
	history = model.fit( X_train_vec , y_train, batch_size=batch_size, nb_epoch=nb_epoch, validation_split=0.1, shuffle = True, sample_weight= sample_weight)
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
	sample_weight = dataset['weights_train']

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
	history = model.fit( X_train_vec , y_train, batch_size=batch_size, nb_epoch=nb_epoch, validation_split=0.2, shuffle = True, sample_weight= sample_weight)
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
	sample_weight = dataset['weights_train']

	# split by "continuous variable" and "categorization variable"
	X_train_vec_dR     = [X_train[:,:,0:4], X_train[:,:,-1]]
	X_concat = None
	pred = None
	llh = None

	if int(o.nLSTMClass) ==2:
		RNNmodel = model_from_json(open( 'V51_LSTM_dR_40epoch_5000kEvts_0nTrackCut_15nMaxTrack_2nLSTMClass_25nLSTMNodes_CMix_architecture.json').read())
		RNNmodel.load_weights( 'V51_LSTM_dR_40epoch_5000kEvts_0nTrackCut_15nMaxTrack_2nLSTMClass_25nLSTMNodes_CMix_model_weights.h5' )
		RNNmodel.compile(loss='binary_crossentropy', optimizer='adam', metrics=["accuracy"])

		pred = RNNmodel.predict( X_train_vec_dR, batch_size)
		X_concat = np.ndarray(shape=( pred[:,0].shape[0], 2))

	if int(o.nLSTMClass) ==4:
		RNNmodel = model_from_json(open( 'V51_LSTM_dR_40epoch_5000kEvts_0nTrackCut_15nMaxTrack_4nLSTMClass_25nLSTMNodes_CMix_architecture.json').read())
		RNNmodel.load_weights( 'V51_LSTM_dR_40epoch_5000kEvts_0nTrackCut_15nMaxTrack_4nLSTMClass_25nLSTMNodes_CMix_model_weights.h5' )
		RNNmodel.compile(loss='binary_crossentropy', optimizer='adam', metrics=["accuracy"])

		pred = RNNmodel.predict( X_train_vec_dR, batch_size)

		llh   = np.log(pred[:,0]/pred[:,2]) 
		#X_concat = np.ndarray(shape=( pred[:, 0].shape[0], 5))
		X_concat = np.ndarray(shape=( pred[:, 0].shape[0], 2))

	for i in range(X_concat.shape[0]):
		if o.Model == "RNNPlusMV2":
			X_concat[i][0] = labels_train[i, 10]
		if o.Model == "RNNPlusSV1":
			X_concat[i][0] = labels_train[i, 9]


		X_concat[i][1] = llh[i]
#		for j in range(pred.shape[1]):
#			#X_concat[i][j+1] = pred[i, j]
#			X_concat[i][j+1] = llh[j]


	model = Sequential()
	if (int(o.nLSTMClass)==2):
		model.add(Dense(10, input_dim=(2)) )
	if (int(o.nLSTMClass)==4):
		#model.add(Dense(10, input_dim=(5)) )
		model.add(Dense(10, input_dim=(2)) )

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
	history = model.fit( X_concat , y_train, batch_size=batch_size, nb_epoch=nb_epoch, validation_split=0.2, show_accuracy=True, shuffle = True, sample_weight=sample_weight)
	print "Finish Training"

	return (model, history)


def generateOutput():

	dataset_dR = makeData( Variables = "dR", padding = "pre" , nLSTMClass=2)

	labels_test = dataset_dR['labels_test']

	X_test_dR = dataset_dR['X_test']
	y_test_dR = dataset_dR['y_test']
	X_test_vec_dR     = [X_test_dR[:,:,0:4], X_test_dR[:,:,4]]

	RNNmodel_dR = model_from_json(open( 'V51_LSTM_dR_40epoch_5000kEvts_0nTrackCut_15nMaxTrack_2nLSTMClass_25nLSTMNodes_CMix_architecture.json').read())
	RNNmodel_dR.load_weights( 'V51_LSTM_dR_40epoch_5000kEvts_0nTrackCut_15nMaxTrack_2nLSTMClass_25nLSTMNodes_CMix_model_weights.h5' )
	RNNmodel_dR.compile(loss='binary_crossentropy', optimizer='adam', metrics=["accuracy"])
	RNNpred_dR = RNNmodel_dR.predict( X_test_vec_dR, batch_size)


	outfile = file("RNNScore.pkl", 'wb')
	cPickle.dump(RNNpred_dR, outfile, protocol=cPickle.HIGHEST_PROTOCOL)

	outfile = file("RNNInputVariables.pkl", 'wb')
	cPickle.dump(X_test_dR, outfile, protocol=cPickle.HIGHEST_PROTOCOL)

	outfile = file("RNNJetInfo.pkl", 'wb')
	cPickle.dump(labels_test, outfile, protocol=cPickle.HIGHEST_PROTOCOL)

	outfile = file("RNNFitTarget.pkl", 'wb')
	cPickle.dump(y_test_dR, outfile, protocol=cPickle.HIGHEST_PROTOCOL)


def saveModel(fileNameBase, model, history = None):

	json_string = model.to_json()
	print 'base ', fileNameBase
	open(fileNameBase+"_architecture.json", 'w').write(json_string)
	model.save_weights(fileNameBase + '_model_weights.h5', overwrite=True)

	history_out = file(fileNameBase+"_history.json", 'wb')
	cPickle.dump(history.history, history_out, protocol=cPickle.HIGHEST_PROTOCOL)


def evalModel(dataset, model, modelname):
	#################
	# Configuration #
	#################

	#################

	X_test = dataset['X_test']
	#sv1_test = dataset['sv1_test']
	y_test = dataset['y_test']
	labels_test = dataset['labels_test']
	#sv1_test = dataset['sv1_test']

	# split by "continuous variable" and "categorization variable"
	X_test_vec  = [X_test[:,:,0:-1],   X_test[:,:, -1]]
	if o.Variables == "Hits":
		X_test_vec  = X_test

	if o.Model == "DenseIP3D":
		X_test_vec  = [  X_test [:, 0:ntrk_cut, 0:2], X_test [:, 0:ntrk_cut,-1]]

	if o.Model == "RNNPlusMV2" or o.Model == "RNNPlusSV1":

		X_test_vec_dR = [X_test[:,:,0:4], X_test[:,:,-1]]
		X_test_vec = None
		pred = None
		llh = None

		if int(o.nLSTMClass) ==2:
			RNNmodel = model_from_json(open( 'V51_LSTM_dR_40epoch_5000kEvts_0nTrackCut_15nMaxTrack_2nLSTMClass_25nLSTMNodes_CMix_architecture.json').read())
			RNNmodel.load_weights( 'V51_LSTM_dR_40epoch_5000kEvts_0nTrackCut_15nMaxTrack_2nLSTMClass_25nLSTMNodes_CMix_model_weights.h5' )
			RNNmodel.compile(loss='binary_crossentropy', optimizer='adam', metrics=["accuracy"])

			pred = RNNmodel.predict( X_test_vec_dR, batch_size)
			X_test_vec = np.ndarray(shape=( pred[:,0].shape[0], 2))

		if int(o.nLSTMClass) ==4:
			RNNmodel = model_from_json(open( 'V51_LSTM_dR_40epoch_5000kEvts_0nTrackCut_15nMaxTrack_4nLSTMClass_25nLSTMNodes_CMix_architecture.json').read())
			RNNmodel.load_weights( 'V51_LSTM_dR_40epoch_5000kEvts_0nTrackCut_15nMaxTrack_4nLSTMClass_25nLSTMNodes_CMix_model_weights.h5' )
			RNNmodel.compile(loss='binary_crossentropy', optimizer='adam', metrics=["accuracy"])

			pred = RNNmodel.predict( X_test_vec_dR, batch_size)

			llh   = np.log(pred[:,0]/pred[:,2])

			#X_test_vec = np.ndarray(shape=( pred[:, 0].shape[0], 5))
			X_test_vec = np.ndarray(shape=( pred[:, 0].shape[0], 2))

			

		for i in range(X_test_vec.shape[0]):
			if o.Model == "RNNPlusMV2":
				X_test_vec[i][0] = labels_test[i, 10]
			if o.Model == "RNNPlusSV1":
				X_test_vec[i][0] = labels_test[i, 9]

			X_test_vec[i][1] = llh[i]
			#		for j in range(pred.shape[1]):
			#			#X_concat[i][j+1] = pred[i, j]
			#			X_concat[i][j+1] = llh[j]

#			for j in range(pred.shape[1]):
#				#X_test_vec[i][j+1] = pred[i, j]
#				X_test_vec[i][j+1] = llh[i, j]

	#if o.Model == "RNNSV1":
		#X_test_vec.append(sv1_test)

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
	#dataset_storage = dataset

	model = None
	history = None
	modelname = "" 
	print o.Model
	if "LSTM" in o.Model or "GRU" in o.Model:
		model, history = buildModel_1hidden(dataset, True)
	if o.Model == "RNNSV1":
		model, history = buildModel_RNNSV1(dataset, True)
	if o.Model == "DenseIP3D":
		model, history = buildModel_SimpleDense(dataset, False)
	print ' ------------------------------------------'
	print o.Model
	if o.Model == "RNNPlusMV2" or o.Model == "RNNPlusSV1":
		model, history = buildModel_RNNPlus(dataset, useAdam=True)


	modelname = o.Version +"_" + o.Model + "_"+ o.Variables + "_" + o.nEpoch + "epoch_" + str( n_events/1000) + 'kEvts_' + str( o.nTrackCut) + 'nTrackCut_' +  o.nMaxTrack + "nMaxTrack_" + o.nLSTMClass +"nLSTMClass_" + o.nLSTMNodes +"nLSTMNodes_"+ o.nLayers + "nLayers"

	model = evalModel(dataset, model, o.Model)
	
	if o.TrackOrder == 'pT':
		modelname += "_SortpT"
	if o.TrackOrder == 'Reverse':
		modelname += "_ReverseOrder"
	if o.TrackOrder == 'SL0':
		modelname += "_SL0"
	if o.doTrainC == 'y':
		modelname += "_CMix"
	if o.AddJetpT == 'y':
		modelname += '_AddJetpT'
	if int(o.EmbedSize) != 2:
		modelname += "_" + o.EmbedSize+"EmbedSize"

	if o.Mode == "R":
		modelname = o.filebase+"_Retrain_"+o.nEpoch
	if o.doLessC == "y":
		modelname += "_LessC"

	if o.doJetpTReweight == "y":
		modelname += "_JetpTReweight"

	#modelname = "test"
	saveModel(modelname, model, history)


def compareROC():

	dataset_dR = makeData( Variables = "dR", padding = "pre" , nLSTMClass=2)
	dataset_Hits = makeData( Variables = "Hits", padding = "pre" , nLSTMClass=2)

	dataset_dR_SL0 = makeData( Variables = "dR", padding = "pre" , nLSTMClass=2, TrackOrder = "SL0")
	dataset_dR_pT = makeData( Variables = "dR", padding = "pre" , nLSTMClass=2, TrackOrder = "pT")

	############################
	X_test_dR = dataset_dR['X_test']
	y_test_dR = dataset_dR['y_test']
	X_test_vec_dR     = [X_test_dR[:,:,0:4], X_test_dR[:,:,4]]
	
	labels_test_dR = dataset_dR['labels_test']
	ntrk_test = labels_test_dR[:,7]
	pt_test = labels_test_dR[:,1]

	X_test_dR_SL0 = dataset_dR_SL0['X_test']
	y_test_dR_SL0 = dataset_dR_SL0['y_test']
	X_test_vec_dR_SL0     = [X_test_dR_SL0[:,:,0:4], X_test_dR_SL0[:,:,4]]
	labels_test_dR_SL0 = dataset_dR_SL0['labels_test']

	X_test_dR_pT = dataset_dR_pT['X_test']
	y_test_dR_pT = dataset_dR_pT['y_test']
	X_test_vec_dR_pT     = [X_test_dR_pT[:,:,0:4], X_test_dR_pT[:,:,4]]
	labels_test_dR_pT = dataset_dR_pT['labels_test']


	X_test_vec_Hits = dataset_Hits['X_test']
	y_test_Hits = dataset_Hits['y_test']
	labels_test_Hits = dataset_Hits['labels_test']
	pt_test_Hits = labels_test_Hits[:,1]

        ip3d_test = labels_test_dR[:,3]
	SV1_test = labels_test_dR[:,9]
	MV2_test = labels_test_dR[:,10]

	#LoadModel("V51_LSTM_Hits_60epoch_5000kEvts_0nTrackCut_15nMaxTrack_4nLSTMClass_50nLSTMNodes_1nLayers_CMix", X_test_vec_Hits, "LSTM 60E 4Class Hit Variables")
	#LoadModel("V51_LSTM_Hits_60epoch_5000kEvts_0nTrackCut_15nMaxTrack_4nLSTMClass_50nLSTMNodes_1nLayers_CMix_JetpTReweight", X_test_vec_Hits, "LSTM 60E 4Class Hit Variables pTReweight")
	
	#LoadModel("V51_LSTM_dR_60epoch_5000kEvts_0nTrackCut_15nMaxTrack_4nLSTMClass_50nLSTMNodes_1nLayers_CMix", X_test_vec_dR, "Rel21 ttbar LSTM 60E 4Class")
	#LoadModel("V51_LSTM_dR_60epoch_5000kEvts_0nTrackCut_15nMaxTrack_4nLSTMClass_50nLSTMNodes_1nLayers_CMix_JetpTReweight", X_test_vec_dR, "Rel21 ttbar LSTM 60E 4Class pTReweight")

	LoadModel("V56_LSTM_dR_60epoch_5000kEvts_0nTrackCut_15nMaxTrack_4nLSTMClass_50nLSTMNodes_1nLayers_CMix", X_test_vec_dR, "Rel21 Z' LSTM 60E 4Class Sd0_{sig} order")
	LoadModel("V56_LSTM_dR_60epoch_5000kEvts_0nTrackCut_15nMaxTrack_4nLSTMClass_50nLSTMNodes_1nLayers_SL0_CMix", X_test_vec_dR_SL0, "Rel21 Z' LSTM 60E 4Class SL0_{sig} order")
	LoadModel("V56_LSTM_dR_60epoch_5000kEvts_0nTrackCut_15nMaxTrack_4nLSTMClass_50nLSTMNodes_1nLayers_SortpT_CMix", X_test_vec_dR_pT, "Rel21 Z' LSTM 60E 4Class p_{T} order")

	LoadModel("V56_LSTM_dR_60epoch_5000kEvts_0nTrackCut_15nMaxTrack_4nLSTMClass_50nLSTMNodes_1nLayers_CMix_JetpTReweight", X_test_vec_dR, "Rel21 Z' LSTM 60E 4Class Sd0_{sig} order pTReweight")
	LoadModel("V56_LSTM_dR_60epoch_5000kEvts_0nTrackCut_15nMaxTrack_4nLSTMClass_50nLSTMNodes_1nLayers_SL0_CMix_JetpTReweight", X_test_vec_dR_SL0, "Rel21 Z' LSTM 60E 4Class SL0_{sig} order pTReweight")
	LoadModel("V56_LSTM_dR_60epoch_5000kEvts_0nTrackCut_15nMaxTrack_4nLSTMClass_50nLSTMNodes_1nLayers_SortpT_CMix_JetpTReweight", X_test_vec_dR_pT, "Rel21 Z' LSTM 60E 4Class pT order pTReweight")


	def DrawROC(models, outputName, bkg="l"):
		labels = ["IP3D", "SV1", "MV2C10"]

		bscores = [ip3d_test[labels_test_dR[:,0]==5], SV1_test[labels_test_dR[:,0]==5], MV2_test[labels_test_dR[:,0]==5]]
		lscores = [ip3d_test[labels_test_dR[:,0]==0], SV1_test[labels_test_dR[:,0]==0], MV2_test[labels_test_dR[:,0]==0]]
		cscores = [ip3d_test[labels_test_dR[:,0]==4], SV1_test[labels_test_dR[:,0]==4], MV2_test[labels_test_dR[:,0]==4]]
		
		for m in models:
			labels.append( m.label)
			if "Hits" not in m.filebase:
				try :
					bscores.append( m.pred[labels_test_dR[:,0]==5, 0] )
					cscores.append( m.pred[labels_test_dR[:,0]==4, 0] )
					lscores.append( m.pred[labels_test_dR[:,0]==0, 0] )
				except IndexError:
					bscores.append( m.pred[labels_test_dR[:,0]==5] )
					cscores.append( m.pred[labels_test_dR[:,0]==4] )
					lscores.append( m.pred[labels_test_dR[:,0]==0] )

			else:
				try :
					bscores.append( m.pred[labels_test_Hits[:,0]==5, 0] )
					cscores.append( m.pred[labels_test_Hits[:,0]==4, 0] )
					lscores.append( m.pred[labels_test_Hits[:,0]==0, 0] )
				except IndexError:
					bscores.append( m.pred[labels_test_Hits[:,0]==5] )
					cscores.append( m.pred[labels_test_Hits[:,0]==4] )
					lscores.append( m.pred[labels_test_Hits[:,0]==0] )

		if bkg =="l":
			plottingUtils.getROC( bscores, lscores, labels, outputName=outputName, Rejection=bkg)
		if bkg =="c":
			plottingUtils.getROC( bscores, cscores, labels, outputName=outputName, Rejection=bkg)


	def DrawLoss(models, outputName):
		train_loss = []
		val_loss = []
		labels_train = []
		labels_val = []

		for m in models:
			labels_train.append(m.label + " train loss")
			labels_val.append(m.label + " val loss")
			train_loss.append(m.loss)
			val_loss.append(m.val_loss)

		plottingUtils.getTrainingCurve( train_loss + val_loss,  labels_train + labels_val, outputName=outputName)


        def getScoreCutList(scoreList):
		bins = 	[100, 300, 500, 900, 1100, 1500, 2000, 3000]
                return plottingUtils.getFixEffCurve(scoreList = scoreList,  varList = pt_test[labels_test_dR[:,0]==5]/1000.0,
						    label = "IdontCare",
						    bins = bins,
						    fix_eff_target = 0.7,
						    onlyReturnCutList = True
                                                    )


        def DrawFlatEfficiencyCurves(models, outputName):
		labels = ["IP3D", "SV1", "MV2C10"]
		bins = 	[100, 300, 500, 900, 1100, 1500, 2000, 3000]
		varList = pt_test[labels_test_dR[:,0]==0]
		varList = varList/1000.
		
		bscores = [ip3d_test[labels_test_dR[:,0]==5], SV1_test[labels_test_dR[:,0]==5], MV2_test[labels_test_dR[:,0]==5]]
		lscores = [ip3d_test[labels_test_dR[:,0]==0], SV1_test[labels_test_dR[:,0]==0], MV2_test[labels_test_dR[:,0]==0]]

		for m in models:
			labels.append( m.label)
			if "Hits" in  m.filebase:

				try :
					bscores.append( m.pred[labels_test_Hits[:,0]==5, 0] )
					lscores.append( m.pred[labels_test_Hits[:,0]==0, 0] )
				except IndexError:
					bscores.append( m.pred[labels_test_Hits[:,0]==5] )
					lscores.append( m.pred[labels_test_Hits[:,0]==0] )
				continue

				
			try :
				bscores.append( m.pred[labels_test_dR[:,0]==5, 0] )
				lscores.append( m.pred[labels_test_dR[:,0]==0, 0] )
			except IndexError:
				bscores.append( m.pred[labels_test_dR[:,0]==5] )
				lscores.append( m.pred[labels_test_dR[:,0]==0] )
				

		approachList = []
		for i in range(len(labels)):
			approachList.append( (lscores[i], varList, ("EffCurvePt"+labels[i], labels[i]), getScoreCutList(bscores[i])) )

                plottingUtils.MultipleFlatEffCurve( outputName,  approachList = approachList, bins = bins )

	def DrawLEff_pT(  models, labels, pt):
		approachList = [
			(ip3d_test[labels_test_dR[:,0]==5], pt_test[labels_test_dR[:,0]==5]/1000., 
			 ip3d_test[labels_test_dR[:,0]==0], pt_test[labels_test_dR[:,0]==0]/1000., ("EffCurve_IP3D", "IP3D Signal Efficiency")),
			(SV1_test[labels_test_dR[:,0]==5], pt_test[labels_test_dR[:,0]==5]/1000.,
			 SV1_test[labels_test_dR[:,0]==0], pt_test[labels_test_dR[:,0]==0]/1000., ("EffCurve_SV1", "SV1 Signal Efficiency")),                                         
			(MV2_test[labels_test_dR[:,0]==5], pt_test[labels_test_dR[:,0]==5]/1000.,
			 MV2_test[labels_test_dR[:,0]==0], pt_test[labels_test_dR[:,0]==0]/1000., ("EffCurve_MV2", "MV2 Signal Efficiency"))]

		for m in models:
			approachList.append( (m.pred[labels[:,0]==5], pt[ labels[:,0]==5]/1000.,
					      m.pred[labels[:,0]==0], pt[ labels[:,0]==0]/1000., ("EffCurve_"+m.label, m.label+" Efficiency")) )
			
		plottingUtils.MultipleRejCurve(
			outputName = "LEffCurveCompare_pT.root", 
			approachList = approachList,
			bins = 	[100, 300, 500, 900, 1100, 1500, 2000, 3000],
			eff_target = 0.7,)



	def DrawBEff_pT( models, labels, pt):
		approachList = [  
			(ip3d_test[labels_test_dR[:,0]==5], pt_test[labels_test_dR[:,0]==5]/1000., ("EffCurve_IP3D", "IP3D Signal Efficiency")),
			(SV1_test[labels_test_dR[:,0]==5], pt_test[labels_test_dR[:,0]==5]/1000., ("EffCurve_SV1", "SV1 Signal Efficiency")),                                         
			(MV2_test[labels_test_dR[:,0]==5], pt_test[labels_test_dR[:,0]==5]/1000., ("EffCurve_MV2", "MV2 Signal Efficiency"))]
		for m in models:
			approachList.append( (m.pred[labels[:,0]==5], pt[ labels[:,0]==5]/1000.,("EffCurve_"+m.label, m.label+" Efficiency")))

		plottingUtils.MultipleEffCurve(                                 
			outputName = "BEffCurveCompare_pT.root",   
			approachList = approachList,
			bins = 	[100, 300, 500, 900, 1100, 1500, 2000, 3000],
			eff_target = 0.7,)

#	Comp_Quick_4Class             = [SavedModels["V51_LSTM_Hits_60epoch_5000kEvts_0nTrackCut_15nMaxTrack_4nLSTMClass_50nLSTMNodes_1nLayers_CMix"],
#					 SavedModels["V51_LSTM_Hits_60epoch_5000kEvts_0nTrackCut_15nMaxTrack_4nLSTMClass_50nLSTMNodes_1nLayers_CMix_JetpTReweight"]]

	Comp_Quick_4Class = [SavedModels["V56_LSTM_dR_60epoch_5000kEvts_0nTrackCut_15nMaxTrack_4nLSTMClass_50nLSTMNodes_1nLayers_CMix"],
			     SavedModels["V56_LSTM_dR_60epoch_5000kEvts_0nTrackCut_15nMaxTrack_4nLSTMClass_50nLSTMNodes_1nLayers_CMix_JetpTReweight"],
			     SavedModels["V56_LSTM_dR_60epoch_5000kEvts_0nTrackCut_15nMaxTrack_4nLSTMClass_50nLSTMNodes_1nLayers_SL0_CMix"],
			     SavedModels["V56_LSTM_dR_60epoch_5000kEvts_0nTrackCut_15nMaxTrack_4nLSTMClass_50nLSTMNodes_1nLayers_SL0_CMix_JetpTReweight"],
			     SavedModels["V56_LSTM_dR_60epoch_5000kEvts_0nTrackCut_15nMaxTrack_4nLSTMClass_50nLSTMNodes_1nLayers_SortpT_CMix"],
			     SavedModels["V56_LSTM_dR_60epoch_5000kEvts_0nTrackCut_15nMaxTrack_4nLSTMClass_50nLSTMNodes_1nLayers_SortpT_CMix_JetpTReweight"]]

	
	DrawROC(Comp_Quick_4Class,         "BL_4Class_Layer.root", bkg="l")
	DrawROC(Comp_Quick_4Class,         "BC_4Class_Layer.root", bkg="c")
	DrawFlatEfficiencyCurves(Comp_Quick_4Class,   "RejAtFlatEff_4Class.root")
	DrawLEff_pT(  Comp_Quick_4Class, labels_test_dR, pt_test)
	DrawBEff_pT(  Comp_Quick_4Class, labels_test_dR, pt_test)

	#DrawLEff_pT(  Comp_Quick_4Class, labels_test_dR, pt_test_Hits)
	#DrawBEff_pT(  Comp_Quick_4Class, labels_test_dR, pt_test_Hits)

        print "Making b jet efficiency v.s. pT curve ... "


########################################

if __name__ == "__main__":
	if o.doBatch == "y":
		currentPWD = os.getcwd()

		modelname = o.Version +"_" + o.Model + "_"+ o.Variables + "_" + o.nEpoch + "epoch_" + str( n_events/1000) + 'kEvts_' + str( o.nTrackCut) + 'nTrackCut_' +  o.nMaxTrack + "nMaxTrack_" + o.nLSTMClass +"nLSTMClass_" + o.nLSTMNodes +"nLSTMNodes_"+o.nLayers + "nLayers"
		if o.TrackOrder == 'pT':
			modelname += "_SortpT"
		if o.TrackOrder == 'Reverse':
			modelname += "_ReverseOrder"
		if o.TrackOrder == 'SL0':
			modelname += "_SL0"
		if o.doTrainC == 'y':
			modelname += '_CMix'
		if o.AddJetpT == 'y':
			modelname += '_AddJetpT'
		if int(o.EmbedSize) != 2:
			modelname += "_" + o.EmbedSize+"EmbedSize"


		if o.Mode == "R":
			modelname = o.filebase+"_Retrain_"+o.nEpoch

		if o.doLessC == "y":
			modelname += "_LessC"

		if o.doJetpTReweight == "y":
			modelname += "_JetpTReweight"
		
		cmd = "bsub -q atlas-t3 -W 80:00 -o 'output/" + modelname + "' THEANO_FLAGS='base_compiledir=" + currentPWD + "/BatchCompileDir/0/' python lstmUtils.py --nEpoch " + o.nEpoch + " --Mode " + o.Mode + " --Var " + o.Variables + " --nEvents " + o.nEvents + " --doTrainC " + o.doTrainC + " --nMaxTrack " + o.nMaxTrack + " --TrackOrder " + o.TrackOrder + " --padding " + o.padding + " --Model " + o.Model + " --nTrackCut " + o.nTrackCut + " --AddJetpT " + o.AddJetpT + " --nLSTMClass " + o.nLSTMClass + " --nLSTMNodes " + o.nLSTMNodes + " --nLayers "+o.nLayers + " --EmbedSize " + o.EmbedSize + " --Filebase " + o.filebase + " --doLessC "+o.doLessC + " --doJetpTReweight " + o.doJetpTReweight + " --Version " + o.Version

		print cmd
		os.system(cmd)

	else:
		if o.Mode == "M" or o.Mode == "R":
			BuildModel()
		if o.Mode == "C":
			compareROC()
		if o.Mode == "P":
			generateOutput()

