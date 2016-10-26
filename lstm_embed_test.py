# THEANO_FLAGS='floatX=float32,device=gpu0,lib.cnmem=1'  python <myscript>.py

import TextToArray as TTA
#import plotting
#import matplotlib.pyplot as plt
import numpy as np
import cPickle
import sys


from keras.preprocessing import sequence
from keras.optimizers import SGD, RMSprop, Adagrad
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Merge, Flatten, TimeDistributedDense, Masking, Lambda
from keras.layers import TimeDistributed
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM, GRU
from keras.models import model_from_yaml, model_from_json

from CustomFunctions import MaskingHack, MaskingHack_output_shape



sys.setrecursionlimit(40000)





# set parameters:
batch_size = 32
max_len = 15

max_embed_features=16
embed_size=2

doWhitening=False

use100kSample=True
useSort_absd0=True
n_events = 500

nb_epoch = 10

postfix =  ('_max'+str(max_len) if max_len != None else '') + ('_whitened' if doWhitening else '') + ('_SORTabsd0' if useSort_absd0 else '') + ('_Nevt'+str(n_events/1000)+'k' if use100kSample else '')  + '_Nep'+str(nb_epoch)


openSaved = True

#outfileName = 'lstm_adam_embed'+ postfix+'.save'
outfileName = 'test_pred.save'


GetMore=False

print "Outfile", outfileName




## if True:
##     print ("Opening model file...")
##     #model = model_from_yaml(open('my_model_architecture.yaml').read())
##     model = model_from_json(open('my_model_architecture.json').read())
##     model.load_weights('my_model_weights.h5')
##     sys.exit(0)





###############################################################################################
#make data
################################################################################################
in_post = ('_100k' if use100kSample else '') + ('_sort_absd0' if useSort_absd0 else '')

print "Getting Data"
f = file('MakeData/test.pkl','r')
trk_arr_all = cPickle.load(f)
labels_all = cPickle.load(f)
f.close()



#####################################
#inputs
#####################################
X = TTA.MakePaddedSequenceTensorFromListArray( trk_arr_all[:,0:2], doWhitening=False, maxlen=max_len)
#trk_grd = TTA.convertToBinaryMap( TTA.convertSequencesFromListArray( trk_arr_all[:,2], dopad=True, maxlen=max_len ), range(0,15) )
trk_grd = TTA.convertSequencesFromListArray( trk_arr_all[:,2], dopad=True, pad_value=-1, maxlen=max_len )


X_all = np.dstack( (X, trk_grd+1) )

X = X_all[:n_events]

X_train, X_test = np.split( X, [ int(0.8*X.shape[0]) ] )

X_train_vec = [X_train[:,:,0:-1], X_train[:,:,-1] ]
X_test_vec = [X_test[:,:,0:-1], X_test[:,:,-1] ]

if GetMore and use100kSample:
    X_more_vec = [X_all[n_events:2*n_events,:,0:-1], X_all[n_events:2*n_events,:,-1] ]

n_cont_vars = X_train_vec[0].shape[2]

print("data shape",X.shape)



#####################################
#labels
#####################################
print "Getting Labels"
labels = labels_all[:n_events]
labels_train, labels_test = np.split( labels, [ int(0.8*labels.shape[0]) ] )

if GetMore and use100kSample:
    labels_more = labels_all[n_events:2*n_events]

ip3d_test = labels_test[:,3]

y = (labels[:,0] ==5)

#y = np.repeat( np.array([ [[l]] for l in labels[:,0] ==5]), X.shape[1], axis=1 )


y_train, y_test = np.split( y, [ int(0.8*y.shape[0]) ] )

print y_train.shape, y_test.shape





################################################################################################
#make model
################################################################################################


# build the model: 2 stacked LSTM
if not openSaved:
    
    print('Build model...')
    
    left = Sequential()
    #left.add( Masking( mask_value=0, input_shape = (max_len, n_cont_vars) ) )
    left.add( Activation('linear', input_shape=(max_len, n_cont_vars)) )

    right = Sequential()
    right.add(Embedding(max_embed_features, embed_size, mask_zero=False, input_length=max_len))

    merged = Merge([left, right],mode='concat')
    #merged = TimeDistributed(Merge([left, right],mode='concat'), input_shape=(max_len, n_cont_vars+embed_size))

    model = Sequential()
    model.add( merged )

    model.add(Lambda(MaskingHack, output_shape = MaskingHack_output_shape))

    model.add( Masking( mask_value=0.) )
    
    model.add(LSTM(50, return_sequences=True)) # try using a GRU instead, for fun
    model.add(Dropout(0.3))

    #model.add(TimeDistributedDense(20,1))
    #model.add(Activation('sigmoid'))

    model.add(LSTM(50)) # try using a GRU instead, for fun
    model.add(Dropout(0.3))
    model.add(Dense(1))
    model.add(Activation('sigmoid'))


    # try using different optimizers and different optimizer configs
    model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=["accuracy"])



    print("Train...")
    model.fit( X_train_vec , y_train, batch_size=batch_size, nb_epoch=nb_epoch, validation_split=0.1)


else:
    print ("Opening model file...")
    model = model_from_json(open('my_model_architecture.json').read())
    model.load_weights('my_model_weights.h5')
    model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=["accuracy"])



score = model.evaluate(X_test_vec, y_test, batch_size=batch_size)
print('Test score:', score)




classes = model.predict_classes(X_test_vec, batch_size=batch_size)
acc = np_utils.accuracy(classes, y_test)
print('Test accuracy:', acc)

acc = np_utils.accuracy(classes[y_test==True], y_test[y_test==True])
print('Test b accuracy:', acc)

acc = np_utils.accuracy(classes[y_test==False], y_test[y_test==False])
print('Test non-b accuracy:', acc)

acc = np_utils.accuracy(classes[labels_test[:,0]==4], y_test[labels_test[:,0]==4])
print('Test c accuracy:', acc)

acc = np_utils.accuracy(classes[labels_test[:,0]==0], y_test[labels_test[:,0]==0])
print('Test l accuracy:', acc)



pred = model.predict(X_test_vec, batch_size=batch_size)

if GetMore and use100kSample:
    pred_more = model.predict(X_more_vec, batch_size=batch_size)


#for t in range(len(pred)):
#    print y_test[t], pred[t]


if True: #not openSaved:
    json_string = model.to_json()
    open('my_model_architecture.json', 'w').write(json_string)
    model.save_weights('my_model_weights.h5', overwrite=True)

    outfile = file(outfileName, 'wb')
    if GetMore and use100kSample:
        cPickle.dump( (labels_more, pred_more),  outfile, protocol=cPickle.HIGHEST_PROTOCOL)
    else:
        cPickle.dump( (labels_test, pred),  outfile, protocol=cPickle.HIGHEST_PROTOCOL)
    outfile.close()

#plotting.ROC( [ pred[labels_test[:,0]==5,0], ip3d_test[labels_test[:,0]==5] ],
#              [ pred[labels_test[:,0]==0,0], ip3d_test[labels_test[:,0]==0] ],
#              label=["DNN","IP3D"] )

