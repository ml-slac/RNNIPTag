import TextToArray as TTA
import plotting
import matplotlib.pyplot as plt
import numpy as np
import cPickle
import sys

from keras.preprocessing import sequence
from keras.optimizers import SGD, RMSprop, Adagrad
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Merge, Flatten, TimeDistributedDense, Masking
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM, GRU
from keras.models import model_from_json

sys.setrecursionlimit(40000)

openSaved = False
outfile = 'lstm_adam_noWhitening_max10_test.save'

batch_size = 32
max_len = 15

################################################################################################
#make data
################################################################################################
print "Getting Data"
f = file('MakeData/test.pkl','r')
trk_arr = cPickle.load(f)
labels = cPickle.load(f)
f.close()

#####################################
#inputs
#####################################
X = TTA.MakePaddedSequenceTensorFromListArray( trk_arr[:,:,0:2], doWhitening=False, maxlen=max_len)
trk_grd = TTA.convertToBinaryMap( TTA.convertSequencesFromListArray( trk_arr[:,:,2], dopad=True, maxlen=max_len ), range(0,15) )

X = np.dstack( (X, trk_grd) )

X_train, X_test = np.split( X, [ int(0.8*X.shape[0]) ] )

print("data shape",X.shape)


#####################################
#labels
#####################################
print "Getting Labels"
labels_train, labels_test = np.split( labels, [ int(0.8*labels.shape[0]) ] )

ip3d_test = labels_test[:,3]

y = (labels[:,0] ==5)

#y = np.repeat( np.array([ [[l]] for l in labels[:,0] ==5]), X.shape[1], axis=1 )


y_train, y_test = np.split( y, [ int(0.8*y.shape[0]) ] )

print y_train.shape, y_test.shape


################################################################################################
#make model
################################################################################################


if openSaved:
    f = open(outfile, 'r')
    model_labels, model_pred = cPickle.load(f)
    model = cPickle.load(f)
    f.close()

else:
    
    # build the model: 2 stacked LSTM
    print('Build model...')
    model = Sequential()
    #model.add(Masking())
    model.add(LSTM( X.shape[2], 20, return_sequences=True)) # try using a GRU instead, for fun
    model.add(Dropout(0.2))
    
    #model.add(TimeDistributedDense(20,1))
    #model.add(Activation('sigmoid'))
    
    #model.add(Flatten())
    #model.add(Dense(X.shape[1], 1))
    
    model.add(LSTM( 20, 20)) # try using a GRU instead, for fun
    model.add(Dropout(0.2))
    model.add(Dense(20, 1))
    model.add(Activation('sigmoid'))


    # try using different optimizers and different optimizer configs
    model.compile(loss='binary_crossentropy', optimizer='adam', class_mode="binary")


    print("Train...")
    model.fit(X_train, y_train, batch_size=batch_size, nb_epoch=10, validation_split=0.1, show_accuracy=True)
    score = model.evaluate(X_test, y_test, batch_size=batch_size)
    print('Test score:', score)







classes = model.predict_classes(X_test, batch_size=batch_size)
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

pred = model.predict(X_test, batch_size=batch_size)

#for t in range(len(pred)):
#    print y_test[t], pred[t]


if not openSaved:
    outfile = file(outfile, 'wb')
    cPickle.dump( (labels_test, pred),  outfile, protocol=cPickle.HIGHEST_PROTOCOL)
    cPickle.dump(model, outfile, protocol=cPickle.HIGHEST_PROTOCOL)
    outfile.close()

plotting.ROC( [ pred[labels_test[:,0]==5,0], ip3d_test[labels_test[:,0]==5] ],
              [ pred[labels_test[:,0]==0,0], ip3d_test[labels_test[:,0]==0] ],
              label=["LSTM","IP3D"] )

