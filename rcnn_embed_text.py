import TextToArray
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
from keras.layers.convolutional import Convolution1D, MaxPooling1D
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM, GRU
from keras.layers.normalization import BatchNormalization

from keras.layers.extras import TimeDistributedPassThrough
from keras.layers.birnn import BiDirectionLSTM


sys.setrecursionlimit(40000)

openSaved = False
outfile = 'rcnn_lstm_adam_embed_max15_test.save'



# set parameters:
batch_size = 32
max_len = 15

max_embed_features=16
embed_size=2

nb_filters = 50
filter_length = 3

hidden_dims = 50
nb_epoch = 3

stride=1

addPadForCNN = True


use100kSample=False
n_events = 150000


#make data#######################
print "Getting Data"

if use100kSample:
    X = TextToArray.MakePaddedSequenceTensor([ 'MakeData/sequences_sd0_100k.txt', 'MakeData/sequences_sz0_100k.txt'], doWhitening=False, maxlen=max_len)[:n_events]
    trk_grd = TextToArray.convertSequences( 'MakeData/sequences_grd_100k.txt', dopad=True, maxlen=max_len )[:n_events]
else:
    X = TextToArray.MakePaddedSequenceTensor([ 'MakeData/sequences_sd0.txt', 'MakeData/sequences_sz0.txt'], doWhitening=False, maxlen=max_len)
    trk_grd = TextToArray.convertSequences( 'MakeData/sequences_grd.txt', dopad=True, maxlen=max_len )

X = np.dstack((X, trk_grd+1))

if addPadForCNN:
    zeros_slice = np.zeros((X.shape[0],1,X.shape[2]))
    print  zeros_slice.shape,  X.shape
    X = np.concatenate( (zeros_slice  , X), axis=1  ) # pre-zero
    X = np.concatenate( (X,  zeros_slice ), axis=1  ) # post-zero


X_train, X_test = np.split( X, [ int(0.8*X.shape[0]) ] )

X_train_vec = [X_train[:,:,0:-1], X_train[:,:,-1] ]
X_test_vec = [X_test[:,:,0:-1], X_test[:,:,-1] ]

n_cont_vars = X_train_vec[0].shape[2]

print("data shape",X.shape)




print "Getting Labels"
if use100kSample:
    labels = TextToArray.convertSequences('MakeData/labels_100k.txt')[:n_events]
else:
    labels = TextToArray.convertSequences('MakeData/labels.txt')
    
labels_train, labels_test = np.split( labels, [ int(0.8*labels.shape[0]) ] )

ip3d_test = labels_test[:,3]

y = (labels[:,0] ==5)

#y = np.repeat( np.array([ [[l]] for l in labels[:,0] ==5]), X.shape[1], axis=1 )


y_train, y_test = np.split( y, [ int(0.8*y.shape[0]) ] )

print y_train.shape, y_test.shape





    
# build the model: 2 stacked LSTM
print('Build model...')

left = Sequential()
left.add(TimeDistributedPassThrough(n_cont_vars))

right = Sequential()
right.add(Embedding(max_embed_features, embed_size))


model = Sequential()
model.add( Merge([left, right],mode='concat') )
model.add(Dropout(0.25))


model.add(Convolution1D(input_dim= (n_cont_vars+embed_size),
                        nb_filter=nb_filters,
                        filter_length=filter_length,
                        border_mode="valid",
                        activation="relu",
                        subsample_length=1))

model.add(BatchNormalization( (((max_len-filter_length +2*(addPadForCNN))/stride + 1 ) ,nb_filters) ) )



#model.add(BiDirectionLSTM( n_cont_vars+embed_size, 20, return_sequences=True, output_mode='sum')) # try using a GRU instead, for fun
model.add(LSTM( nb_filters, 50, return_sequences=False)) # try using a GRU instead, for fun
model.add(Dropout(0.25))

#model.add(TimeDistributedDense(20,1))
#model.add(Activation('sigmoid'))

#model.add(LSTM( 50, 50)) # try using a GRU instead, for fun
#model.add(Dropout(0.25))

model.add(Dense(50, 1))
model.add(Activation('sigmoid'))




# try using different optimizers and different optimizer configs
model.compile(loss='binary_crossentropy', optimizer='adam', class_mode="binary")



print("Train...")
model.fit( X_train_vec , y_train, batch_size=batch_size, nb_epoch=nb_epoch, validation_split=0.1, show_accuracy=True)
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

for t in range(len(pred)):
    print y_test[t], pred[t]


if True:
    outfile = file(outfile, 'wb')
    cPickle.dump(model, outfile, protocol=cPickle.HIGHEST_PROTOCOL)
    outfile.close()

plotting.ROC( [ pred[labels_test[:,0]==5,0], ip3d_test[labels_test[:,0]==5] ],
              [ pred[labels_test[:,0]==0,0], ip3d_test[labels_test[:,0]==0] ],
              label=["DNN","IP3D"] )

