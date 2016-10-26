from keras import backend as K
from keras.layers.core import Lambda, MaskedLayer


#class PassThrough(Lambda):
#   def __init__(self, input_shape):
#       super(PassThrough, self).__init__(function=lambda x: x, input_shape=input_shape, output_shape=input_shape)


## class Pass(MaskedLayer):
##     ''' Do literally nothing
##         It can the first layer
##     '''
##     def __init__(self, ndim=2, **kwargs):
##         super(Pass, self).__init__(**kwargs)
##         self.input = K.placeholder(ndim=ndim)

##     def get_output(self, train=False):
##         X = self.get_input(train)
##         return X

##     @property
##     def output_shape(self):
##         print "output_shape=", self.input_shape
##         return self.input_shape


class TimeDistributedPassThrough(MaskedLayer):
    '''
       passthrough
       Tensor input dimensions:   (nb_sample, shared_dimension, input_dim)
       Tensor output dimensions:  (nb_sample, shared_dimension, input_dim)

    '''
    def __init__(self, input_shape, **kwargs):
        #super(TimeDistributedPassThrough, self).__init__()
        #self.input_shape = input_shape
        #self.output_shape = input_shape
        self.input = K.placeholder(ndim=3)
        kwargs['input_shape'] = input_shape
        super(TimeDistributedPassThrough, self).__init__(**kwargs)
        
    def get_output(self, train=False):
        X = self.get_input(train)
        return X

    @property
    def output_shape(self):
        #print "output_shape=", self.input_shape
        return self.input_shape
