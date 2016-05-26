import keras.backend as K


def MaskingHack(x):
    #mask = K.repeat_elements( K.any(x[:,:,0:-2], axis=-1), rep=x.shape[-1], axis=-1 )
    mask = K.any(x[:,:,0:-2], axis=-1, keepdims=True)
    return x*mask

def MaskingHack_output_shape(input_shape):
    return input_shape
 
