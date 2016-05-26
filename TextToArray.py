import StringIO
import cPickle
import sys
import numpy
import scipy
import theano
from keras.preprocessing.sequence import pad_sequences

from sequence import *


#################################################################################
# list of arrays as input
#################################################################################
def MakePaddedSequenceTensorFromListArray( in_arr, pad_value=0., doWhitening=False, maxlen=None, padding = 'pre'):
    seq_list = numpy.array([])

    arr = in_arr
    if len(in_arr.shape)==1:
        arr = numpy.array([ in_arr ])

    for i in range( arr.shape[1] ):
        current = convertSequencesFromListArray( arr[:,i], dopad=True, doWhitening=doWhitening, maxlen=maxlen, padding = padding )
        
        if len(seq_list)==0:
            seq_list = current 
        else:
            seq_list = numpy.dstack((seq_list, current) )

    return seq_list


def convertSequencesFromListArray( arr, dopad=False, pad_value = 0., doWhitening=False, maxlen=None, padding = 'pre' ):

    seq_list = []

    if doWhitening:
        len_elements = 0.0
        sum_elements = 0.0
        sum_sqr_elements = 0.0
        
        for i in range(arr.shape[0]):
            seq_list.append(arr[i])
            
            len_elements = len_elements + len(arr[i])
            sum_elements = sum_elements + sum(arr[i])
            sum_sqr_elements = sum_sqr_elements + sum( [x**2 for x in arr[i]] )

        if len_elements > 0:
            mean_elements = 1.0 * sum_elements / (1.0*len_elements)
            sum_sqr_elements = numpy.sqrt( (1.0*sum_sqr_elements / (1.0*len_elements) )**2 - mean_elements**2  )

            if sum_sqr_elements > 0:
                seq_list = [ (( s - mean_elements) / sum_sqr_elements) for s in seq_list ]

    if dopad:
        seq_list = pad_sequences( (seq_list if doWhitening else arr),
                                  maxlen=maxlen, dtype=float, padding=padding, truncating='post' )

    return numpy.array( seq_list )
    

#################################################################################
# text file input
#################################################################################
def convertSequences( filename, dopad=False, pad_value = 0., doWhitening=False, maxlen=None ):

    f=open(filename,'r')

    seq_list = []

    len_elements = 0.0
    sum_elements = 0.0
    sum_sqr_elements = 0.0

    for line in f:
        a = numpy.loadtxt(StringIO.StringIO(line), dtype=float, delimiter=',', ndmin=1)
        seq_list.append(a)

        if doWhitening:
            len_elements = len_elements + len(a)
            sum_elements = sum_elements + sum(a)
            sum_sqr_elements = sum_sqr_elements + sum( [i**2 for i in a] )

    if doWhitening and len_elements > 0:
        mean_elements = 1.0 * sum_elements / (1.0*len_elements)
        sum_sqr_elements = numpy.sqrt( (1.0*sum_sqr_elements / (1.0*len_elements) )**2 - mean_elements**2  )

        if sum_sqr_elements > 0:
            seq_list = [ (( s - mean_elements) / sum_sqr_elements) for s in seq_list ]
    
    #print seq_list

    #splitBySequenceLength( sorted(seq_list, key = lambda x: len(x)) )

    #print sorted(seq_list, key = lambda x: len(x))

    if dopad:
        seq_list = pad_sequences( seq_list, maxlen=maxlen, dtype=float, padding='pre', truncating='post' )    

    f.close()
        
    return numpy.array( seq_list )


def MakePaddedSequenceTensor( filename_list, doWhitening=False, maxlen=None):
    seq_list = numpy.array([])
    
    for fn in filename_list:
        current =  convertSequences( fn, dopad=True, doWhitening= doWhitening, maxlen=maxlen) 
        
        if len(seq_list)==0:
            seq_list = current 
        else:
            seq_list = numpy.dstack((seq_list, current) )

    return seq_list




###################################################################################################
# maniplation functions
###################################################################################################

def convertToBinaryMap( seq_list, categories ):

    binmap_indices = dict((b,i) for i,b in enumerate(categories))

    #print seq_list.shape

    new_seq_list = numpy.zeros( (seq_list.shape[0], seq_list.shape[1], len(categories)) )

    for l in range(seq_list.shape[0]):
        for t in range( len(seq_list[l]) ) :
            new_seq_list[l, t, binmap_indices[seq_list[l,t]] ] = 1.0

    return  new_seq_list 
    



def MakeSortedVariableSequenceArray( seq_list ):
    nfs = numpy.array(seq_list)

    SortedVariableSequenceArray = []

    for iseq in range(nfs.shape[1]):
        seq_stack = nfs[0,iseq]
        for ivar in range(1,nfs.shape[0]):
            seq_stack = numpy.vstack( (seq_stack,nfs[ivar,iseq]) )
        SortedVariableSequenceArray.append( seq_stack.T )#numpy.vstack((nfs[0,iseq], nfs[1,iseq])).T )

    SortedVariableSequenceArray.sort(cmp=lambda x,y: cmp(x.shape[0], y.shape[0]), reverse=True )

    return SortedVariableSequenceArray


def splitBySequenceLength( seq_list ):

    seq_list.sort(cmp=lambda x,y: cmp(x.shape[0], y.shape[0]), reverse=True )

    split_seq = []

    last_len = -1
    curr_len = -1
    curr_seq_list = []

    for seq in seq_list:
        if last_len == -1:
            last_len = seq.shape[0]

        curr_len = seq.shape[0]

        if curr_len != last_len:
            split_seq.append( numpy.array( curr_seq_list ) )
            curr_seq_list = []
            last_len = curr_len
            
        curr_seq_list.append( seq )

    if(len(curr_seq_list) > 0):
        split_seq.append( numpy.array( curr_seq_list ) )

    #print split_seq
    print split_seq[-1].shape, split_seq[-1]

    print split_seq[4].shape, split_seq[4]

    
    return split_seq














if __name__=="__main__":

    f = file('MakeData/test.pkl','r')
    trk_arr = cPickle.load(f)
    lab = cPickle.load(f)

    #print MakePaddedSequenceTensorFromListArray(trk_arr)[0,:,0]

    print trk_arr[:,2]
    
    sys.exit(0)

    #MakePaddedSequenceTensor([ 'sequences_sd0.txt', 'sequences_sz0.txt'])
    
    #gg = convertToBinaryMap( convertSequences( 'MakeData/sequences_grd.txt', dopad=True ), range(0,15) )

    ff =   splitBySequenceLength( MakeSortedVariableSequenceArray([convertSequences('MakeData/sequences_sd0.txt'), \
                                                                   convertSequences('MakeData/sequences_sz0.txt'), \
                                                                   convertToBinaryMap(convertSequences('MakeData/sequences_grd.txt'), range(0,15))] ))

    #print ff
    

    #print gg

    #print numpy.count_nonzero(gg)
        
