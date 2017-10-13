# this import is required
import rootpy.tree
from rootpy.io import root_open
  
#from rootpy.root2array import tree_to_ndarray
import root_numpy as rnp

import sys

import cPickle 
import numpy as np

import glob

def sort_arrays_in_list(m, index_list, mask_list=[], printInfo=False):
    out_m = np.copy(m)

    if printInfo:
        print m
        print range(len(m))
        print mask_list
        print (mask_list != [])
    
#    for i in range(len(m)):
#        out_m[i] = m[i][index_list[i]]
#
#        if printInfo:
#            print m[i]
#            print m[i][index_list[i]]
#            print mask_list[i]
#
#
#        if len(mask_list) != 0:
#            out_m[i] = out_m[i][ mask_list[i] ]
#
#
#            if printInfo:
#                print out_m[i]
#                if np.count_nonzero( (out_m[i] < 0) ) > 0:
#                    print m.shape, len(index_list), len(mask_list)
#                    print m[i][index_list[i]]
#                    print mask_list[i]
#                    print out_m[i]

    out_m = m[index_list]

    return out_m


def get_sort_index_list(m, sort_type=None):
    index_list = []

    reverse_index = (sort_type.find("rev")!=-1)

#    for ientry in m:
#        index = None
#        
#        if sort_type == None or sort_type == "rev":
#            index = np.argsort(ientry)
#
#        elif sort_type.find("abs") != -1:
#            index = np.argsort(np.abs(ientry))
#
#        else:
#            print "get_sort_index_list: sort_type",sort_type,"not recognized"
#            sys.exit(0)
#        
#        index_list.append( index[::-1] if reverse_index else index )

    index = None
        
    if sort_type == None or sort_type == "rev":
        index = np.argsort(m)

    elif sort_type.find("abs") != -1:
        index = np.argsort(np.abs(m))

    else:
        print "get_sort_index_list: sort_type",sort_type,"not recognized"
        sys.exit(0)
        
    index_list.append( index[::-1] if reverse_index else index )


    return index_list

def get_neg_mask_list(m):
    out_m = np.copy(m)
    for i in range(len(m)):
        out_m[i] = (m[i] >= 0)

    return out_m
    



def sort_rows_of_matrix(m, index=None, reverse_sort = None):

    if len(m.shape)!=2:
        print "sort_rows_of_matrix: matrix must be of dimension 2!"
        sys.exit(0)
    
    if index == None:
        index = np.argsort(m, axis=1)
        if reverse_sort:
            index = index[:,::-1]

    if index.shape != m.shape:
        print "sort_rows_of_matrix: matrix and index have different size!"
        sys.exit(0)

    ##########################################################################################
    #numpy magic... make a 3D list of 2D indices of the desired order.
    # so 3D list will be [ [[0,0,0,...],[ 1,1,1,...],...]], index ]
    #first matrix gives row coordinate (all elements stay to same row)
    #second is the new column coordinate (i.e. the desired sorted position within the row)
    ##########################################################################################
    return m[ [ i*np.ones(index.shape[1]) for i in range(index.shape[0])], index]


def get_matrix_sort_index(m, sort_type=None, m2 = None):
        
    if len(m.shape)!=2:
        print "get_matrix_sort_index: matrix must be of dimension 2!"
        sys.exit(0)

    reverse_index = (sort_type.find("rev")!=-1)
    index = None
    
    if sort_type == None or sort_type == "rev":
        index = np.argsort(m)

    elif sort_type.find("abs") != -1:
        index = np.argsort(np.abs(m))

    elif sort_type.find("quad") != -1:
        if m.shape != m2.shape:
            print "get_matrix_sort_index: matrices must have same shape when doing QUAD sort"
            sys.exit(0)
        m3 = np.sqrt( np.power(m,2) + np.power(m2,2) )
        index = np.argsort(m3)
    
    return (index[:,::-1] if reverse_index else index)
