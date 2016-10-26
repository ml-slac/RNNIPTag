# this import is required
import rootpy.tree
from rootpy.io import root_open
  
#from rootpy.root2array import tree_to_ndarray
import root_numpy as rnp

import sys

import cPickle 
import numpy as np




makeTRAINsample = False

normImage = False

addSpecVars = True

if makeTRAINsample:
    ROOTfileNames =  ['qcd_michael_25_50_rotated.root',    'qcd_michael_50_75_rotated.root',    'qcd_michael_75_IN_rotated.root', \
                      'signal_michael_25_50_rotated.root', 'signal_michael_50_75_rotated.root', 'signal_michael_75_IN_rotated.root' ]
    outfileName = 'alldata_'+('norm_' if normImage else '')+ 'TRAIN.pkl'
    
else:
    ROOTfileNames = ['TEST_qcd_skim_fixed_rotated.root', 'TEST_signal_skim_fixed_rotated.root']
    outfileName = 'alldata_'+('norm_' if normImage else '')+ 'TEST.pkl'



firstFile = True

for fname in ROOTfileNames:

    sampleType = -1
    if fname.find('signal') >= 0:
        sampleType = 1
    elif fname.find('qcd') >= 0:
        sampleType = 0

    if sampleType == -1:
        print "Could not determine sample type from file name"
        sys.exit(0)
        
    with root_open(fname) as f:
  
        tree = f.images
        

        
        dr_arr = np.array([ rnp.tree2array(tree, "jet_delta_R") ]).T
        im_arr = rnp.tree2array(tree, "image")
        target_arr = ( np.ones((im_arr.shape[0],1)) if sampleType == 1 else np.zeros((im_arr.shape[0],1)) )
        

        if normImage:
            im_arr = im_arr / np.linalg.norm(im_arr, axis=1)[:,None] # the [:,None] allows matrix / vector division to work


        arr = np.hstack( (target_arr, dr_arr, im_arr) )
    

        if addSpecVars:
            #branches = ["jet_m","jet_pt","tau_21"]
            m_arr = np.array([ rnp.tree2array(tree, "jet_m") ]).T
            pt_arr = np.array([ rnp.tree2array(tree, "jet_pt") ]).T
            t21_arr = np.array([ rnp.tree2array(tree, "tau_21") ]).T
            mpt_arr = np.hstack( (m_arr, pt_arr, t21_arr) )

            
        if firstFile:
            evt_arr = arr
            if addSpecVars:
                spec_arr = mpt_arr
                
            firstFile= False
        else:
            evt_arr = np.vstack( (evt_arr, arr) )
            if addSpecVars: 
                spec_arr = np.vstack( (spec_arr, mpt_arr) )


outfile = file(outfileName, 'wb')
cPickle.dump(evt_arr, outfile, protocol=cPickle.HIGHEST_PROTOCOL)
if addSpecVars:
    cPickle.dump(spec_arr, outfile, protocol=cPickle.HIGHEST_PROTOCOL)
outfile.close()


'''
For testing (100k each - let me know if you would like more)

/nfs/slac/g/atlas/u01/users/bnachman/JetImages/signal_skim_fixed_rotated.root 
/nfs/slac/g/atlas/u01/users/bnachman/JetImages/qcd_skim_fixed_rotated.root 

And for training (10k each)  Fisher:

/nfs/slac/g/atlas/u01/users/bnachman/JetImages/qcd_michael_25_50_rotated.root
/nfs/slac/g/atlas/u01/users/bnachman/JetImages/qcd_michael_50_75_rotated.root
/nfs/slac/g/atlas/u01/users/bnachman/JetImages/qcd_michael_75_IN_rotated.root

/nfs/slac/g/atlas/u01/users/bnachman/JetImages/signal_michael_25_50_rotated.root
/nfs/slac/g/atlas/u01/users/bnachman/JetImages/signal_michael_50_75_rotated.root
/nfs/slac/g/atlas/u01/users/bnachman/JetImages/signal_michael_75_IN_rotated.root

Note that these are *not normalized* so if that is needed by Fisher, you will need to do it yourself.
'''
