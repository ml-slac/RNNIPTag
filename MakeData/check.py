import numpy as np
import cPickle

f = None
f = file('Dataset_IP3D_pTFrac_dR_5m_CMix_MV2C20.pkl','r')

trk_arr_all = cPickle.load(f)
labels_all = cPickle.load(f)
f.close()

for i in range(len(labels_all)):
    if i % 100000 ==0:
        print "njet ", i

    if int(labels_all[i, 7]) != len(trk_arr_all[i][0]):
        print "not equal ", i, " jet"



    

