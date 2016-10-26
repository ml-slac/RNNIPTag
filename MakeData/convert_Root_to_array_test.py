# this import is required
import rootpy.tree
from rootpy.io import root_open
from ROOT import gDirectory, TLorentzVector, TVector3
  
#from rootpy.root2array import tree_to_ndarray
import root_numpy as rnp
import sys
import cPickle 
import numpy as np
import glob
import sorting_funcs 
from copy import deepcopy
from math import sqrt

Njets = 5000000
Nevents = int(1.0*Njets / 5.0)  #assume 5 jets per event

outfileName = "Dataset_IP3D_pTFrac_dR_5m_CMix_MV2C20.pkl"
outfileName = "Dataset_IP3D_pTFrac_dR_5m_CMix_MV2C20_SL0Sort.pkl"
outfileName = "Dataset_V47_IP3D_pTFrac_dphi_deta_5m.pkl"
outfileName = "Dataset_V47_onefile_comparison.pkl"
outfileName = "Dataset_V47_IP3D_pTFrac_d0_z0_5m.pkl"

#outfileName = "Dataset_V47_IP3D_pTFrac_dR_sl0order_5m.pkl"
#outfileName = "Dataset_V47_IP3D_pTFrac_dR_5m.pkl"
outfileName = "Dataset_V47_IP3D_pTFrac_dR_reverse_sd0order_5m.pkl"
#outfileName = "Dataset_V47_IP3D_pTFrac_dR_sv1_5m.pkl"

##################
print "- Making File List"

#ROOTfileNames = glob.glob("/atlas/local/BtagOptimizationNtuples/group.perf-flavtag.mc15_13TeV.410000.PowhegPythiaEvtGen_s2608_s2183_r7377_r7351.BTAGNTUP_V42cfull_Akt4EMTo/*.root*")
ROOTfileNames = glob.glob("/atlas/local/BtagOptimizationNtuples/V47/group.perf-flavtag.mc15_13TeV.410000.PowhegPythiaEvtGen_s2608_s2183_r7725_r7676.BTAGNTUP_V47_full_Akt4EMTo/*.root*")
#ROOTfileNames = glob.glob("/atlas/local/BtagOptimizationNtuples/V47/group.perf-flavtag.mc15_13TeV.410000.PowhegPythiaEvtGen_s2608_s2183_r7725_r7676.BTAGNTUP_V47_full_Akt4EMTo/group.perf-flavtag.8324358.Akt4EMTo._001543.root")
#"/atlas/local/BtagOptimizationNtuples/user.vdao.mc15_13TeV.410000.PowhegPythiaEvtGen_nonallhad.merge.AOD.e3698_s2608_s2183_r6630_r6264.BTAGNTUP_V9full_BTAGSTREAM.31148054/*.root*")

firstFile = True
EventSum = 0

out_trk_arr = None
out_label_arr = None
out_sv1_arr = None

for fname in ROOTfileNames:

    Nleft = 0
    if out_label_arr!= None and len(out_label_arr) > Njets:
        break

    Nleft = Nevents - EventSum
    if Nleft <= 0:
        break        
        
    with root_open(fname) as f:
        print "- opened file = ", fname
  
        tree = f.bTag_AntiKt4EMTopoJets

        EventSum = EventSum + (Nleft if Nleft < f.bTag_AntiKt4EMTopoJets.GetEntries() else f.bTag_AntiKt4EMTopoJets.GetEntries())

        ####################################################################################
        #per track info
        # comes out in crazy form.... lots of processing needed...
        # array of events, event = array of jets,  jet  = array of variable
        ####################################################################################
        print "- extracting per track info"
        sd0_raw =  rnp.tree2array(tree, "jet_trk_ip3d_d0sig",  stop=Nleft).flatten()
        sz0_raw =  rnp.tree2array(tree, "jet_trk_ip3d_z0sig",  stop=Nleft).flatten()
        d0_raw =  rnp.tree2array(tree, "jet_trk_d0",  stop=Nleft).flatten()
        z0_raw =  rnp.tree2array(tree, "jet_trk_z0",  stop=Nleft).flatten()
        grade_raw =  rnp.tree2array(tree, "jet_trk_ip3d_grade",  stop=Nleft).flatten()
        llr_raw =  rnp.tree2array(tree, "jet_trk_ip3d_llr",  stop=Nleft).flatten()
        pt_raw =  rnp.tree2array(tree, "jet_trk_pt",  stop=Nleft).flatten()
        eta_raw =  rnp.tree2array(tree, "jet_trk_eta",  stop=Nleft).flatten()
        theta_raw =  rnp.tree2array(tree, "jet_trk_theta",  stop=Nleft).flatten()
        phi_raw =  rnp.tree2array(tree, "jet_trk_phi",  stop=Nleft).flatten()

        print "- sorting per track info"
        sd0_arr = None
        sz0_arr = None
        d0_arr = None
        z0_arr = None
        grade_arr = None
        llr_arr = None
        pt_arr = None
        pTFrac_arr = None
        eta_arr = None
        phi_arr = None
        deta_arr = None
        dtheta_arr = None
        dphi_arr = None
        dR_arr = None
        for i in range(len(sd0_raw)):
            ################ sorting variable #######################
            #index_list = sorting_funcs.get_sort_index_list( sd0_raw[i].flatten()*sd0_raw[i].flatten()+sz0_raw[i].flatten()*sz0_raw[i].flatten(), sort_type="absrev" )
            #index_list = sorting_funcs.get_sort_index_list( sd0_raw[i].flatten(),  sort_type="absrev" )
            index_list = sorting_funcs.get_sort_index_list( sd0_raw[i].flatten(),  sort_type="abs" )
            ################ sorting variable #######################
            mask_list = sorting_funcs.get_neg_mask_list( sorting_funcs.sort_arrays_in_list(grade_raw[i].flatten(), index_list) )

            sd0_sort =   sorting_funcs.sort_arrays_in_list(sd0_raw[i].flatten(), index_list, mask_list)
            sz0_sort =   sorting_funcs.sort_arrays_in_list(sz0_raw[i].flatten(), index_list, mask_list)
            d0_sort  =   sorting_funcs.sort_arrays_in_list(d0_raw[i].flatten(), index_list, mask_list)
            z0_sort  =   sorting_funcs.sort_arrays_in_list(z0_raw[i].flatten(), index_list, mask_list)
            grade_sort = sorting_funcs.sort_arrays_in_list(grade_raw[i].flatten(), index_list, mask_list)
            llr_sort =   sorting_funcs.sort_arrays_in_list(llr_raw[i].flatten(), index_list, mask_list)
            pt_sort =    sorting_funcs.sort_arrays_in_list(pt_raw[i].flatten(), index_list, mask_list)
            eta_sort =   sorting_funcs.sort_arrays_in_list(eta_raw[i].flatten(), index_list, mask_list)
            theta_sort = sorting_funcs.sort_arrays_in_list(theta_raw[i].flatten(), index_list, mask_list)
            phi_sort =   sorting_funcs.sort_arrays_in_list(phi_raw[i].flatten(), index_list, mask_list)


            if i==0:
                sd0_arr =   sd0_sort
                sz0_arr =   sz0_sort
                d0_arr  =   d0_sort
                z0_arr  =   z0_sort
                grade_arr = grade_sort
                llr_arr =   llr_sort
                pt_arr =    pt_sort
                eta_arr =   eta_sort
                theta_arr = theta_sort
                phi_arr =   phi_sort

            else:
                sd0_arr = np.hstack( (sd0_arr, sd0_sort) )
                sz0_arr = np.hstack( (sz0_arr, sz0_sort) )
                d0_arr = np.hstack( (d0_arr, d0_sort) )
                z0_arr = np.hstack( (z0_arr, z0_sort) )
                grade_arr = np.hstack( (grade_arr, grade_sort) )
                llr_arr = np.hstack( (llr_arr, llr_sort) )
                pt_arr = np.hstack( (pt_arr, pt_sort) )
                eta_arr = np.hstack( (eta_arr, eta_sort) )
                theta_arr = np.hstack( (theta_arr, theta_sort) )
                phi_arr = np.hstack( (phi_arr, phi_sort) )


        ####################################################################################
        #lables array
        # comes out as array of arrays  (NOT 2D array, but array of arrays)
        # convert to tuple for easy stacking into single array
        ####################################################################################
        print "- extracting jet info"
        jet_flav =      np.hstack(tuple(rnp.tree2array(tree, "jet_LabDr_HadF",  stop=Nleft)))        
        jet_pt =        np.hstack(tuple(rnp.tree2array(tree, "jet_pt",  stop=Nleft)))        
        jet_eta =       np.hstack(tuple(rnp.tree2array(tree, "jet_eta", stop=Nleft)))        
        jet_m =       np.hstack(tuple(rnp.tree2array(tree, "jet_m", stop=Nleft)))        
        jet_theta =     2*np.arctan( np.exp( - jet_eta))
        jet_phi =       np.hstack(tuple(rnp.tree2array(tree, "jet_phi", stop=Nleft)))        
        jet_llr =       np.hstack(tuple(rnp.tree2array(tree, "jet_ip3d_llr", stop=Nleft)))        
        jet_ip3d_pb =   np.hstack(tuple(rnp.tree2array(tree, "jet_ip3d_pb",  stop=Nleft)))        
        jet_ip3d_pc =   np.hstack(tuple(rnp.tree2array(tree, "jet_ip3d_pc",  stop=Nleft)))        
        jet_ip3d_pu =   np.hstack(tuple(rnp.tree2array(tree, "jet_ip3d_pu",  stop=Nleft)))        
        jet_ip3d_ntrk = np.hstack(tuple(rnp.tree2array(tree, "jet_ip3d_ntrk", stop=Nleft)))        
        jet_mv2c20 = np.hstack(tuple(rnp.tree2array(tree, "jet_mv2c20",  stop=Nleft)))        
        jet_mv2c10 = np.hstack(tuple(rnp.tree2array(tree, "jet_mv2c10",  stop=Nleft)))        
        jet_aliveafterOR = np.hstack(tuple(rnp.tree2array(tree, "jet_aliveAfterOR",  stop=Nleft)))        
        jet_sv1_llr = np.hstack(tuple(rnp.tree2array(tree, "jet_sv1_llr", stop=Nleft)))        
        jet_JVT =        np.hstack(tuple(rnp.tree2array(tree, "jet_JVT",  stop=Nleft)))        
        jet_sv1_ntrkv =   np.hstack(tuple(rnp.tree2array(tree, "jet_sv1_ntrkv",  stop=Nleft)))        
        jet_sv1_n2t =   np.hstack(tuple(rnp.tree2array(tree, "jet_sv1_n2t",  stop=Nleft)))        
        jet_sv1_m =   np.hstack(tuple(rnp.tree2array(tree, "jet_sv1_m",  stop=Nleft)))        
        jet_sv1_efc =   np.hstack(tuple(rnp.tree2array(tree, "jet_sv1_efc",  stop=Nleft)))        
        jet_sv1_sig3d =   np.hstack(tuple(rnp.tree2array(tree, "jet_sv1_sig3d",  stop=Nleft)))        
        jet_sv1_normdist =   np.hstack(tuple(rnp.tree2array(tree, "jet_sv1_normdist",  stop=Nleft)))        
        
        pTFrac_arr = deepcopy(pt_arr)
        dR_arr = deepcopy(pt_arr)
        deta_arr = deepcopy(pt_arr)
        dtheta_arr = deepcopy(pt_arr)
        dphi_arr = deepcopy(pt_arr)

        for iJet in range(jet_pt.shape[0]):
            for iTrk in range(pTFrac_arr[iJet].shape[0]):
                pTFrac_arr[iJet][iTrk] = pTFrac_arr[iJet][iTrk]/jet_pt[iJet]
                dR_arr [iJet][iTrk] = sqrt( (eta_arr[iJet][iTrk]-jet_eta[iJet])**2+(phi_arr[iJet][iTrk]-jet_phi[iJet])**2 )
                deta_arr [iJet][iTrk] = eta_arr[iJet][iTrk]-jet_eta[iJet]
                dtheta_arr [iJet][iTrk] = theta_arr[iJet][iTrk]-jet_theta[iJet]
                dphi_arr [iJet][iTrk] = phi_arr[iJet][iTrk]-jet_phi[iJet]

        label_arr = np.dstack( (jet_flav, jet_pt, jet_eta, jet_llr, jet_ip3d_pb, jet_ip3d_pc, jet_ip3d_pu, jet_ip3d_ntrk, jet_mv2c20, jet_sv1_llr, jet_mv2c10, jet_JVT, jet_aliveafterOR) )[0]

        #trk_arr = np.dstack( (sd0_arr, sz0_arr, pTFrac_arr, dphi_arr, deta_arr, grade_arr, llr_arr, pt_arr) )[0] 
        #trk_arr = np.dstack( (sd0_arr, sz0_arr, pTFrac_arr, d0_arr, z0_arr, grade_arr, llr_arr, pt_arr) )[0] 
        trk_arr = np.dstack( (sd0_arr, sz0_arr, pTFrac_arr, dR_arr, grade_arr, llr_arr, pt_arr) )[0] 

        ###########################
        ## extract sv1 vertex info
        ###########################

        jet_njets = np.hstack(tuple(rnp.tree2array(tree, "njets",  stop=Nleft)))        
        PVx = np.hstack(tuple(rnp.tree2array(tree, "PVx",  stop=Nleft)))        
        PVy = np.hstack(tuple(rnp.tree2array(tree, "PVy",  stop=Nleft)))        
        PVz = np.hstack(tuple(rnp.tree2array(tree, "PVz",  stop=Nleft)))        
        jet_sv1_vtx_x_raw = rnp.tree2array(tree, "jet_sv1_vtx_x",  stop=Nleft).flatten()
        jet_sv1_vtx_y_raw = rnp.tree2array(tree, "jet_sv1_vtx_y",  stop=Nleft).flatten()
        jet_sv1_vtx_z_raw = rnp.tree2array(tree, "jet_sv1_vtx_z",  stop=Nleft).flatten()
        
        jet_sv1_L3d = np.zeros(sum(jet_njets))
        jet_sv1_dR = np.zeros(sum(jet_njets))
        jet_sv1_Lxy = np.zeros(sum(jet_njets))

        v_jet = TLorentzVector()
        pv2sv = TVector3()

        njet = 0
        for ievt in range(jet_njets.shape[0]):
            for ijet in range(jet_njets[ievt]):
                
                v_jet.SetPtEtaPhiM(jet_pt[ijet], jet_eta[ijet], jet_phi[ijet], jet_m[ijet])
                
                if jet_sv1_vtx_x_raw[ievt][ijet].shape[0]==1:
                    dx = jet_sv1_vtx_x_raw[ievt][ijet][0]-PVx[ievt]
                    dy = jet_sv1_vtx_y_raw[ievt][ijet][0]-PVy[ievt]
                    dz = jet_sv1_vtx_z_raw[ievt][ijet][0]-PVz[ievt]
                    pv2sv.SetXYZ(dx, dy, dz)
                    jetAxis = TVector3(v_jet.Px(), v_jet.Py(), v_jet.Pz())
                    
                    jet_sv1_L3d[ijet+njet] = sqrt(dx**2+ dy**2 + dz**2)
                    jet_sv1_Lxy[ijet+njet] = sqrt(dx**2+ dy**2)
                    jet_sv1_dR [ijet+njet] = pv2sv.DeltaR(jetAxis)
                    
                else:
                    jet_sv1_L3d[ijet+njet] = -100
                    jet_sv1_Lxy[ijet+njet] = -100
                    jet_sv1_dR[ijet+njet] = -1
                    
            njet += jet_njets[ievt]

        print "jet_sv1_l3d", jet_sv1_L3d
        print "jet_sv1_lxy", jet_sv1_Lxy
        print "jet_sv1_dR", jet_sv1_dR

        sv1_arr = np.dstack( (  jet_sv1_ntrkv, jet_sv1_n2t, jet_sv1_m, jet_sv1_efc, jet_sv1_sig3d, jet_sv1_normdist, jet_sv1_L3d, jet_sv1_Lxy, jet_sv1_dR  ) )[0]            

        if firstFile:
            out_trk_arr = trk_arr
            out_label_arr = label_arr
            out_sv1_arr = sv1_arr
                
            firstFile= False
        else:
            out_trk_arr = np.vstack( (out_trk_arr, trk_arr) )
            out_label_arr = np.vstack( (out_label_arr, label_arr) ) 
            out_sv1_arr = np.vstack( (out_sv1_arr, sv1_arr) ) 

#        print 'track pt', pTFrac_arr
#        print 'track sd0', sd0_arr 
#        print 'jet pt', jet_pt
#        print 'jet JVT', jet_JVT
#        print sv1_arr


print "- finished, saving"
outfile = file(outfileName, 'wb')
cPickle.dump(out_trk_arr, outfile, protocol=cPickle.HIGHEST_PROTOCOL)
cPickle.dump(out_label_arr, outfile, protocol=cPickle.HIGHEST_PROTOCOL)
cPickle.dump(out_sv1_arr, outfile, protocol=cPickle.HIGHEST_PROTOCOL)
outfile.close()

print "######### Summary #########"
print "OutFile = ", outfileName
print "Nevents = ", EventSum
print "Njets = ", len(out_label_arr)
print "###########################"
