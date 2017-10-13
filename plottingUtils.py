import AtlasStyle as Atlas
import ROOT
import array
import cPickle
import json
import numpy as np
from LaurenColor import *

colorlist = colorind

# signal should be in format [ array1, array2, array3 ], where each array represents one approach
# Each array should be an array of scores per data, aligned with other arrays
def getROC( signal, background, label, cut_start=None, cut_end=None, outputName="myROC.root", Rejection="l", omission = []):
	ROCList = []
	markerlist = [21, 8 , 22, 23]
	print "get roc ", label
	for ivar in range(len(signal)):
		s_sort = np.sort( signal[ivar] )
		b_sort = np.sort( background[ivar] )

		#c_start=(0.0 if cut_start==None else cut_start)
		#c_end=  (1.0 if cut_end==None else cut_end)
		
		print s_sort, b_sort

		for i in range(s_sort.shape[0]):
			if s_sort[i] == float("Inf"):
				s_sort[i] = 100000
			if s_sort[i] == float("-Inf"):
				s_sort[i] = -1000000

		for i in range(b_sort.shape[0]):
			if b_sort[i] == float("Inf"):
				b_sort[i] = 100000
			if b_sort[i] == float("-Inf"):
				b_sort[i] = -1000000

		c_start=np.min( (s_sort[0], b_sort[0]) )
		c_end=  np.max( (s_sort[len(s_sort)-1], b_sort[len(b_sort)-1]) )

		if c_start==-float('inf'):
			c_start = -2*c_end

		print label[ivar], "min(", s_sort[0],  b_sort[0],  ")=", c_start
		print label[ivar], "max(", s_sort[-1], b_sort[-1], ")=", c_end

		s_eff=[]
		b_rej=[]

		n_points = 1000
		c_delta = (1.0*c_end - 1.0*c_start) / (1.0*n_points)
		for i in range(1000):
			cut = c_start + i*1.0*c_delta
			s_eff.append( 1.0*np.count_nonzero( s_sort > cut ) / (1.0*len(s_sort))  )
			
			b_count = np.count_nonzero( b_sort > cut )
			b_rej.append(  (1.0*len(b_sort)) / (1.0 if b_count==0 else (1.0*b_count))  )


		ROC = ROOT.TGraph(n_points, array.array('d', s_eff), array.array('d', b_rej))
		ROC.SetName("ROC_%i" % (ivar))

		ROCList.append(ROC)

	f = ROOT.TFile(outputName, "update")

	canvas = ROOT.TCanvas("ROC_Overlay", "ROC_Overlay", 800, 600)
	canvas.cd()

	mg = ROOT.TMultiGraph()

	legend = ROOT.TLegend(0.5, 0.5, 0.75, 0.75)

	for i in range(len(ROCList)):
		if i in omission:
			continue
		ROC = ROCList[i]
		ROC.SetLineWidth(3)
		ROC.SetLineColor(colorlist[i])
		ROC.SetMarkerColor(colorlist[i])
		ROC.SetMarkerSize(0)
		ROC.SetMarkerStyle(0)
		ROC.SetLineStyle(i+1)
		
		mg.Add(ROC)
		if omission == []:
			legend.AddEntry(ROC, label[i], "lp")

		f.WriteTObject(ROC, "ROC_"+label[i], "Overwrite")

	if omission!=[]:
		legend.AddEntry(ROCList[1], label[1], "lp")
		legend.AddEntry(ROCList[4], label[4], "lp")
		legend.AddEntry(ROCList[5], label[5], "lp")
		legend.AddEntry(ROCList[2], label[2], "lp")
		

	mg.Draw("AL")
	mg.GetXaxis().SetTitle("b-jet efficiency, #varepsilon_{b}")
	if Rejection == "l":
		mg.GetYaxis().SetTitle("light-jet rejection, 1/#varepsilon_{l}")
	if Rejection == "c":
		mg.GetYaxis().SetTitle("c-jet rejection, 1/#varepsilon_{c}")

	legend.Draw("same")
	Atlas.ATLASLabel(0.2, 0.88,0.13, "Simulation Internal",color=1)
	Atlas.myText(0.2, 0.81 ,color=1, size=0.04,text="#sqrt{s}=13 TeV, t#bar{t}") 
	Atlas.myText(0.2, 0.75 ,color=1, size=0.04,text="p_{T}>20 GeV, |#eta|<2.5") 
	#Atlas.myText(0.2, 0.69 ,color=1, size=0.04,text="Rel21") 

	#canvas.Update()
	canvas.Draw()

	f.WriteTObject(canvas, canvas.GetName(), "Overwrite")

	f.Close()

	return (ROCList, canvas)



###############################################################################################################

# show some specific feature (usually the training Loss) as function of epoch
# support multiple curve comparison
# histories = [ array1, array2, ... ]
def getTrainingCurve( histories,  labels, outputName="myTrainingCurve.root"):
	f = ROOT.TFile(outputName, "update")
	legend = ROOT.TLegend(0.5, 0.5, 0.75, 0.75)
	mg = ROOT.TMultiGraph()

	TrainCurveList = []
	localcolor = colorlist

	for i in range(len(histories)):
		history = histories[i]

		x_list = range(len(history))
		y_list = history

		trainingCurve = ROOT.TGraph(len(x_list), array.array('d', x_list), array.array('d', y_list))
		TrainCurveList.append(trainingCurve)

		trainingCurve.SetName("TrainCurve_"+labels[i])
		trainingCurve.SetLineWidth(2)
		trainingCurve.SetLineColor(localcolor[i])
		trainingCurve.SetMarkerColor(localcolor[i])

		mg.Add(trainingCurve)
		legend.AddEntry(trainingCurve, labels[i], "lp")
		f.WriteTObject(trainingCurve, "TrainCurve_%i" % (i), "Overwrite")

	canvas = ROOT.TCanvas("TrainingCurve_Overlay", "TrainingCurve_Overlay", 800, 600)
	canvas.cd()

	mg.Draw("AL*")
	legend.Draw()

	canvas.Update()

	f.WriteTObject(canvas, canvas.GetName(), "Overwrite")

	f.Close()

	return (TrainCurveList, canvas)


def compareOptimizater():
	model_list = [
	               "lstm2hidden",
	               "lstm2hiddenAdam",
	             ]
	feature = [
	             "loss",
	             "loss",
	          ]

	labels = [
	           "2 LSTM hidden-layer. RMSprop. Train.",
	           "2 LSTM hidden-layer. Adam. Train.",
	         ]

	if type(feature) == str:
		feature = [feature]*(len(labels))

	history_list = []
	for i in range(len(model_list)):
		fileName = model_list[i]+"History.json"
		print "Loading %s ..." % (fileName)

		f = open(fileName, "r")
		history = json.load(f)
		history_list.append(history[feature[i]])

	print "Plotting Training Curves ... "
	getTrainingCurve(history_list, labels, "TrainCurveOptimizer.root")


def compareRNN():
	model_list = [
	               "lstm1hiddenAdam",
	               "lstm1hiddenAdam",
	               "lstm2hiddenAdam",
	               "lstm2hiddenAdam",
	             ]
	feature = [
	             "loss",
	             "val_loss",
	             "loss",
	             "val_loss",
	          ]

	labels = [
	           "1 LSTM hidden-layer. Adam. Train.",
	           "1 LSTM hidden-layer. Adam. Valid.",
	           "2 LSTM hidden-layer. Adam. Train.",
	           "2 LSTM hidden-layer. Adam. Valid.",
	         ]

	if type(feature) == str:
		feature = [feature]*(len(labels))

	history_list = []
	for i in range(len(model_list)):
		fileName = model_list[i]+"History.json"
		print "Loading %s ..." % (fileName)

		f = open(fileName, "r")
		history = json.load(f)
		history_list.append(history[feature[i]])

	print "Plotting Training Curves ... "
	getTrainingCurve(history_list, labels, "TrainCurveCrossValidation.root")

################################################################################################################################

def getPDF(name, scoreList):
	h = ROOT.TH1D(name, name, 12000, -0.1, 1.1)
	h.Sumw2()

	for isample in range(scoreList.shape[0]):
		score = scoreList[isample]
		h.Fill(score)

	return h


################################################################################################################################

# produce efficiency curve as function of "var".
# scoreList should be the list of either signal or background (in numpy array format)
# varList should be aligned list of variable for same sample (in numpy array format)
# label should be pair of (histName, displayLabel)
def getEffCurve(scoreList, varList, label, bins, scoreCut = None, eff_target = 0.7):
	# initialize histogram
	histName, displayLabel = label

	h_base   = ROOT.TH1D(histName+"_beforeCut", histName+"_beforeCut", len(bins)-1, array.array('d', bins))
	h_base.Sumw2()

	h_select = ROOT.TH1D(histName+"_afterCut", histName+"_afterCut", len(bins)-1, array.array('d', bins)) 
	h_select.Sumw2()

	# determine cuts
	if scoreCut is None:
		scoreCut = getCutValue(scoreList, eff_target)

	print "score Cut", scoreCut

	# now loop over data points
	for isampe in range(scoreList.shape[0]):
		score = scoreList[isampe]
		pt = varList[isampe]

		h_base.Fill(pt)
		if score > scoreCut:
			h_select.Fill(pt)

	# get efficiency curve
	heff = ROOT.TEfficiency(h_select, h_base)
	heff.SetNameTitle(histName, displayLabel)

	return heff

def getFixEffCurve(scoreList, varList, label, bins, fix_eff_target, scoreCutList=None, onlyReturnCutList=False):
        # get pt-dependent cut in order to reach a fixed efficiency for each pT bin                                                                                                                                                                                           
	print 'varlist ', varList
	print 'scorelist ', scoreList
	print "ptbins", bins

        if scoreCutList is None:
                scoreCutList = []
		for ibin in range(len(bins)-1):
			ptmin = bins[ibin]
                        ptmax = bins[ibin+1]

			scoreList_ptslice = scoreList[ np.logical_and(varList>=ptmin, varList<ptmax) ]
			scoreCutList.append(getCutValue(scoreList_ptslice, fix_eff_target))

        if onlyReturnCutList:
		return scoreCutList

	histName, displayLabel = label



	h_base   = ROOT.TH1D(histName+"_beforeCut", histName+"_beforeCut", len(bins)-1, array.array('d', bins))
        h_base.Sumw2()

	h_select = ROOT.TH1D(histName+"_afterCut", histName+"_afterCut", len(bins)-1, array.array('d', bins))
	h_select.Sumw2()
        for isampe in range(scoreList.shape[0]):
                score = scoreList[isampe]
                pt = varList[isampe]

                if pt < bins[0]: continue
                if pt >= bins[-1]: continue

                ptbin = np.digitize(pt, bins)-1
                scoreCut = scoreCutList[ptbin]

                h_base.Fill(pt)
                if score > scoreCut:
                        h_select.Fill(pt)

        heff = ROOT.TEfficiency(h_select, h_base)
        heff.SetNameTitle(histName, displayLabel)

        return heff


def getLRejCurveFixedEff(scoreList, varList, LightscoreList, LightvarList, label, bins, scoreCut = None, eff_target = 0.7):
	# initialize histogram
	histName, displayLabel = label

	h_base   = ROOT.TH1D(histName+"_beforeCut", histName+"_beforeCut", len(bins)-1, array.array('d', bins))
	h_base.Sumw2()

	h_select = ROOT.TH1D(histName+"_afterCut", histName+"_afterCut", len(bins)-1, array.array('d', bins)) 
	h_select.Sumw2()

	# determine cuts for bjets
	if scoreCut is None:
		scoreCut = getCutValue(scoreList, eff_target)

	# now loop over data points of ljets
	for isampe in range(LightscoreList.shape[0]):
		score = LightscoreList[isampe]
		pt = LightvarList[isampe]

		h_base.Fill(pt)
		if score > scoreCut:
			h_select.Fill(pt)

	# get efficiency curve
	heff = ROOT.TEfficiency(h_select, h_base)
	heff.SetNameTitle(histName, displayLabel)

	return heff

def getCutValue(disc, eff_target):
	return np.sort(disc)[ int((1.0-eff_target)*len(disc)) ]

# each item in approachList should be (scoreList, varList, label) for each approach
def MultipleEffCurve(outputName, approachList, bins, scoreCut = None, eff_target = 0.7):

	fout = ROOT.TFile(outputName, "recreate")

	for scoreList, varList, label in approachList:
		heff = getEffCurve(scoreList, varList, label, bins, scoreCut, eff_target)
		fout.WriteTObject(heff, heff.GetName(), "Overwrite")

	fout.Close()


# each item in approachList should be (scoreList, varList, LightscoreList, LightvarList, label) for each approach
def MultipleRejCurve(outputName, approachList, bins, scoreCut = None, eff_target = 0.7):

	fout = ROOT.TFile(outputName, "update")

	for scoreList, varList, LightscoreList, LightvarList, label in approachList:
		heff = getLRejCurveFixedEff(scoreList, varList, LightscoreList, LightvarList, label, bins, scoreCut, eff_target)
		fout.WriteTObject(heff, heff.GetName(), "Overwrite")

	fout.Close()

def ConvertEffToGraph(effplot, bins, doEff=True):
    print effplot
    eff = []
    efferror = []
    for i in range(len(bins)):
        if doEff:
            eff.append(effplot.GetEfficiency(i+1))
            efferror.append(effplot.GetEfficiencyErrorLow(i+1))

        else:
		try :
			eff.append(1./effplot.GetEfficiency(i+1))
			efferror.append( (effplot.GetEfficiencyErrorLow(i+1)/effplot.GetEfficiency(i+1))/effplot.GetEfficiency(i+1) )
			#efferror.append(0)
		except ZeroDivisionError:
			eff.append(0)
			efferror.append( 0)

    plotbins =  [35, 70, 120, 225, 400]
    bins_width = [15, 20, 30, 75, 100]
			
    newgraph = ROOT.TGraphErrors (len(bins), array.array('d', plotbins), array.array('d', eff), array.array('d', bins_width), array.array('d', efferror))
    return newgraph


# here approachList also includes scoreCutList, which should be consistent with bins                  
def MultipleFlatEffCurve(outputName, approachList, bins, binslong,flav = "L"):
	markerlist = [21, 8 , 22, 23, 29, 34]
        fout = ROOT.TFile(outputName, "recreate")
	fout.cd()
	Canv = ROOT.TCanvas("EffComb", "EffComb", 0, 800, 0, 800)
	Canv.cd()

	EffCurves = []
        for scoreList, varList, label, scoreCutList in approachList:
                heff = getFixEffCurve(scoreList, varList, label, binslong, fix_eff_target=0.7, scoreCutList=scoreCutList, onlyReturnCutList=False)
		EffCurves.append(heff)

	legend = ROOT.TLegend(0.5, 0.5, 0.75, 0.75)
	legend_rel = ROOT.TLegend(0.5, 0.5, 0.75, 0.75)
	ROCs = []
	mg = ROOT.TMultiGraph()

	for i in range(len(EffCurves)):

		ROC = ConvertEffToGraph(EffCurves[i],bins, False)
		ROC.SetLineWidth(2)
		ROC.SetLineColor(colorlist[i])
		ROC.SetMarkerColor(colorlist[i])
		ROC.SetMarkerSize(1)
		ROC.SetMarkerStyle(markerlist[i])
		ROC.SetLineStyle(i+1)
    
		mg.Add(ROC)
		
		legend.AddEntry(ROC, approachList[i][2][1], "lp")

		ROCs.append(ROC)

	mg.Draw("AP")
	mg.GetXaxis().SetTitle("b-jet p_{T} [GeV]")
	if flav == "L":
		mg.GetYaxis().SetTitle("light-jet Rejection, 1/#varepsilon_{l}")
	if flav == "C":
		mg.GetYaxis().SetTitle("c-jet Rejection,  1/#varepsilon_{c}")
	legend.Draw("same")
	Atlas.ATLASLabel(0.2, 0.88,0.13, "Simulation Internal",color=1)
	Atlas.myText(0.2, 0.81 ,color=1, size=0.04,text="#sqrt{s}=13 TeV, t#bar{t}") 
	Atlas.myText(0.2, 0.75 ,color=1, size=0.04,text="p_{T}>20 GeV, |#eta|<2.5") 
	Atlas.myText(0.2, 0.69 ,color=1, size=0.04,text="Flat 70% b-tagging WP") 

	Canv.Write()

	fout.Close()
####################################################################################################



def test():
	model_list = [
	               "lstm2hiddenAdam",
	               "lstm2hiddenAdam",
	             ]
	feature = [
	             "loss",
	             "val_loss",
	          ]

	labels = [
	           "2 LSTM hidden-layer. Adam. Train.",
	           "2 LSTM hidden-layer. Adam. Valid.",
	         ]

	if type(feature) == str:
		feature = [feature]*(len(labels))

	history_list = []
	for i in range(len(model_list)):
		fileName = model_list[i]+"History.json"
		print "Loading %s ..." % (fileName)

		f = open(fileName, "r")
		history = json.load(f)
		history_list.append(history[feature[i]])

	print "Plotting Training Curves ... "
	getTrainingCurve(history_list, labels)


			








