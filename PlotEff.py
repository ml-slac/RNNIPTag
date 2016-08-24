from ROOT import *
from math import sqrt
from optparse import OptionParser
from AtlasStyle import *
import array
import numpy as np


#EffFile = TFile("LEffCurveCompare_pT.root", 'read')
#EffFile = TFile("BEffCurveCompare_pT.root", 'read')
EffFile = TFile("ZihaoTest_FlatEffCurveCompare_Flavor0.root", 'read')
#EffFile = TFile("V42/ZihaoTest_FlatEffCurveCompare_Flavor0.root", 'read')

EffCurve_IP3D = EffFile.Get("EffCurvePt_IP3D")
EffCurve_SV1 = EffFile.Get("EffCurvePt_SV1")
EffCurve_MV2 = EffFile.Get("EffCurvePt_MV2")
EffCurve_RNN = EffFile.Get("EffCurvePt_RNN")
EffCurve_RNNSV1 = EffFile.Get("EffCurvePt_RNNSV1")
EffCurve_RNNMV2 = EffFile.Get("EffCurvePt_RNNMV2")


OUTPUT = TFile("OUTPUT.root", 'recreate')
Canv = TCanvas("EffComb", "EffComb", 0, 800, 0, 800)
Canv.cd()

EffCurves = [EffCurve_IP3D, EffCurve_SV1,  EffCurve_MV2, EffCurve_RNN, EffCurve_RNNSV1, EffCurve_RNNMV2]
label = ["IP3D", "SV1", "MV2", "RNN", "RNN + SV1", "RNN + MV2"]
colorlist = [2, 4, 8, 28, 51, 93, 30, 38, 41, 42, 46]

mg = TMultiGraph()

bins = [20, 50, 80, 120, 200, 300, 500]


def ConvertEffToGraph(effplot, bins, doEff=True):
    print effplot
    eff = []
    efferror = []
    for i in range(len(bins)):
        if doEff:
            eff.append(effplot.GetEfficiency(i+1))
            efferror.append(effplot.GetEfficiencyErrorLow(i+1))

            print eff
        else:
            eff.append(1./effplot.GetEfficiency(i+1))
            efferror.append( (effplot.GetEfficiencyErrorLow(i+1)/effplot.GetEfficiency(i+1))/effplot.GetEfficiency(i+1) )

    newgraph = TGraphErrors (len(bins), array.array('d', bins), array.array('d', eff), array.array('d', [0]*len(bins)), array.array('d', efferror))
    return newgraph
    #for i in range(len(bins)):
    #    newgraph.SetBinError(i+1, eff[i])

def GetRelativeRej(rejplot1, rejplot2, bins):
    rel = []
    relerror = []

    for i in range(len(bins)):
        rel.append( rejplot1.GetEfficiency(i+1)/rejplot2.GetEfficiency(i+1) )
        relerror.append( rel[i]*sqrt( (rejplot1.GetEfficiencyErrorLow(i+1)/rejplot1.GetEfficiency(i+1))**2 + (rejplot2.GetEfficiencyErrorLow(i+1)/rejplot2.GetEfficiency(i+1))**2 ) )

    newgraph = TGraphErrors (len(bins), array.array('d', bins), array.array('d', rel), array.array('d', [0]*len(bins)), array.array('d', relerror))
    return newgraph

legend = TLegend(0.5, 0.5, 0.75, 0.75)
legend_rel = TLegend(0.5, 0.5, 0.75, 0.75)
ROCs = []
mg = TMultiGraph()
mg_rel = TMultiGraph()

for i in range(len(EffCurves)):

    ROC = ConvertEffToGraph(EffCurves[i],bins, False)

    ROC.SetLineWidth(2)
    ROC.SetLineColor(colorlist[i])
    ROC.SetMarkerColor(colorlist[i])
    ROC.SetMarkerStyle(1)
    
    mg.Add(ROC)

    legend.AddEntry(ROC, label[i], "lp")

    ROCs.append(ROC)


#for i in range(3):
#
#    #ROC = GetRelativeRej(EffCurves[0], EffCurves[i+1], bins)
#    ROC = GetRelativeRej(EffCurves[0], EffCurves[i+1], bins)
#
#    ROC.SetLineWidth(2)
#    ROC.SetLineColor(2+i)
#    ROC.SetMarkerColor(2+i)
#    ROC.SetMarkerStyle(1)
#
#    
#    mg_rel.Add(ROC)
#
#    legend_rel.AddEntry(ROC, label[i+1], "lp")


mg.Draw("AL*")
mg.GetXaxis().SetTitle("b-jet p_{T} [GeV]")
mg.GetYaxis().SetTitle("l-jet Rejection")

#mg_rel.Draw("AL*")
#mg_rel.GetXaxis().SetTitle("l-jet p_{T} [GeV]")
#mg_rel.GetYaxis().SetTitle("Light Jet Relative Rjection")
#mg.GetXaxis().SetTitle("Number of tracks in b-jet")
#mg.GetYaxis().SetTitle("Efficienc")
#mg.GetXaxis().SetTitle("l-jet p_{T} [GeV]")
#mg.GetYaxis().SetTitle("Rejection")

legend.Draw("same")
OUTPUT.cd()
Canv.Write()

