#!/usr/bin/env python


#this is a script to make some dummy ROOT histograms with a pretty color
#palette.  The colors are specified in RGB.
#You can use https://color.adobe.com/explore to find your
#pretty color palette.

import ROOT


###A bunch of style setup for a certain experiment

aStyle= ROOT.TStyle("aStyle","A style")

# use plain black on white colors
icol=0
aStyle.SetFrameBorderMode(icol)
aStyle.SetCanvasBorderMode(icol)
aStyle.SetPadBorderMode(icol)
aStyle.SetPadColor(icol)
aStyle.SetCanvasColor(icol)
aStyle.SetStatColor(icol)
#aStyle.SetFillColor(icol)

# set the paper & margin sizes
aStyle.SetPaperSize(20,26)
aStyle.SetPadTopMargin(0.05)
aStyle.SetPadRightMargin(0.05)
aStyle.SetPadBottomMargin(0.16)
aStyle.SetPadLeftMargin(0.12)

# use large fonts
font=42
tsize=0.05
aStyle.SetTextFont(font)


aStyle.SetTextSize(tsize)
aStyle.SetLabelFont(font,"x")
aStyle.SetTitleFont(font,"x")
aStyle.SetLabelFont(font,"y")
aStyle.SetTitleFont(font,"y")
aStyle.SetLabelFont(font,"z")
aStyle.SetTitleFont(font,"z")

aStyle.SetLabelSize(tsize,"x")
aStyle.SetTitleSize(tsize,"x")
aStyle.SetLabelSize(tsize,"y")
aStyle.SetTitleSize(tsize,"y")
aStyle.SetLabelSize(tsize,"z")
aStyle.SetTitleSize(tsize,"z")

#use bold lines and markers
aStyle.SetMarkerStyle(20)
aStyle.SetMarkerSize(1.2)
aStyle.SetHistLineWidth(2)
aStyle.SetLineStyleString(2,"[12 12]") # postscript dashes

#get rid of X error bars and y error bar caps
#aStyle.SetErrorX(0.001)

#do not display any of the standard histogram decorations
aStyle.SetOptTitle(0)
#aStyle.SetOptStat(1111)
aStyle.SetOptStat(0)
#aStyle.SetOptFit(1111)
aStyle.SetOptFit(0)

# put tick marks on top and RHS of plots
aStyle.SetPadTickX(1)
aStyle.SetPadTickY(1)

ROOT.gROOT.SetStyle("Plain")

#gStyle.SetPadTickX(1)
#gStyle.SetPadTickY(1)
ROOT.gROOT.SetStyle("ATLAS")
ROOT.gROOT.ForceStyle() 
ROOT.gStyle.SetOptTitle(0)
ROOT.gStyle.SetOptStat(0) 
ROOT.gStyle.SetOptFit(0) 

aStyle.SetMarkerSize(1.0)
aStyle.SetPadLeftMargin(0.14)  
aStyle.SetPadRightMargin(0.03)    
aStyle.SetPadBottomMargin(0.12)     
aStyle.SetPadTopMargin(0.02)  
aStyle.SetFrameFillColor(0)
aStyle.SetLegendBorderSize(0)


#Setup color palette
#gettysburg: https://color.adobe.com/Gettysburg-color-theme-209416/?showPublished=true
#redish,deep-purple,grey,blue,cream in RGB
colors =[(247,122,82),(150,45,62),(52,136,153),(52,54,66),(151,156,156),(13,122,49),(106,100,3),(183,174,9),(99, 136, 66), (164,154,135), (7,67,87), (22,72,89)]

#times changing nhttps://color.adobe.com/Times-Changing-color-theme-1536595/edit/?copy=true
#colors =[(51,37,50),(100,77,82),(247,122,82),(255,151,79),(164,154,135)]
#flat design https://color.adobe.com/Flat-design-colors-1-color-theme-3044245/edit/?copy=true
#colors =[(202,45,36),(168,61,70),(7,67,87),(22,72,89),(46,68,94)]

tcolors=[]
colorind=[]
colorIndBase = 2000;

#make custom root TColor objects
for i,c in enumerate(colors):
    tcolors.append(ROOT.TColor(colorIndBase+i,float(c[0])/255.0,float(c[1])/255.0,float(c[2])/255.0))
    colorind.append(colorIndBase+i)

#make a nice color for an error band (matches to gettysburg)
errcolor = ROOT.TColor(colorIndBase-1,255./255.,211./255.,78./255.)

