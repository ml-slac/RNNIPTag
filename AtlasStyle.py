from ROOT import gROOT,gPad,gStyle,TCanvas,TFile,TLine,TLatex,TAxis,TLegend,TPostScript
from ROOT import TH2D,TArrow,TCut,TPad,TPaveText,TGraph,TGraphAsymmErrors
from ROOT import TGraph2D,TStyle,TBranch,gSystem,gDirectory
from ROOT import TPave,TPaveStats

atlasStyle= TStyle("ATLAS","Atlas style")

# use plain black on white colors
icol=0
atlasStyle.SetFrameBorderMode(icol)
atlasStyle.SetCanvasBorderMode(icol)
atlasStyle.SetPadBorderMode(icol)
atlasStyle.SetPadColor(icol)
atlasStyle.SetCanvasColor(icol)
atlasStyle.SetStatColor(icol)
#atlasStyle.SetFillColor(icol)

# set the paper & margin sizes
atlasStyle.SetPaperSize(20,26)
atlasStyle.SetPadTopMargin(0.1)
atlasStyle.SetPadRightMargin(0.05)
atlasStyle.SetPadBottomMargin(0.16)
atlasStyle.SetPadLeftMargin(0.12)

# use large fonts
font=42
tsize=0.05
atlasStyle.SetTextFont(font)


atlasStyle.SetTextSize(tsize)
atlasStyle.SetLabelFont(font,"x")
atlasStyle.SetTitleFont(font,"x")
atlasStyle.SetLabelFont(font,"y")
atlasStyle.SetTitleFont(font,"y")
atlasStyle.SetLabelFont(font,"z")
atlasStyle.SetTitleFont(font,"z")

atlasStyle.SetLabelSize(tsize,"x")
atlasStyle.SetTitleSize(tsize,"x")
atlasStyle.SetLabelSize(tsize,"y")
atlasStyle.SetTitleSize(tsize,"y")
atlasStyle.SetLabelSize(tsize,"z")
atlasStyle.SetTitleSize(tsize,"z")


#use bold lines and markers
atlasStyle.SetMarkerStyle(20)
atlasStyle.SetMarkerSize(0)
atlasStyle.SetHistLineWidth(2)
atlasStyle.SetLineStyleString(2,"[12 12]") # postscript dashes

#get rid of X error bars and y error bar caps
#atlasStyle.SetErrorX(0.001)

#do not display any of the standard histogram decorations
atlasStyle.SetOptTitle(0)
#atlasStyle.SetOptStat(1111)
atlasStyle.SetOptStat(0)
#atlasStyle.SetOptFit(1111)
atlasStyle.SetOptFit(0)

# put tick marks on top and RHS of plots
atlasStyle.SetPadTickX(1)
atlasStyle.SetPadTickY(1)

gROOT.SetStyle("Plain")

#gStyle.SetPadTickX(1)
#gStyle.SetPadTickY(1)
gROOT.SetStyle("ATLAS")
gROOT.ForceStyle() 
gStyle.SetOptTitle(0)
gStyle.SetOptStat(0) 
gStyle.SetOptFit(0) 
# overwrite atlas styles

atlasStyle.SetMarkerSize(1.0)
atlasStyle.SetPadLeftMargin(0.14)  
atlasStyle.SetPadRightMargin(0.03)    
atlasStyle.SetPadBottomMargin(0.12)     
atlasStyle.SetPadTopMargin(0.05)  
atlasStyle.SetFrameFillColor(0)

def ATLASLabel(x,y,shift,studytype,color=1): 
  l=TLatex(x, y, "ATLAS") 
  l.SetNDC()
  l.SetTextFont(72)
  l.SetTextColor(color)
  l.DrawLatex(x,y,"ATLAS")
  if (studytype != ""):
    p=TLatex(x+shift, y, studytype)
    p.SetNDC();
    p.SetTextFont(42)
    p.SetTextColor(color)
    p.DrawLatex(x+shift,y,studytype);


def ATLASVersion(version="1.0",x=0.88,y=0.975,color=1): 
  if (version): 
    l=TLatex() 
    l.SetTextAlign(22)
    l.SetTextSize(0.04)
    l.SetNDC()
    l.SetTextFont(72)
    l.SetTextColor(color)
    l.DrawLatex(x,y,versionString)

def myText(x,y,color=1,size=0.08,text=""): 
  l=TLatex()
  l.SetTextSize(size); 
  l.SetNDC();
  l.SetTextColor(color);
  l.DrawLatex(x,y,text);


def ATLAS_LABEL(x,y,color=1): 
  l=TLatex() 
  l.SetNDC()
  l.SetTextFont(72)
  l.SetTextColor(color);
  l.DrawLatex(x,y,"ATLAS");


def myBoxText(x,y,boxsize,mcolor,text): 
  tsize=0.06;

  l=TLatex(); l.SetTextAlign(12);  
  l.SetNDC();
  l.DrawLatex(x,y,text);

  y1=y-0.25*tsize;
  y2=y+0.25*tsize;
  x2=x-0.3*tsize;
  x1=x2-boxsize;
  mbox= TPave(x1,y1,x2,y2,0,"NDC");

  mbox.SetFillColor(mcolor);
  mbox.SetFillStyle(1001);
  mbox.Draw();

  mline=TLine()
  mline.SetLineWidth(4)
  mline.SetLineColor(1)
  mline.SetLineStyle(1)
  y=(y1+y2)/2.
  mline.DrawLineNDC(x1,y,x2,y)
