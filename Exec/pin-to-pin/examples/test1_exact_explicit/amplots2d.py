## Example plot file
import os
import glob
import math
import yt
from plotall import plotGap
from plotall import plot1dList

def _Emag(field, data):
    return (
        (data['boxlib', 'Efieldx']**2+
         data['boxlib', 'Efieldy']**2+
         data['boxlib', 'Efieldz']**2)**(0.5)
    )

yt.add_field(
    name=("boxlib", "Efieldmag"),
    function=_Emag,
    sampling_type="local",
    units="",
)


#yt.enable_parallelism()

#plotdir = "hypersonic-simulations/med-mesh-250um-A"
plotdir = "atm-simulations/UPDATED-pelec_data-5lev-300K-1atm-cylindricalPinsFP-1p25mmh-250um-2p5mm-sigmoidPulse-15kVanode-adiabatic"
#plotdir = "hypersonic-simulations/med-mesh-250um-E"
#plotdir = "newCodeSheath/hypersonic-mesh-experiment"
#os.chdir(plotdir)


pltlist = glob.glob('plt?????')
pltlist = sorted(pltlist)
#pltlist = pltlist[1::5]
#fieldlist = ["n(E)","Temp","pressure","density"]
fieldlist = ["Efieldmag","n(E)","n(O2+)"]

pltfile = "plt00894"
field   = "Efieldmag"
ext = ".png"
cutplane = "z"
refL = 0.5  # [cm]
plotgrids = True
minGridLevel = 5
clocation = [0.99 ,0.3854370117187500 ,0.99]
#xyedge = [0.5 , 0.5]
xyedge = [0.0001 , 0.0001]
# Colormap and axes limit default

for ibool in [0,1]:
    cmapvar="jet"
    gridson = ibool
    zlims = []
    for i in pltlist:
       for j in fieldlist:
           print(i)
           zlims=[]
           logbool = False
           if j == "n(E)":
              logbool=True
              cmapvar="black_blueish"
              #zlims=[1e11, 1e18]
           elif j == "Temp":
              cmapvar="jet"
              zlims=[300, 1500]
        
           elif j == "avg_pressure":
              cmapvar="RdYlBu"
              #zlims=[0.5e5, 1.5e5]
    
           elif j == "density":
              cmapvar="Greys"
              #zlims=[0.0, 2.5e-3]
           
           elif j == "Efieldmag":
              cmapvar="Greys"
              #zlims=[0.0, 2.5e-3]
    
           cmapvar = "jet"
           #plotmax(i,j,logbool)
           plotGap(i,j,logbool,xyedge,cmapvar,zlims,gridson)
           #os.mkdir(j)
           #os.chdir(j)
           
           #plotelectrode(i,j,logbool)
    
