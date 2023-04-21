## Example plot file
import os
import glob
import math
import yt
import numpy as np
from matplotlib import pyplot as plt
from plotall import plotGap
from plotall import plot1dList
import os
if os.path.exists("data.txt"):
  os.remove("data.txt")
else:
  print("The file does not exist")

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

def _qn(field, data):
    return (
        (data['boxlib', 'n(O2+)']-
         data['boxlib', 'n(E)']
    )
)
yt.add_field(
    name=("boxlib", "qden"),
    function=_qn,
    sampling_type="local",
    units="",
)


pltlist = glob.glob('plt?????')
pltlist = sorted(pltlist)
pltlist = pltlist[0::2]
fieldlist = ["Efieldmag","n(E)"]

t = []
Emaxloc = []
Emax = []
nemax = []
ne_avg = []
q_tot = []

ext = ".png"

xyedge = [1.25 , 1.25]
dx = 1e-6*1e4
ystart = 0.6
yend = 1.25
## 1-d FIGS data and plot
for i in pltlist:
   for j in fieldlist:
       print(i)
       zlims=[]
       logbool = False
       cmapvar = "jet"
       field = j
       start1d = [xyedge[0]-dx, ystart, xyedge[1]-dx]
       end1d = [xyedge[0]-dx, yend, xyedge[1]-dx]
   
       ds = yt.load(i)
      
       ds.field_list
       if field=="n(E)":
          t.append(ds.current_time)
          logbool = True
          (value,location)=ds.find_max(field)
          nemax.append(value)
          weight = ("cell_volume")  # The weight for the average
          ad = ds.all_data()  # This is a region describing the entire box,
          average_value = ad.quantities.weighted_average_quantity(field, weight)
          average_q = ad.quantities.weighted_average_quantity("qden", weight)
          ne_avg.append(average_value*(1.25**3))
          q_tot.append(average_q*1.60217663e-19*(1.25**3))
          cmin = 1.0e14
          cmax = 1.0e17
       elif field=="Efieldmag":
          (value,location)=ds.find_max(field)
          Emaxloc.append(location[1])
          print(location[1])
          Emax.append(value)
          logbool = False
          cmin = 700
          cmax = 1500
   
       ## 1-D slice axial
       ray = ds.ortho_ray(1, (xyedge[0]-dx, xyedge[1]-dx))
       srt = np.argsort(ray["index", "y"])
       if logbool:
          fig1 = plt.figure(1)
          plt.semilogy(np.array(ray["index", "y"][srt]), 1e6*1e-12*np.array(ray[field][srt]))
          plt.ylabel("Ne/10$^{12}$ [m$^{-3}$]")
       else:
          fig2 = plt.figure(2)
          plt.plot(np.array(ray["index", "y"][srt]), 1e-6*1e-7*np.array(ray[field][srt]))
          plt.ylabel("E [MV/cm]")
       plt.xlim([ystart,yend])
       plt.xlabel('y [cm]')
       plotname = i+"_Axis"+field+ext
fig1.savefig("neaxis.png")
fig2.savefig("Efieldaxis.png")
plt.clf()

# L(t) plot
t = np.array(t)
Emaxloc = np.array(Emaxloc)
Emax = np.array(Emax)
nemax = np.array(nemax)
ne_avg = np.array(ne_avg)
q_tot = np.array(q_tot)


# Write to file
fname = 'data.txt'
hname = '# t [s]  Emaxloc [cm]  Emax [kV/cm]  n_e tot [-]  Q total [C]  '
zipped = np.column_stack((t,Emaxloc,Emax*1e-10,ne_avg*4,q_tot*4))
#print(np.size(zipped))
np.savetxt(fname, zipped,header=hname)

# Plots
#
#Lt = 1.25-Emaxloc
#plt.clf()
#plt.plot(t*1e9,Lt)
##plt.xlim([ystart,yend])
#plt.title("Streamer length")
#plt.ylabel("L(t) [cm]")
#plt.xlabel('t [ns]')
#plotname = "Lt.png"
#plt.savefig(plotname)
#plt.clf()
#
## L(t)-vt plot
#v = 0.05*1e9 # cm/ ns *1 ns/1s
#Lt = 1.25-Emaxloc
#plt.clf()
#plt.plot(t*1e9,Lt-v*t)
#plt.title("Streamer length")
##plt.xlim([ystart,yend])
#plt.ylabel("L(t)-vt [cm]")
#plt.xlabel('t [ns]')
#plotname = "Ltmvt.png"
#plt.savefig(plotname)
#plt.clf()
#
#
## Emax
#plt.clf()
#plt.plot(Lt,Emax*1e-7*1e-3)
#plt.title("Max E-field")
##plt.xlim([ystart,yend])
#plt.ylabel("E [kV/cm]")
#plt.xlabel('Lt [cm]')
#plotname = "Emax.png"
#plt.savefig(plotname)
#plt.clf()
#
## nemax
#plt.clf()
#plt.plot(Lt,ne_avg*4/1.0e12)
#plt.title("Total Electrons")
##plt.xlim([ystart,yend])
#plt.ylabel("Ne/10$^{12}$")
#plt.xlabel('Lt [cm]')
#plotname = "nemax.png"
#plt.savefig(plotname)
#
## qtot
#plt.clf()
#plt.plot(Lt,q_tot*4*1e9)
#plt.title("Total Charge")
##plt.xlim([ystart,yend])
#plt.ylabel("Q [nC]")
#plt.xlabel('Lt [cm]')
#plotname = "Qcharge.png"
#plt.savefig(plotname)

