import matplotlib.pyplot as plt
import numpy as np
import yt
import glob
import os

def _Ntot(field, data):
    return (
        (np.array(data['boxlib', 'pressure'])*0.1/(1.380649e-23*np.array(data['boxlib', 'Temp'])))
    )

yt.add_field(
    name=("boxlib", "Ntot"),
    function=_Ntot,
    sampling_type="local",
    units="",
)


def _Epow(field, data):
    return (
        (1.6021766e-19*5.0e24*data['boxlib', 'n(E)']*1.0e6*(data['boxlib', 'Efieldy']*1.0e-5)**2/data['boxlib', 'Ntot'])
    )

yt.add_field(
    name=("boxlib", "Epow"),
    function=_Epow,
    sampling_type="local",
    units="",
)



# Input solution pattern
pltlist = glob.glob('plt?????')
pltlist = sorted(pltlist)

ts = yt.load(
    "plt?????",
)

# Add fields to get data
fields = ["Temp","n(N2(v1))","n(N2(v2))","n(N2(v3))","n(N2(v4))","n(N2(v5))","n(O)","n(N)","n(O(1D))",
         "n(N2(A3Sigma))","n(O2)","n(N2)","n(E)","n(N2(B3Pi))","n(N2(C3Pi))","n(O2(a1Delta))","n(O2(b1Sigma))", 
         "n(NO)","n(N2+)","n(O2+)","Ntot","Epow"]

# Calculate dimensions and create dummy variables
fieldno = len(fields)
tno = len(ts)
npfields = np.empty([tno,fieldno])
t = np.empty([tno,1])
fval = np.empty([1,1])

# Loop over solution to get data
ik = 0
for ds in ts:
    print(ds)
    # Get the value of the field at particular point
    pt = ds.point((0.124,0.124,0.124))
    
    # Get data
    fieldvals = np.empty([1,fieldno])
    ij = 0;
    for i in fields:
       fval = pt[i]
       fieldvals[0,ij] = fval
       #print(fval)
       ij = ij+1
  
    # Append to desired vectors 
    npfields[ik,:] = fieldvals
    t[ik] = ds.current_time
    ik = ik+1

# Write to file
fname = 'data.txt'
hname = '(1) t [s]  '
ik = 2
for i in fields:
       tstr = '('+str(ik)+')'
       hname = hname+'  '+ tstr+' '+ i + '  '
       ik = ik+1
zipped = np.column_stack((t,npfields))
np.savetxt(fname, zipped,header=hname)

