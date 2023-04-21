import numpy as np
from matplotlib import pyplot as plt

myfile = "data.txt"
fileCWI = "/scratch1/07638/aduarteg/PeleScratch/streamer_test/Case1/CWI/CWI_AMR.txt"
fileFR = "/scratch1/07638/aduarteg/PeleScratch/streamer_test/Case1/FR/1e13_FR4_tEzmaxNeQrmax.res"
fileCN = "/scratch1/07638/aduarteg/PeleScratch/streamer_test/Case1/CN/CN.txt"

# Indices for each
files = [myfile,fileCWI,fileFR,fileCN]
skipidx = [1 ,1, 0,2]
tidx = [0 ,0, 0,0]
Elocidx = [1, 2, 2,4]
Emaxidx = [2, 1, 1,3]
neidx = [3, 3, 3 ,2]
qidx = [4, 4, 4,1]

labelvec = ["PeleC","CWI","FR","CN"]
lsc = ["-o","-x","-v","-"]
tfac = [1.0,1.0,1.0,1.0e-9]
Efac = [1.0,1.0e-5,1e-5,1.0]
zfac = [1.0,1.0e2,1.0,0.1]
#zoff = [0.0,0.0,-1.25,0.0]
nfac = []
qfac = []

for i in range(0,4):
    print(i)
    lstr = lsc[i]
    labelstr = labelvec[i]
    # Load , organize, convert data
    fileid = files[i]
    data = np.loadtxt(fileid,skiprows=skipidx[i])

    t = data[:,tidx[i]]*tfac[i]
    Emaxloc = data[:,Elocidx[i]]*zfac[i]
    Emax = data[:,Emaxidx[i]]*Efac[i]
    ne_tot = data[:,neidx[i]]
    q_tot = data[:,qidx[i]]
    Lt = 1.25-Emaxloc
    if labelstr =="FR":
        Lt = Emaxloc

    # Plots
    fig1 = plt.figure(1)
    plt.plot(t*1e9,Lt,lstr,label=labelstr)
    plt.legend()
    plt.title("Streamer length")
    plt.ylabel("L(t) [cm]")
    plt.xlabel('t [ns]')
    plotname1 = "Lt.png"
    
    # L(t)-vt plot
    fig2 = plt.figure(2)
    v = 0.05*1e9 # cm/ ns *1 ns/1s
    plt.plot(t*1e9,Lt-v*t,lstr,label=labelstr)
    plt.legend()
    plt.title("Streamer length")
    plt.ylabel("L(t)-vt [cm]")
    plt.xlabel('t [ns]')
    plotname2 = "Ltmvt.png"
    
    
    # Emax
    fig3 = plt.figure(3)
    plt.plot(Lt,Emax,lstr,label=labelstr)
    plt.title("Max E-field")
    plt.legend()
    plt.ylabel("E [kV/cm]")
    plt.xlabel('Lt [cm]')
    plotname3 = "Emax.png"
    
    # nemax
    fig4 = plt.figure(4)
    plt.plot(Lt,ne_tot/1.0e12,lstr,label=labelstr)
    plt.title("Total Electrons")
    plt.legend()
    plt.ylabel("Ne/10$^{12}$")
    plt.xlabel('Lt [cm]')
    plotname4 = "nemax.png"
    
    # qtot
    fig5 = plt.figure(5)
    plt.plot(Lt,q_tot*1e9,lstr,label=labelstr)
    plt.title("Total Charge")
    plt.legend()
    plt.ylabel("Q [nC]")
    plt.xlabel('Lt [cm]')
    plotname5 = "Qcharge.png"
    
    
fig1.savefig(plotname1)
fig2.savefig(plotname2)
fig3.savefig(plotname3)
fig4.savefig(plotname4)
fig5.savefig(plotname5)
