import yt
import matplotlib
import numpy as np
from matplotlib import pyplot as plt

def plotelectrode(pltfile,field,logbool):
    print("Plotting elec...")
    

#
#
#    ext = ".png"
#    cutplane = "z"
#    refL = 0.5  # [cm]
#    plotgrids = True
#    minGridLevel = 6
#    clocation = [0.9 ,0.625 ,0.99]
#    xyedge = [1.0 , 1.0]
#
#    ds = yt.load(pltfile)
#    ds.field_list
#    #(value,location)=ds.find_max(field)
#    #print(location)
#  
#    # cathode zoom in with cells
#    #sl = yt.SlicePlot(ds,cutplane,field,center=[0.99,0.375,0.99],width=[0.02,0.02])
#    #sl.set_log((field), logbool)
#    #sl.annotate_cell_edges()
#    #sl.annotate_contour("vfrac", clim=(0.5, 0.99), ncont=5, label=False,take_log=False)
#    #sl.annotate_timestamp(corner="upper_left", draw_inset_box=True)
#    #slname = pltfile+"_Slice"+field+"_anode_"+cutplane+"_wcells"+ext
#    #sl.save("Cell_visual.png")
#    
#    
#    ## 2-D slice anode
#    ds = yt.load(pltfile)
#    ds.field_list
#    sl = yt.SlicePlot(ds,cutplane,field,center=clocation,width=[0.1,0.1])
#    #sl = yt.SlicePlot(ds,cutplane,field,center=("max", field),width=[0.1,0.1])
#    sl.set_log((field), logbool)
#    #sl.annotate_grids(min_level=minGridLevel)
#    sl.annotate_timestamp(corner="upper_left", draw_inset_box=True)
#    slname = pltfile+"_Slice"+field+"_anode_"+cutplane+ext
#    sl.save(slname)
#    
#    
#    ## 2-D slice cathode
#    ds = yt.load(pltfile)
#    ds.field_list
#    sl = yt.SlicePlot(ds,cutplane,field,center=[0.9,0.375,0.99],width=[0.2,0.2])
#    sl.set_log((field), logbool)
#    sl.annotate_timestamp(corner="upper_left", draw_inset_box=True)
#    slname = pltfile+"_Slice"+field+"_cathode_"+cutplane+ext
#    sl.save(slname)
#
#    # full gap
#    clocation = [xyedge[0]-0.1 ,0.5 ,xyedge[1]-0.01]
#    clocy = [xyedge[0]-0.1 ,0.2 ,xyedge[1]-0.1]
#
#    ds = yt.load(pltfile)
#    ds.field_list
#  
#    
#    ## 2-D slice Gap
#    #ds = yt.load(pltfile)
#    ds.field_list
#    sl = yt.SlicePlot(ds,cutplane,field,center=clocation,width=[0.2,1.0],fontsize=30)
#    #sl = yt.SlicePlot(ds,cutplane,field,center=("max", field),width=[0.1,0.1])
#    sl.set_log((field), logbool)
#    #sl.annotate_grids(min_level=minGridLevel)
#    ##sl.set_zlim(field, 590, 600)
#    sl.annotate_timestamp(corner="upper_left", draw_inset_box=True)
#    slname = pltfile+"_Slicez"+field+"_Gap_"+cutplane+ext
#    sl.save(slname)
#   
#    ## 2-D slice Gap
#    cutplane = "y" 
#    ds = yt.load(pltfile)
#    ds.field_list
#    #sl = yt.SlicePlot(ds,cutplane,field,center=clocy,width=[0.2,0.2],fontsize=30)
#    sl = yt.SlicePlot(ds,cutplane,field,center=[0.9, 0.1, 0.99],width=[0.2,0.2])
#    sl.annotate_timestamp(corner="upper_left", draw_inset_box=True)
#    slname = pltfile+"_Slicey"+field+"_Gap_"+cutplane+ext
#    sl.save(slname)
    
def plotmax(pltfile,field,logbool):
    print("Plotting centered around max")

    ext = ".png"
    cutplane = "z"
    refL = 0.5  # [cm]

    ds = yt.load(pltfile)
    print(ds.field_list)
    _, clocation= ds.find_max(field)

    print("The max is located at:")
    print(clocation)

    cstart = [clocation[0] ,0.125 ,clocation[2]]
    cend = [clocation[0] ,0.375 ,clocation[2]]

    ## 2-D slice
    ds = yt.load(pltfile)
    ds.field_list
    sl = yt.SlicePlot(ds,cutplane,field,center=clocation,width=[0.1,0.1])
    sl.set_log((field), logbool)
    #sl.annotate_grids(min_level=minGridLevel)
    sl.annotate_timestamp(corner="upper_left", draw_inset_box=True)
    slname = pltfile+"_Slice_"+field+"_max_"+cutplane+ext
    sl.save(slname)


    #### 1-D slice
    plot = yt.LinePlot(ds,field,cstart,cend, 4000,figure_size=[15,10])
    plot.annotate_legend(field)
    plot.set_x_unit("cm")
    sl.annotate_timestamp(corner="upper_left", draw_inset_box=True)
    plotname = pltfile+"_Axis_"+field+"_max_"+cutplane+ext
    plot.save(plotname)


def plotGap(pltfile,field,logbool,xyedge,cmapvar,zlims,gridson):
    print("Plotting...")

    ext = ".png"
    cutplane = "z"
    #cutplane = "x"
    refL = 0.5  # [cm]
    plotgrids = True
    minGridLevel = 2
    clocation = [1.25-0.1 ,0.625 ,1.25-0.0000001]
    #clocation = [xyedge[0]-0.0001 ,0.25 ,xyedge[1]-0.125]
    clocy = [xyedge[0]-0.1 ,0.2 ,xyedge[1]-0.1]
    start1d = [xyedge[0]-0.01, 0.125, xyedge[1]-0.01]
    end1d = [xyedge[0]-0.01, 0.375, xyedge[1]-0.01]

    ds = yt.load(pltfile)
    ds.force_periodicity()
    ad = ds.region(ds.domain_center, ds.domain_left_edge, ds.domain_right_edge,fields=field)
    #ad = ds.region(ds.domain_center, ds.domain_left_edge, ds.domain_right_edge,fields=field)
    #ad = ds.smoothed_covering_grid(3, ds.domain_left_edge, ds.domain_dimensions * 2**3,fields=field)
    #left_corner = [0.0 ,0.0 ,xyedge[1]-1e-1]
    #right_corner = [xyedge[0]-1e-6 ,0.4999 ,xyedge[1]-1e-6]
    #ad = ds.box(left_corner, right_corner)

    no_pin = ad.cut_region(['obj["vfrac"] > 0.99999'])
    ds.field_list
    if field=="n(E)":
       cmin = 1.0e14
       cmax = 1.0e17
    elif field=="Temp":
       cmin = 700
       cmax = 1500

    annotateL = False
    annstr = ""
    ## 2-D slice Gap
    #ds = yt.load(pltfile)
    #(maxval,maxloc)=ds.find_max(field)
    maxloc = [0.0,0.0,0.0]
    
    ds.field_list
    sl = yt.SlicePlot(ds,cutplane,field,center=clocation,width=[0.2,1.0],fontsize=20,origin='center-right-window',data_source=no_pin,buff_size=[3000,3000])
    #sl = yt.ProjectionPlot(ds,cutplane,field,i,weigth_field=field,center=clocation,width=[0.3,0.5],fontsize=20,origin='center-right-window',data_source=no_pin,buff_size=[3000,3000])
    #sl = yt.ProjectionPlot(ds,cutplane,field,weight_field=field,center=clocation,width=[0.3,0.3],fontsize=20,origin='center-right-window',data_source=no_pin,buff_size=[3000,3000])
    #sl = yt.ProjectionPlot(ds,cutplane,field,weight_field=field,center=clocation,width=[0.3,0.3],fontsize=20,origin='center-right-window',data_source=no_pin,buff_size=[3000,3000])
    sl.set_axes_unit("mm")
    sl.set_xlabel('r (mm)')
    sl.set_ylabel('z (mm)')
    #sl = yt.SlicePlot(ds,cutplane,field,center=clocation,width=[0.25,0.35],fontsize=20,origin='center-right-domain')
    sl.set_log((field), logbool)
    if bool(zlims):
       sl.set_zlim(field, zlims[0], zlims[1])
    sl.set_cmap(field, cmapvar)
    sl.set_background_color(field, color="gray")
    sl.annotate_timestamp(corner="upper_left", draw_inset_box=True)
    #sl.flip_horizontal()
    #sl.set_origin('center-right-window')
    if field=="n(E)" or field=="n(O(1D))" or field=="n(N2(v1))" or field=="n(N2(v2))" or field=="n(N2(A3Sigma))":
       plot = sl.plots[field]      
       colorbar = plot.cb
       sl._setup_plots()
       cbstr = field+" (cm$^{-3}$)"
       colorbar.set_label(cbstr)
    elif field=="Efieldmag":
       plot = sl.plots[field]      
       colorbar = plot.cb
       sl._setup_plots()
       colorbar.set_label('E (erg/C-cm)')
    elif field=="pressure":
       plot = sl.plots[field]      
       colorbar = plot.cb
       sl._setup_plots()
       colorbar.set_label('P (Ba)')
    plot = sl.plots[list(sl.plots)[0]]
    ax = plot.axes
    img = ax.images[0]
    img.set_interpolation("bicubic")
   # if annotateL:
   #    sl.annotate_line((0.999, 0.125,0.999), (0.999, 0.375,0.999),coord_system="data",color="red")
   #    sl.annotate_line((0.97, 0.1, 0.97), (0.97, 0.4, 0.97),coord_system="data",color="white")
   #    sl.annotate_line((0.95, 0.13,0.999), (0.999, 0.13,0.999),coord_system="data",color="black")
   #    sl.annotate_line((0.95, 0.37,0.999), (0.999, 0.37,0.999),coord_system="data",color="yellow")
   #    sl.annotate_line((0.95, 0.25,0.999), (0.999, 0.25,0.999),coord_system="data",color="green")
   #    annstr = "annotated"
   # gridson = False
    if gridson:
       sl.annotate_grids(max_level=5,min_level=3)
       annstr = "grids"
    slname = pltfile+"_Slicez"+field+"_Gap1_"+annstr+cutplane+ext
    sl.save(slname)
    #sl.close()
   

    #sl = yt.SlicePlot(ds,cutplane,field,center=("max",field),width=[0.05,0.05],fontsize=20)
    #sl.set_log((field), logbool)
    #sl.annotate_timestamp(corner="upper_left", draw_inset_box=True)
    #sl.set_zlim(field, cmin, cmax)
    #slname = pltfile+"_Slicez"+field+"_Gapmax_"+cutplane+ext
    #sl.save(slname)

    #clocation = [xyedge[0]-0.1 ,0.375 ,xyedge[1]-0.0001]
    #ds.field_list
    #sl = yt.SlicePlot(ds,cutplane,field,center=clocation,width=[0.2,0.2],fontsize=20)
    #sl.set_log((field), logbool)
    #sl.annotate_timestamp(corner="upper_left", draw_inset_box=True)
    #sl.annotate_cell_edges()
    #slname = pltfile+"_Slicez"+field+"_Gap2_"+cutplane+ext
    #sl.save(slname)
    #sl.clf()

    #clocation = [xyedge[0]-0.1 ,0.125 ,xyedge[1]-0.0001]
    #ds.field_list
    #sl = yt.SlicePlot(ds,cutplane,field,center=clocation,width=[0.2,0.2],fontsize=20)
    #sl.set_log((field), logbool)
    #sl.annotate_cell_edges()
    #sl.annotate_timestamp(corner="upper_left", draw_inset_box=True)
    #slname = pltfile+"_Slicez"+field+"_Gap3_"+cutplane+ext
    #sl.save(slname)
    #sl.clf()

   
    ## 2-D slice Gap
    #cutplane = "y" 
    #ds = yt.load(pltfile)
    #ds.field_list
    #sl = yt.SlicePlot(ds,cutplane,field,center=clocy,width=[0.2,0.2],fontsize=30)
    #sl = yt.SlicePlot(ds,cutplane,field,center=("max", field),width=[0.05,0.05])
    #sl.set_log((field), logbool)
    #sl.annotate_grids(max_level=4,min_level=4)
    #sl.annotate_timestamp(corner="upper_left", draw_inset_box=True)
    #sl.set_zlim(field, cmin, cmax)
    #slname = pltfile+"_Slicey"+field+"_Gap_"+cutplane+ext
    #sl.save(slname)
    #sl.clf()
    gridson=False
    yheight = [0.25]

   ## for i in yheight:
   ##     yhstr = str(i)
   ##     clocy = [xyedge[0]-0.08 ,i ,xyedge[1]-0.08]
   ##     cutplane = "y" 
   ##     ds = yt.load(pltfile)
   ##     ds.field_list
   ##     sl = yt.SlicePlot(ds,cutplane,field,center=clocy,width=[0.16,0.16],fontsize=30,buff_size=[2000,2000])
   ##     #sl = yt.SlicePlot(ds,cutplane,field,center=("max", field),width=[0.05,0.05])
   ##     sl.set_log((field), logbool)
   ##     #sl.annotate_cell_edges()
   ##     #sl.set_zlim(field, 590, 2500)
   ##     sl.set_cmap(field, cmapvar)
   ##     sl.annotate_timestamp(corner="upper_left", draw_inset_box=True)
   ##     #sl.annotate_contour(field,levels=6)
   ##     if gridson:
   ##        sl.annotate_grids(max_level=4,min_level=3)
   ##        annstr = "grids"
   ##     slname = pltfile+"_Slicey"+field+"_Gap_"+yhstr+annstr+cutplane+ext
   ##     sl.save(slname)
        #sl.clf()

def plot1d(pltfile,field,logbool,xyedge):
    print("Plotting...")

    ext = ".png"
    cutplane = "z"
    refL = 0.5  # [cm]
    plotgrids = True
    minGridLevel = 2
    clocation = [xyedge[0]-0.125 ,0.25 ,xyedge[1]]
    clocy = [xyedge[0]-0.1 ,0.2 ,xyedge[1]-0.1]
    start1d = [xyedge[0]-0.01, 0.125, xyedge[1]-0.01]
    end1d = [xyedge[0]-0.01, 0.375, xyedge[1]-0.01]

    ds = yt.load(pltfile)
    ds.field_list
    if field=="n(E)":
       cmin = 1.0e14
       cmax = 1.0e17
    elif field=="Temp":
       cmin = 700
       cmax = 1500
    
    ## 1-D slice axial
    ray = ds.ortho_ray(1, (xyedge[0]-0.01, xyedge[1]-0.01))
    #ray = ds.ortho_ray(1, (maxloc[2], maxloc[0]))
    srt = np.argsort(ray["index", "y"])
    plt.plot(np.array(ray["index", "y"][srt]), np.array(ray[field][srt]))
    #plt.plot(np.array(ray["index", "y"][srt]), np.array(ray["Temp"][srt]))
    #plt.plot(np.array(ray["index", "y"][srt]), np.array(ray["pressure"][srt]))
    #plt.plot(np.array(ray["index", "y"][srt]), np.array(ray["velocity_magnitude"][srt]))
    plt.xlim([0.125,0.375])
    plt.ylabel(field)
    plt.xlabel('y [cm]')
    plotname = pltfile+"_Axis"+field+ext
    plt.savefig(plotname)
    plt.clf()
    
    ## 1-D radialc
    ray = ds.ortho_ray(0, (0.25, xyedge[1]-0.001))
    srt = np.argsort(ray["index", "x"])
    plt.plot(np.array(ray["index", "x"][srt]), np.array(ray[field][srt]))
    plt.xlim([0.5,1.0])
    plt.ylabel(field)
    plt.xlabel('r [cm]')
    plotname = pltfile+"_Radial"+field+ext
    plt.savefig(plotname)
    plt.clf()


def plot1dList(pltlist,field,logbool,xyedge,dir,loc,lims):
    print("Plotting list...")

    for i in pltlist:
       pltfile=i
       ext = ".png"
       cutplane = "z"
       refL = 0.5  # [cm]
       plotgrids = True
       minGridLevel = 2
       clocation = [xyedge[0]-0.125 ,0.25 ,xyedge[1]-0.0001]
       clocy = [xyedge[0]-0.1 ,0.2 ,xyedge[1]-0.1]
       start1d = [xyedge[0]-0.01, 0.125, xyedge[1]-0.01]
       end1d = [xyedge[0]-0.01, 0.375, xyedge[1]-0.01]

       ds = yt.load(i)
       ds.field_list
       if field=="n(E)":
          cmin = 1.0e14
          cmax = 1.0e17
       elif field=="Temp":
          cmin = 700
          cmax = 1500
      
       ti = ds.current_time.in_units("s")
       print(ti) 
       ## 1-D slice axial
       if dir==1:
          ray = ds.ortho_ray(1, (loc[0], loc[1]))
          #ray = ds.ortho_ray(1, (maxloc[2], maxloc[0]))
          #plt.hold(True)
          srt = np.argsort(ray["index", "y"])
          labelstr = str(ti)
          labelstr = labelstr[0:4]+labelstr[-6:]
          plt.plot(np.array(ray["index", "y"][srt]), np.array(ray[field][srt]),label=labelstr)
          #plt.plot(np.array(ray["index", "y"][srt]), np.array(ray["Temp"][srt]))
          #plt.plot(np.array(ray["index", "y"][srt]), np.array(ray["pressure"][srt]))
          #plt.plot(np.array(ray["index", "y"][srt]), np.array(ray["velocity_magnitude"][srt]))
          locstr = "z_"+str(loc[0])+"x_"+str(loc[1])
          plt.xlim(lims)
          plt.ylabel(field)
          plt.xlabel('y [cm]')
          if logbool: plt.yscale("log")
          plotname = "List"+"_Axis"+field+locstr+ext
          #plt.savefig(plotname)
          #plt.clf()
       elif dir==0:
          ## 1-D radialc
          ray = ds.ortho_ray(0, (loc[0], loc[1]))
          srt = np.argsort(ray["index", "x"])
          labelstr = str(ti)
          labelstr = labelstr[0:4]+labelstr[-6:]
          plt.plot(np.array(ray["index", "x"][srt]), np.array(ray[field][srt]),label=labelstr)
          plt.xlim(lims)
          plt.ylabel(field)
          plt.xlabel('r [cm]')
          if logbool: plt.yscale("log")
          locstr = "y_"+str(loc[0])+"z_"+str(loc[1])
          plotname = "List"+"_Radial"+field+locstr+ext
          #plt.savefig(plotname)
    plt.legend()
    plt.savefig(plotname)
    plt.clf()

def plotvfracs(pltfile,field,logbool,xyedge,cmapvar,zlims,gridson):
    print("Plotting...")

    ext = ".png"
    cutplane = "z"
    #cutplane = "x"
    refL = 0.5  # [cm]
    plotgrids = True
    minGridLevel = 2
    clocation = [xyedge[0]-0.125 ,0.25 ,xyedge[1]-0.0001]
    #clocation = [xyedge[0]-0.0001 ,0.25 ,xyedge[1]-0.125]
    clocy = [xyedge[0]-0.1 ,0.2 ,xyedge[1]-0.1]
    start1d = [xyedge[0]-0.01, 0.125, xyedge[1]-0.01]
    end1d = [xyedge[0]-0.01, 0.375, xyedge[1]-0.01]

    ds = yt.load(pltfile)
    ad = ds.region(ds.domain_center, ds.domain_left_edge, ds.domain_right_edge,fields=field)
    v1_frac = ad.cut_region(['obj["n(v1frac)"] > 1e-2'])
    v2_frac = ad.cut_region(['obj["n(v2frac)"] > 1e-2'])
    v3_frac = ad.cut_region(['obj["n(v3frac)"] > 1e-2'])
    v4_frac = ad.cut_region(['obj["n(v4frac)"] > 1e-2'])
    v5_frac = ad.cut_region(['obj["n(v5frac)"] > 1e-2'])
    ds.field_list

    annotateL = False
    annstr = ""
    ## 2-D slice Gap
    #ds = yt.load(pltfile)
    #(maxval,maxloc)=ds.find_max(field)
    maxloc = [0.0,0.0,0.0]
    
    ds.field_list
    sl = yt.SlicePlot(ds,cutplane,"n(v1frac)",center=clocation,width=[0.25,0.35],fontsize=20,origin='center-right-window',data_source=v1_frac)
    #sl = yt.SlicePlot(ds,cutplane,"n(v2frac)",center=clocation,width=[0.25,0.35],fontsize=20,origin='center-right-window',data_source=v2_frac)
    #sl = yt.SlicePlot(ds,cutplane,"n(v3frac)",center=clocation,width=[0.25,0.35],fontsize=20,origin='center-right-window',data_source=v3_frac)
    #sl = yt.SlicePlot(ds,cutplane,"n(v4frac)",center=clocation,width=[0.25,0.35],fontsize=20,origin='center-right-window',data_source=v4_frac)
    #sl = yt.SlicePlot(ds,cutplane,"n(v5frac)",center=clocation,width=[0.25,0.35],fontsize=20,origin='center-right-window',data_source=v5_frac)
    sl.hide_colorbar()
    sl.set_axes_unit("mm")
    sl.set_xlabel('r (mm)')
    sl.set_ylabel('z (mm)')
    #sl = yt.SlicePlot(ds,cutplane,field,center=clocation,width=[0.25,0.35],fontsize=20,origin='center-right-domain')
    fieldlist = ["n(v1frac)","n(v2frac)","n(v3frac)","n(v3frac)","n(v5frac)"]
    #for ix in fieldlist:
       #sl.set_log(ix, logbool)
       #sl.set_zlim(ix, 1e-4, 1e-2)
       #sl.set_cmap(ix, cmapvar)
       #sl.set_background_color(field, color="gray")
    if gridson:
       sl.annotate_grids(max_level=3,min_level=1)
    sl.annotate_timestamp(corner="upper_left", draw_inset_box=True)
    #sl.flip_horizontal()
    #sl.set_origin('center-right-window')
    slname = pltfile+annstr+"_Slicez"+"vfracs"+"_Gap1_"+cutplane+ext
    sl.save(slname)
    #sl.close()
   
