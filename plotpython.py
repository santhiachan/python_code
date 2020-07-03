from netCDF4 import Dataset
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import itertools
import os.path
from os import path
import math
import xarray as xr
import glob

#plot from droplet netcdf file
#setup plot layout
plt.clf()
fig1,ax1=plt.subplots(2,4,figsize=(18, 10))
ncolor=7#len(N_seed)*len(r_seed)
#color_new = mpl.cm.get_cmap('jet', ncolor)
color_new=plt.cm.gist_ncar(np.linspace(0,1,ncolor))#gist_rainbow#brg
n=0
#import droplet data
seedcase=['5micron','gravity','no_solute','5micron_seed','double_seeding','GCCN']#,'gravity_seed','no_solute_seed']
figtag=['Run A','Run B','Run C','Run D1','Run D2','Run D3','Seed-NoTurb','Seed-NoSolu']
ndrop=[6094,6094,6094,6796,7498,6796]

#selectcase=seedcase[1:2]
#selectndrop=ndrop[1:2]

selectcase=['gravity_doublecheckrun']
#['noventi_grav_fall_nossfluc','noventi_noturb_noinert_nossfluc','noventi_noturb_noinert']#['debug_5micron']#['noseed','r10n1','r1n10']['noventi']
selectndrop=[6094]
#[7720,7720,7720]#[6094]#[7020,7088,7720][7720]
ncolor=7#len(N_seed)*len(r_seed)
#color_new = mpl.cm.get_cmap('jet', ncolor)
color_new=plt.cm.gist_ncar(np.linspace(0,1,ncolor))#gist_rainbow#brg
#color_new=plt.cm.jet(np.linspace(0,1,ncolor))#nipy_spectral
#dir='/glade/u/home/sisichen/work_dir/projects_results/seeding_cases/condensation_only/'
dir='/glade/u/home/sisichen/work_dir/projects_results/seeding_cases/conden+coll/'
#dir='/glade/u/home/sisichen/work_dir/projects_results/IUGG_UAE/DNS_model/condensation_only/'
#ncfile
rhow=1000.0
vol=16.5e-2**3
rhoa=1.112683792645852
air_mass=rhoa*vol
P0=1.0e5
RaCp=287.0/1004.0
for (iseedcase,indrop) in zip(selectcase,selectndrop):
    #ncdir=dir+str(iseedcase)+'/nc_files/'
    ncdir=dir+str(iseedcase)+'/'
    print(ncdir)
    ncfilelist=glob.glob(ncdir+'drop*')
    npoint=0
    figlabel=str(iseedcase)
    Timeseries=np.empty(shape=0)
    rmean_series=np.empty(shape=0)
    disp_series=np.empty(shape=0)
    lwc_series=np.empty(shape=0)
    mass20_series=np.empty(shape=0)
    rmax_series=np.empty(shape=0)
    r_follow_series=np.empty(shape=0)
    rccn_follow_series=np.empty(shape=0)
    for ncfile in ncfilelist:
        fh = Dataset(ncfile,mode='r')
        #radius
        radius=fh.variables['R'][:].data
        radius_ccn=fh.variables['R_CCN'][:].data
        theta=fh.variables['thetapp'][:].data
        pp=fh.variables['PP'][:].data
        sp=fh.variables['sp'][:].data
        exner=(pp/P0)**RaCp
        temp=theta*exner
        radius=radius[0,:]
        radius_ccn=radius_ccn[0,:]
        #droplet id
        idp=fh.variables['IDP'][:].data
        idp=idp[0,:]
        #time
        times=fh.variables['TIMES'][:].data
        times=times[0]
        #number of maximum allowable droplets in each processor
        ndropmaxindex=np.linspace(0,indrop*64*3,65, dtype = int)
        ndropmaxindex=ndropmaxindex[1:]-1
        #real number of droplets in each processor
        ndropreal=np.array(idp[ndropmaxindex],dtype=int)
        #index of droplets
        idp_index=np.linspace(0,ndropreal[0]-1,ndropreal[0],dtype=int)
        for i in range(63): #i=1,63
            j=i+1
            idp_index=np.append(idp_index,ndropmaxindex[i]+1+np.linspace(0,ndropreal[j]-1,ndropreal[j],dtype=int))
        idp_index=np.ndarray.tolist(idp_index)
        r_dropreal=radius[idp_index]
        rccn_dropreal=radius_ccn[idp_index]
        idp_dropreal=idp[idp_index]
        #print('sp-seq_dropreal=',min(sp-seq_dropreal))
        #print('seq=',max(seq_dropreal),'sp=',sp)
        
        Timeseries=np.append(Timeseries,times)
        #statistics
        rmean=r_dropreal.mean()
        rmax=r_dropreal.max()
        if n==0:
            idloc_rmax=np.where(r_dropreal == rmax)#find the location of r_max in r_dropreal
            id_follow=int(idp_dropreal[idloc_rmax])#find corresponding idp in idp_dropreal
        id_follow=np.where(idp_dropreal == id_follow)#find location of the global idp in idp_dropreal
        r_follow=r_dropreal[id_follow]
        r_follow_series=np.append(r_follow_series,r_follow)
        rccn_follow=rccn_dropreal[id_follow]
        rccn_follow_series=np.append(rccn_follow_series,rccn_follow)
        #r2mean=r_dropreal**2
        #r2mean=r2mean.mean()
        #r3mean=r_dropreal**3
        #r3mean=r3mean.mean()
        curv_follow=2.0*7.61e-2/(476*1000.0*temp*r_follow)
        solu_follow=2.0*18.0/132.14*1726.0*rccn_follow**3/1000.0
        seq_follow=math.exp(curv_follow-solu_follow/(r_follow**3-rccn_follow**3))-1.0
        r_var=np.var(r_dropreal)
        disp=math.sqrt(r_var)/rmean
        r19=r_dropreal[r_dropreal>19e-6];
        r20=r_dropreal[r_dropreal>20e-6];
        r21=r_dropreal[r_dropreal>21e-6];
        mass=sum(r_dropreal**3)*4/3*math.pi*rhow
        lwc=mass/air_mass
        mass20=sum(r20**3)*4/3*math.pi*rhow
        mass20=mass20/air_mass
        rmax_series=np.append(rmax_series,rmax)
        mass20_series=np.append(mass20_series,mass20)
        lwc_series=np.append(lwc_series,lwc)
        rmean_series=np.append(rmean_series,rmean)
        disp_series=np.append(disp_series,disp)
        time_ascen_index=sorted(range(len(Timeseries)), key=lambda k: Timeseries[k]) #sort out the timeseries in ascending order
    ax1[0,0].plot(Timeseries[time_ascen_index]/60, lwc_series[time_ascen_index]*1e3,label=figlabel,color=color_new[n])
    ax1[0,1].plot(Timeseries[time_ascen_index]/60, rmax_series[time_ascen_index]*1e6, color=color_new[n])
    ax1[0,2].plot(Timeseries[time_ascen_index]/60, rmean_series[time_ascen_index]*1e6, color=color_new[n])
    ax1[0,3].plot(Timeseries[time_ascen_index]/60, disp_series[time_ascen_index], color=color_new[n])#mean radius not volume mean radius
    #ax1[1,1].plot(time1[:min(len(mean_R1),len(time1))]/60, mean_R1, color=color_new[n])#mean r based on LWC
    #ax1[1,1].plot(Timeseries[time_ascen_index]/60, r_follow_series[time_ascen_index], color=color_new[n])
    #ax1[1,0].plot(Timeseries[time_ascen_index]/60,auto_mass19*1e3,color=color_new[n])
    ax1[1,1].plot(Timeseries[time_ascen_index]/60,mass20_series[time_ascen_index]*1e3,color=color_new[n])
    #ax1[1,2].plot(Timeseries[time_ascen_index]/60,auto_mass21*1e3,color=color_new[n])
    #ax1[1,3].plot(Timeseries[time_ascen_index]/60,auto_mass22*1e3,color=color_new[n])
    n=n+1
ax1[0,0].legend(loc='lower right',bbox_to_anchor=(5, -1))
ax1[0,0].set_title('LWC(g/kg)')
#ax1[0,0].set_xlim(0,6)
ax1[0,0].set_xlabel('Time(min)')
