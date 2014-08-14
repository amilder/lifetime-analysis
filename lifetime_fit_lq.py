#!/usr/bin/python
from matplotlib.pyplot import *
from numpy import *
import numpy as np
from scipy import *
from scipy.optimize import leastsq
import glob
import os

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

files={}
data={}
data1={}
d={}
d_norm={}
data_norm={}
start=0.
tend=3120
tshift=0.

#os.chdir("/home/amilder/2014-08-08")
basePath = '/home/amilder/2014-08-08/'
files = glob.glob(basePath+'*.txt')
files.sort()
print files
index=0

def readdata (files, data, tend):
    for n in range (0,15):
        print n
        N=genfromtxt(files[n], names='lag, counts')
        d[n]=N['counts'][1:tend]
        data[n+1] = d[n]
    return data,d


# data[0] is IRF data, data[i] (i>0) are the data taken from fluorescent samples.

fileirf='/home/amilder/2014-08-08_IRF3.pt3.txt'
I=genfromtxt(fileirf,names='lag,counts')
data[0]=I['counts']
new_data=data[0][:tend]
data_subt=new_data-median(new_data)
totalirf=data_subt/sum(data_subt)
irf=totalirf[1:]

data,d= readdata(files, data, tend)
d=d.values()
d=asarray(d)
xdata=arange(tend)*4/1000.
xl=xdata[1:]

fig=figure()
plot(xl,irf)
#fig.show()

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#fit model
p0=zeros(5)
p0[0]=3.
p0[1]=0.0
#p0[2]=0.5
p0[2]=3.0
p0[3]=1.0
#p0[5]=0.1
p0[4]=0.
#p0[5]=15.

def pre_model(x,p):
	term0= (p[0]*1e3) * exp(-((x-tshift)/p[2]))
	term1= (p[1]*1e3) * exp(-((x-tshift)/p[3]))
	#term2= (p[2]*1e4) * exp(-((x-tshift)/p[5]))
	#term3= abs(par[3]) * exp(-(x-par[6])/(par[7]))
	fitfunc= term0+term1
	return fitfunc
	
	
def errfunc(p,x,y):
	model=real(ifft(fft(pre_model(x,p[:4])) * fft(irf[:]))) + p[4]
	residual=(y[:]-model[:]) / sqrt(y[:])
	#residual=asarray(residual).reshape(-1)
	return residual

def fitdata(files,data,tend,n,p0,xl):
    yl=data[n]
    pfit,cov,infodict,mesg,ier=leastsq(errfunc,p0,args=(xl[:],yl[:]),full_output=1)
    chisq=(infodict['fvec']**2).sum()
    dof=len(xl[start:])-len(pfit) #
    
    pre_mod_final=pre_model(xl[:],pfit[:4])
    final_model=real(ifft(fft(pre_mod_final) * fft(irf[:]))) + pfit[4]
    
    pname=['A0','A1','tau0','tau1','offset']
    print "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"
    print "converged with chi squared", chisq
    print "degrees of freedom, dof", dof
    print "Reduced chisq (variance of rsiduals)", chisq/dof
    print
    print "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"
    #for i,pmin in enumerate(pfit):
    #	print "%2i % -10s %12f+/- %10f "%(i,pname[i],pmin,sqrt(cov[i,i])*sqrt(chisq/dof))
    	#,sqrt(cov[i,i])*sqrt(chisq/dof))+/- %10f
    for i,pmin in enumerate(pfit):
    	print  i,pname[i],pmin
    
    fig=figure()
    suptitle(files[n-1])
    ax1=subplot2grid((4,1),(0,0),rowspan=3)
    #ax1.plot(xl[:],irf[:])
    ax1.plot(xl[:],yl[:],color='k',linewidth=3)#
    ax1.plot(xl[:],final_model[:],color='r',linewidth=2)#
    #ax1.set_xlim(2.5,10)
    ax1.set_ylabel('intensity (arb. units)',fontsize=18)
    ax1.yaxis.set_tick_params(labelsize=15)
    ax1.xaxis.set_tick_params(labelsize=14)
    
    ax2=subplot2grid((4,1),(3,0),rowspan=1)
    ax2.plot(xl[:],(final_model[:]-yl[:]),'.',color='k')#
    ax2.axhline(y=0)
    ax2.set_ylim(-300,300)
    #ax2.set_yticks([-200,-100,0,100,200])
    ax2.yaxis.set_tick_params(labelsize=15)
    ax2.xaxis.set_tick_params(labelsize=15)
    #ax2.set_xlim(2.5,10)
    ax2.set_xlabel('time (ns)',fontsize=18)
    ax2.set_ylabel('residuals',fontsize=18)
    
    #fig.show()
    return pfit

def pre_modelglobal(x,g):
        term={}
        for n in range(0,15):
            term[n]=(g[n]*1e3) * exp(-((x-tshift)/g[30]))+(g[n+15]*1e3) * exp(-((x-tshift)/g[31]))
        #term0= (g[0]*1e3) * exp(-((x-tshift)/g[2]))
	#term1= (g[1]*1e3) * exp(-((x-tshift)/g[3]))
	#term2= (p[2]*1e4) * exp(-((x-tshift)/p[5]))
	#term3= abs(par[3]) * exp(-(x-par[6])/(par[7]))
	#fitfunc= term0+term1
        term=term.values()
	return term
	
	
def errfunc2(g,x,y):
	model=real(np.fft.ifftn(np.fft.fftn(pre_modelglobal(x,g)) * np.fft.fftn(irf[:]))) + g[32]
        residual=(y-model) / sqrt(y)
	#residual=asarray(residual).reshape(-1)
        a=residual.shape
        print g
        return residual.flatten()

def globalfit(files,data,tend,g0,xl):
    yl=data
    pfit,cov,infodict,mesg,ier=leastsq(errfunc2,g0,args=(xl[:],yl),full_output=1)
    chisq=(infodict['fvec']**2).sum()
    dof=len(xl[start:])-len(pfit) #
    
    pre_mod_final=pre_model(xl[:],pfit[:4])
    final_model=real(ifft(fft(pre_mod_final) * fft(irf[:]))) + pfit[4]
    
    pname=['A0','A1','tau0','tau1','offset']
    print "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"
    print "converged with chi squared", chisq
    print "degrees of freedom, dof", dof
    print "Reduced chisq (variance of rsiduals)", chisq/dof
    print
    print "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"
    #for i,pmin in enumerate(pfit):
    #	print "%2i % -10s %12f+/- %10f "%(i,pname[i],pmin,sqrt(cov[i,i])*sqrt(chisq/dof))
    	#,sqrt(cov[i,i])*sqrt(chisq/dof))+/- %10f
    for i,pmin in enumerate(pfit):
    	print  i,pname[i],pmin
    
    fig=figure()
    suptitle("global")
    ax1=subplot2grid((4,1),(0,0),rowspan=3)
    #ax1.plot(xl[:],irf[:])
    for n in range(0,15):
        ax1.plot(xl[:],yl[n],color='k',linewidth=3)#
    ax1.plot(xl[:],final_model[:],color='r',linewidth=2)#
    #ax1.set_xlim(2.5,10)
    ax1.set_ylabel('intensity (arb. units)',fontsize=18)
    ax1.yaxis.set_tick_params(labelsize=15)
    ax1.xaxis.set_tick_params(labelsize=14)
    
    ax2=subplot2grid((4,1),(3,0),rowspan=1)
    for n in range(0,15):
        ax2.plot(xl[:],(final_model[:]-yl[n]),'.',color='k')#
    ax2.axhline(y=0)
    ax2.set_ylim(-300,300)
    #ax2.set_yticks([-200,-100,0,100,200])
    ax2.yaxis.set_tick_params(labelsize=15)
    ax2.xaxis.set_tick_params(labelsize=15)
    #ax2.set_xlim(2.5,10)
    ax2.set_xlabel('time (ns)',fontsize=18)
    ax2.set_ylabel('residuals',fontsize=18)
    
    fig.show()
    return pfit

pfit={}
tau1s=zeros(15)
tau2s=zeros(15)
a1=zeros(15)
a2=zeros(15)
for n in range(1,1):
    a = fitdata(files,data,tend,n,p0,xl)
    pfit[n] = a
    tau1s[n-1]= a[2]
    tau2s[n-1]= a[3]
    a1[n-1]= a[0]
    a2[n-1]=a[1]

fig=figure()
xs=([35,35,35,45,45,45,55,55,55,65,65,65,75,75,75])
plot(xs,tau1s, 'ro', label="tau 1")
plot(xs,tau2s, 'bv', label="tau 2")
legend()
#fig.show()

fig=figure()
plot(xs,a1, 'ro', label="A 1")
plot(xs,a2, 'bv', label="A 2")
legend()
#fig.show()

g=ones(33)
g[30]=3.8
g[31]=.2
g[32]=30
gfit=globalfit(files,d,tend,g,xl)
print gfit
