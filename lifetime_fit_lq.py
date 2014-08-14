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
    for n in range(0,15):
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
    """
    x is of shape (nPts,)
    g is of shape (nCurves,)
    returns (nCurves,nPts)
    """
    (tau1, tau2) = g[:2]
    per_curve = g[2:]
    amp1 = per_curve[len(files):2*len(files)]
    amp2 = per_curve[2*len(files):3*len(files)]

    curves  = (amp1[:,newaxis]*1e3) * exp(-((x[newaxis,:] - tshift) / tau1))
    curves += (amp2[:,newaxis]*1e3) * exp(-((x[newaxis,:] - tshift) / tau2))
    return curves
	
def model_global(x,g,irf):
    """
    x is of shape (nPts,)
    y is of shape (nCurves,nPts)
    g is of shape (nCurves,)
    returns (nCurves,nPts)
    """
    per_curve = x[2:]
    offsets = per_curve[0:len(files)]
    curves = pre_modelglobal(x,g)
    models = []
    for (offset, m) in zip(offsets, curves):
	model = real(np.fft.ifft(np.fft.fft(m) * np.fft.fft(irf)))
        model += offset
        models.append(model)
    return np.array(models)
    
def errfunc2(g,x,y,irf):
    """
    x is of shape (nPts,)
    y is of shape (nCurves,nPts)
    g is of shape (nCurves,)
    returns (nCurves*nPts)
    """
    print g
    return ((y - model_global(x,g,irf)) / sqrt(y)).flatten()

def globalfit(yl,tend,g0,xl,irf):
    pfit,cov,infodict,mesg,ier=leastsq(errfunc2,g0,args=(xl[:],yl,irf),full_output=1)
    chisq=(infodict['fvec']**2).sum()
    dof=len(xl[start:])-len(pfit) #
    
    #pname=['A0','A1','tau0','tau1','offset']
    #print "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"
    #print "converged with chi squared", chisq
    #print "degrees of freedom, dof", dof
    #print "Reduced chisq (variance of rsiduals)", chisq/dof
    #print
    #print "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"
    ##for i,pmin in enumerate(pfit):
    ##	print "%2i % -10s %12f+/- %10f "%(i,pname[i],pmin,sqrt(cov[i,i])*sqrt(chisq/dof))
    #	#,sqrt(cov[i,i])*sqrt(chisq/dof))+/- %10f
    #for i,pmin in enumerate(pfit):
    #	print  i,pname[i],pmin
    return pfit

def plot_global(yl, p, xl, irf):
    final_model = model_global(xl[:], p, irf)

    fig=figure()
    suptitle("global")
    ax1=subplot2grid((4,1),(0,0),rowspan=3)
    #ax1.plot(xl[:],irf[:])
    for n in range(0,15):
        ax1.plot(xl[:],yl[n],color='k',linewidth=3)#
        ax1.plot(xl[:],final_model[n],color='r',linewidth=2)#
    #ax1.set_xlim(2.5,10)
    ax1.set_ylabel('intensity (arb. units)',fontsize=18)
    ax1.yaxis.set_tick_params(labelsize=15)
    ax1.xaxis.set_tick_params(labelsize=14)
    
    ax2=subplot2grid((4,1),(3,0),rowspan=1)
    for n in range(0,15):
        ax2.plot(xl[:],(final_model[n]-yl[n]),'.',color='k')#
    ax2.axhline(y=0)
    ax2.set_ylim(-300,300)
    #ax2.set_yticks([-200,-100,0,100,200])
    ax2.yaxis.set_tick_params(labelsize=15)
    ax2.xaxis.set_tick_params(labelsize=15)
    #ax2.set_xlim(2.5,10)
    ax2.set_xlabel('time (ns)',fontsize=18)
    ax2.set_ylabel('residuals',fontsize=18)
    
    fig.show()

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

# g is offsets+amp1+amp2+[tau1,tau2]
g = ones(3*len(files) + 2)
for i,file in enumerate(d):
    g[i] = d.min()
g[0]=3.8  # tau1
g[1]=0.2  # tau2
plot_global(d, g, xl, irf)
dec = 4 # decimation factor
gfit=globalfit(d[:,::dec],tend,g,xl[::dec],irf[::dec])
plot_global(d, gfit, xl, irf)
gfit=globalfit(d,tend,gfit,xl,irf)
plot_global(d, gfit, xl, irf)
print gfit
