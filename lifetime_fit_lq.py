#!/usr/bin/python
from matplotlib.pyplot import *
from numpy import *
import numpy as np
from scipy import *
from scipy.optimize import leastsq
import glob
import os
from lmfit import Parameters
import argparse
#rewriting as variable using lmfit parameter
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#create arrays for files and data as well as create the ph array and starting and ending points for reading data
files={}
data={}
data1={}
d={}
xd={}
xdata={}
d_norm={}
data_norm={}
start=0.
tend=3120
tshift= 0.0

parser = argparse.ArgumentParser(description='Fit lifetime data')
parser.add_argument('folder', type=str, help='Folder containing data')
parser.add_argument('irffile', type=str, help='IRF file')
parser.add_argument('-g', '--global-lifetimes', default=0, type=int, help='Set the number of global decay lifetimes')
parser.add_argument('-l', '--lifetimes', default=0, type=int, help='Set the number of local decay lifetimes')
parser.add_argument('-t', '--time-shift', action='store_true', default=False, help='Set to fit time shift')
parser.add_argument('-b', '--background', action='store_true', default=False, help='Set to fit background')
args = parser.parse_args()

assert args.global_lifetimes > 0 or args.lifetimes > 0


#phs=input("Please enter a list of phs ")
#phs=([35,35,35,40,40,40,45,45,45,50,50,50,54,54,54,56,56,56,60,60,60,65,65,65,70,70,70,75,75,75,80,80,80])
#phs=([25,35,45,55,55,65,75,75,65,55,55,45,35,25])    #need to find a way to make this automatic

#imports all files from a folder(basepath) and sorts them alphabeticaly then prints the list
#os.chdir("/home/amilder/2014-08-08")
#basePath = '/home/amilder/2014-11-13/2014-11-13/2014-11-13/'
#basePath = input("Please enter folder containing data ")
basePath = args.folder
files = glob.glob(basePath+'*.txt')
files.sort()
#print files
index=0

#2 different options for how many of the files to use
num_of_files_used=len(files)
#num_of_files_used=14

#method readdata reads a file with 2 columns and makes the second column into a dictionary d, data
#in this dictionary the n is the location of the array of counts
def readdata (files, data, tend, num_used):
    for n in range (0,num_used):
        #print n
        N=genfromtxt(files[n], names='lag, counts')
        d[n]=N['counts'][1:tend]
        data[n+1] = d[n]
	xd[n]=N['lag'][1:tend]
	xdata[n+1] = xd
    return data,d,xdata,xd


# data[0] is IRF data, data[i] (i>0) are the data taken from fluorescent samples.
#reads an IRF and adds that to the data dictionary
#fileirf='/home/amilder/2014-11-13/2014-11-13/run001.pt3.txt'
#fileirf=input("Please enter the irf file ")
fileirf= args.irffile
I=genfromtxt(fileirf,names='lag,counts')
data[0]=I['counts']
new_data=data[0][:tend]

#normalize the IRF by subtracting off a baseline and then dividing by the total counts
def normalize_irf(irf):
    centered = irf - median(irf)
    return centered / sum(centered)

#call the methods to normalize the IRF, read the data, create a lag array called xl which
#is the numbers 4/1000 to tend*4/1000 counting by 4/1000
irf=normalize_irf(new_data)[1:]

data,d,xdata,xd= readdata(files, data, tend, num_of_files_used)
d=d.values()
d=asarray(d)
#xld=arange(tend)*8/1000.
#xl=xld[1:]
xl=xd[1]/1000.

#plot a semilogy graph of the IRF
fig=figure()
ax= fig.add_subplot(1,1,1)
ylabel('Normalized Counts')
xlabel('Time in ns')
plot(xl,irf)
ax.set_yscale('log')
fig.show()

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#fit coefficients starting
#p0=zeros(7)
#p0[0]=.5
#p0[1]=.5
#p0[2]=2.0
#p0[3]=4.0
#p0[4]=0.5
#p0[5]=.5
#p0[6]=40
lp0= Parameters()
lp0.add('amp1' , value =0.0 , vary=False)
lp0.add('amp2' , value =0.0 , vary=False)
lp0.add('tau1' , value =2.0 , vary=True)
lp0.add('tau2' , value =4.0 , vary=True)
lp0.add('amp3' , value =0.0 , vary=False)
lp0.add('tau3' , value =.5 , vary=True)
lp0.add('tshift' , value =0.0 , vary=False)
lp0.add('offset' , value =40.0 , vary=False)
gp0= Parameters()
gp0.add('amp1' , value =0.0 , vary=False)
gp0.add('amp2' , value =0.0 , vary=False)
gp0.add('tau1' , value =2.0 , vary=True)
gp0.add('tau2' , value =4.0 , vary=True)
gp0.add('amp3' , value =0.0 , vary=False)
gp0.add('tau3' , value =.5 , vary=True)
gp0.add('tshift' , value =0.0 , vary=False)
gp0.add('offset' , value =40.0 , vary=False)

if args.global_lifetimes>=1:
    gp0['amp1'].vary=True
if args.global_lifetimes>=2:
    gp0['amp2'].vary=True
if args.global_lifetimes>=3:
    gp0['amp3'].vary=True
if args.lifetimes>=1:
    lp0['amp1'].vary=True
if args.lifetimes>=2:
    lp0['amp2'].vary=True
if args.lifetimes>=3:
    lp0['amp3'].vary=True
if args.time_shift:
    lp0['tshift'].vary=True
    gp0['tshift'].vary=True
if args.background:
    lp0['offset'].vary=True
    gp0['offset'].vary=True

#the model of 3 exponentials that we are using, the method evaluates the model for all x's givin
def pre_model(x,p):
    term0= (p[0]*1e3) * exp(-((x-tshift)/p[2]))
    term1= (p[1]*1e3) * exp(-((x-tshift)/p[3]))
    term2= (p[4]*1e3) * exp(-((x-tshift)/p[5]))
    #term3= abs(par[3]) * exp(-(x-par[6])/(par[7]))
    fitfunc= term0+term1+term2
    return fitfunc

#calculates the resuduals weighted 1/sqrt(N) by comparing the experiental data with fourier
#transform convolution of our model and IRF
def errfunc(p,x,y):
    model=real(ifft(fft(pre_model(x,p[:6])) * fft(irf[:]))) + p[6]
    residual=(y[:]-model[:]) / sqrt(y[:])
    #residual=asarray(residual).reshape(-1)
    return residual

#preforms least squares fitting of the model above to the experimental data for each file
#prints out for each file the fit coeficients and chi squared
#plots the experimental data, fit line, and residuals for each file
#returns the fit coefficients
def fitdata(files,data,tend,n,p0,xl):
    yl=data[n]
    pfit,cov,infodict,mesg,ier=leastsq(errfunc,p0,args=(xl[:],yl[:]),
            full_output=1)
    chisq=(infodict['fvec']**2).sum()
    dof=len(xl[start:])-len(pfit) #
    
    pre_mod_final=pre_model(xl[:],pfit[:6])
    final_model=real(ifft(fft(pre_mod_final) * fft(irf[:]))) + pfit[6]
    
    pname=['A0','A1','tau0','tau1','A2','tau2','offset']
    print "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"
    print "converged with chi squared", chisq
    print "degrees of freedom, dof", dof
    print "Reduced chisq (variance of residuals)", chisq/dof
    print
    print "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"
    #for i,pmin in enumerate(pfit):
    #   print "%2i % -10s %12f+/- %10f "%(i,pname[i],pmin,sqrt(cov[i,i])*sqrt(chisq/dof))
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

#global model
def pre_modelglobal(x,g):
    """
    x is of shape (nPts,)
    g is of shape (nCurves,)
    returns (nCurves,nPts)
    """
    (tau1, tau2, tau3, tshift) = g[:4]
    per_curve = g[4:]
    amp1 = per_curve[num_of_files_used:2*num_of_files_used]
    amp2 = per_curve[2*num_of_files_used:3*num_of_files_used]
    amp3 = per_curve[3*num_of_files_used:4*num_of_files_used]

    curves  = (amp1[:,newaxis]*1e3) * exp(-((x[newaxis,:] - tshift) / tau1))
    curves += (amp2[:,newaxis]*1e3) * exp(-((x[newaxis,:] - tshift) / tau2))
    curves += (amp3[:,newaxis]*1e3) * exp(-((x[newaxis,:] - tshift) / tau3))
    return curves
    
def print_global(g):
    per_curve = g[4:]
    print 'tausand tshift = ', g[:4]
    print 'offsets = ', per_curve[0:num_of_files_used]
    print 'amp1 = ', per_curve[num_of_files_used:2*num_of_files_used]
    print 'amp2 = ', per_curve[2*num_of_files_used:3*num_of_files_used]
    print 'amp3 = ', per_curve[3*num_of_files_used:4*num_of_files_used]
    print

def model_global(x,g,irf):
    """
    x is of shape (nPts,)
    y is of shape (nCurves,nPts)
    g is of shape (nCurves,)
    returns (nCurves,nPts)
    """
    per_curve = g[4:]
    #print_global(g)
    offsets = per_curve[0:num_of_files_used]
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
    return ((y - model_global(x,g,irf)) / sqrt(y)).flatten()

def globalfit(yl,g0,xl,irf):
    """
    yl is of shape (nCurves,nPts)
    g0 is of shape (2+3*nCurves,)
    xl is of shape (nPts,)
    irf is of shape (nPts,)
    """
    pfit,cov,infodict,mesg,ier=leastsq(errfunc2,g0,args=(xl,yl,irf),
            full_output=1)
    redchisqs=zeros(num_of_files_used)

    for i,(fname,y,ym) in enumerate(zip(files, yl, model_global(xl,pfit,irf))):
        resid = (y - ym) / sqrt(y)
        chisq = sum(resid**2)
        dof = len(resid) - (2+3)
        print '%s:' % fname
        print '  chi^2          = %10f' % chisq
        print '  dof            = %10f' % dof
        print '  reduced chi^2  = %10f' % (chisq/dof)
        redchisqs[i]=chisq/dof

    chisq=(infodict['fvec']**2).sum()
    dof=np.product(yl.shape) - len(pfit) # degrees of freedom
    print "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"
    print "converged with chi squared", chisq
    print "degrees of freedom, dof", dof
    print "Reduced chisq (variance of residuals)", chisq/dof
    print
    print "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"
    #for i,pmin in enumerate(pfit):
    #  print "%2i % -10s %12f+/- %10f "%(i,pmin,sqrt(cov[i,i])*sqrt(chisq/dof))
    return pfit, redchisqs

def plot_global(yl, p, xl, irf):
    final_model = model_global(xl[:], p, irf)

    fig=figure()
    suptitle("global")
    ax1=subplot2grid((4,1),(0,0),rowspan=3)
    #ax1.plot(xl[:],irf[:])
    for n in range(0,num_of_files_used):
        ax1.plot(xl[:],yl[n],'+',color='k',linewidth=3, zorder=1)
        ax1.plot(xl[:],final_model[n],'-',color='r',linewidth=2, zorder=2)
    #ax1.set_xlim(2.5,10)
    ax1.set_ylabel('intensity (arb. units)',fontsize=18)
    ax1.yaxis.set_tick_params(labelsize=15)
    ax1.xaxis.set_tick_params(labelsize=14)
    
    ax2=subplot2grid((4,1),(3,0),rowspan=1)
    for n in range(0,num_of_files_used):
        ax2.plot(xl[:],(final_model[n]-yl[n]),',',color='k')#
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
le=num_of_files_used-1
tau1s=zeros(le)
tau2s=zeros(le)
a1=zeros(le)
a2=zeros(le)

if args.lifetimes>0:
    for n in range(1,le+1):
        a = fitdata(files,data,tend,n,p0,xl)
        pfit[n] = a
        tau1s[n-1]= a[2]
        tau2s[n-1]= a[3]
        a1[n-1]= a[0]
        a2[n-1]=a[1]

    fig=figure()
    plot(phs,tau1s, 'ro', label="tau 1")
    plot(phs,tau2s, 'bv', label="tau 2")
    legend()
    fig.show()

    fig=figure()
    plot(phs,a1, 'ro', label="A 1")
    plot(phs,a2, 'bv', label="A 2")
    legend()
    fig.show()

# g is [tau1,tau2,tau3]+offsets+amp1+amp2+amp3
if args.global_lifetimes>0:
    g = ones(4*num_of_files_used + 4)
    for i,file in enumerate(d):
        g[i+3] = d.min() # offset
    g[0]=4  # tau1
    g[1]=2  # tau2
    g[2]=.3
    g[3]=.5
    plot_global(d, g, xl, irf)

    # Rough optimization
    dec = 3 # decimation factor
    gfit,redchisqs=globalfit(d[:,::dec],g,xl[::dec],normalize_irf(irf[::dec]))
    print_global(gfit)
    plot_global(d, gfit, xl, irf)
    #gfit=globalfit(d,gfit,xl,irf)
    #plot_global(d, gfit, xl, irf)
    #print gfit

    (tau1, tau2) = gfit[:2]
    per_curve = gfit[4:]
    amp1 = per_curve[num_of_files_used:2*num_of_files_used]
    amp2 = per_curve[2*num_of_files_used:3*num_of_files_used]

    fig1=figure()
    scatter(phs, amp1*tau1 / (amp1*tau1+amp2*tau2))
    xlabel('pH x10')
    ylabel('% of dye in basic state')
    fig1.show()

    print redchisqs
    fig2=figure()
    scatter(phs, redchisqs)
    xlabel('pH x10')
    ylabel('X2')
    fig2.show()
