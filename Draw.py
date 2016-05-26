#coding:utf-8
import numpy as np
import pylab as pl
bx= np.array((50,100,200,400,600,800,1000,2000,3304))
kx=np.array((200,400,600,800,1000,2000,3304))
svmabc=np.array((97.125,)*7)
#bMTL
ba=np.array((95.25,95.125,93.125,92.5,91.75,90.875,90.375,88.75,86.75))
ka=np.array((90.125,88.125,88.375,88.875,88.875,87.375,86.625))
kwa=np.array((92.375,91.125,91.875,91.25,91.125,91.5,91.75))

#bCMTL
bb=np.array((87.25,86.25,87.75,87.0,87.25,87.125,87.5,87.5,86.75))
kb=np.array((83.0,83.125,83.25,84.25,86.0,86.5,86.625))
kwb=np.array((82.0,83.875,85.0,84.625,85.625,85.625,86.125))
#bCEMTL
bc=np.array((95.125,95.125,93.125,92.375,91.875,90.875,90.25,88.75,86.75))
kc=np.array((90.5,87.75,88.5,88.875,88.625,87.75,86.625))
kwc=np.array((92.375,91.0,91.75,91.5,91.0,91.5,91.75))
#
pl.plot(bx,ba,label="bMTL",color="red")
pl.plot(bx,bb,label="bCMTL",color="blue")
pl.plot(bx,bc,label="bCEMTL",color="green")
pl.plot(kx,ka,"r--",label="kMTL")
pl.plot(kx,kb,"b--" ,label="kCMTL")
pl.plot(kx,kc,"g--",label="kCEMTL")
pl.plot(kx,kwa,"r-o",label="kwMTL")
pl.plot(kx,kwb,"b-o" ,label="kwCMTL")
pl.plot(kx,kwc,"g-o",label="kwCEMTL")
pl.plot(kx,svmabc,"y--",label="svm")
pl.xlabel("attributes")
pl.ylabel("accuracy")
pl.legend()
pl.show()
