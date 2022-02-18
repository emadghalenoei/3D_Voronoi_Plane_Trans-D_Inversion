# -*- coding: utf-8 -*-
"""
Created on Sat Nov 28 14:18:39 2020

@author: emadg
"""
import numpy as np
from Log_Likelihood import Log_Likelihood
def death(XnYnZn,LogLc,xc,yc,zc,rhoc,T,Kernel_Grv,Kernel_Mag,dg_obs,dT_obs,ARgc,ARTc):
    
    ARgc_2D = ARgc.reshape((4,4))
    ARTc_2D = ARTc.reshape((4,4))
    
    i = np.random.randint(0, np.size(xc))
    xp = xc.copy()
    yp = yc.copy()
    zp = zc.copy()
    rhop = rhoc.copy()
    xp = np.delete(xc, i)
    yp = np.delete(yc, i)
    zp = np.delete(zc, i)
    rhop = np.delete(rhoc, i)
    LogLp = Log_Likelihood(Kernel_Grv,Kernel_Mag,dg_obs,dT_obs,xp,yp,zp,rhop,ARgc_2D,ARTc_2D,XnYnZn)[0]
    MHP = np.exp((LogLp - LogLc)/T)
    if np.random.rand()<=MHP:
        LogLc = LogLp
        xc = xp.copy()
        yc = yp.copy()
        zc = zp.copy()
        rhoc = rhop.copy()

    return [LogLc,xc,yc,zc,rhoc]
     
     
