# -*- coding: utf-8 -*-
"""
Created on Sat Nov 28 13:51:33 2020

@author: emadg
"""
import numpy as np
from Log_Likelihood import Log_Likelihood
def birth(XnYnZn,globals_par,LogLc,XYZLinec,xc,yc,zc,rhoc,T,Kernel_Grv,Kernel_Mag,dg_obs,dT_obs,ARgc,ARTc):
    
    ARgc_2D = ARgc.reshape((4,4))
    ARTc_2D = ARTc.reshape((4,4))
    
    rho_salt_min = globals_par[1,0]
    rho_salt_max = globals_par[1,1]
    rho_base_min = globals_par[2,0]
    rho_base_max = globals_par[2,1]
    zn_min = globals_par[4,0]
    
    X1 = XYZLinec[0].copy() 
    X2 = XYZLinec[1].copy() 
    Y1 = XYZLinec[2].copy() 
    Y2 = XYZLinec[3].copy() 
    Z1 = XYZLinec[4].copy() 
    Z2 = XYZLinec[5].copy() 
    
    minX = np.minimum(X1,X2)
    maxX = np.maximum(X1,X2)
    minY = np.minimum(Y1,Y2)
    maxY = np.maximum(Y1,Y2)
    minZ = np.minimum(Z1,Z2)
    maxZ = np.maximum(Z1,Z2)
    
    xp = np.random.rand()
    yp = np.random.rand()
    zp = zn_min + np.random.rand()*(1-zn_min)

    r = np.random.rand()
    logic_salt = (xp>=minX) & (xp<=maxX) & (yp>=minY) & (yp<=maxY) & (zp>=minZ) & (zp<=maxZ)
    logic_salt = logic_salt.astype(float)
    logic_base = (xp>=minX) & (xp<=maxX) & (yp>=minY) & (yp<=maxY) & (zp>maxZ)
    logic_base = logic_base.astype(float)
    rhop = logic_salt*(rho_salt_min+r*(rho_salt_max-rho_salt_min))+(logic_base)*(rho_base_min+r*(rho_base_max-rho_base_min))
    
    xp = np.append(xc,xp).astype('float32').copy()
    yp = np.append(yc,yp).astype('float32').copy()
    zp = np.append(zc,zp).astype('float32').copy()
    rhop = np.append(rhoc,rhop).astype('float32').copy()
    
    LogLp = Log_Likelihood(Kernel_Grv,Kernel_Mag,dg_obs,dT_obs,xp,yp,zp,rhop,ARgc_2D,ARTc_2D,XnYnZn)[0]
    MHP = np.exp((LogLp - LogLc)/T)
    if np.random.rand()<=MHP:
            LogLc = LogLp
            xc = xp.copy() 
            yc = yp.copy() 
            zc = zp.copy() 
            rhoc = rhop.copy() 
        
    return [LogLc,xc,yc,zc,rhoc]


