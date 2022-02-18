# -*- coding: utf-8 -*-
"""
Created on Sat Nov 28 14:52:59 2020

@author: emadg
"""
import numpy as np
from Log_Likelihood import Log_Likelihood
from cauchy_dist import cauchy_dist

def move(XnYnZn,globals_par,LogLc,XYZLinec,xc,yc,zc,rhoc,T,Kernel_Grv,Kernel_Mag,dg_obs,dT_obs,ARgc,ARTc):
    
    ARgc_2D = ARgc.reshape((4,4))
    ARTc_2D = ARTc.reshape((4,4))
    
    Nnode=int(np.size(xc))
    rho_salt_min = globals_par[1,0]
    rho_salt_max = globals_par[1,1]
    rho_base_min = globals_par[2,0]
    rho_base_max = globals_par[2,1]
    zn_min = globals_par[4,0]
    
    X1c = XYZLinec[0].copy()
    X2c = XYZLinec[1].copy()
    Y1c = XYZLinec[2].copy()
    Y2c = XYZLinec[3].copy()
    Z1c = XYZLinec[4].copy()
    Z2c = XYZLinec[5].copy()
    
    minX = np.minimum(X1c,X2c)
    maxX = np.maximum(X1c,X2c)
    minY = np.minimum(Y1c,Y2c)
    maxY = np.maximum(Y1c,Y2c)
    minZ = np.minimum(Z1c,Z2c)
    maxZ = np.maximum(Z1c,Z2c)
    
    dsalt = rho_salt_max-rho_salt_min
    dbase = rho_base_max-rho_base_min
    
#     Mnode = np.minimum(30,Nnode)
#     Npar = np.unique(np.concatenate((np.where(rhoc!=0)[-1],np.random.choice(Nnode, Mnode))))

    a1 = np.where(rhoc!=0)[-1]
    Mnode1 = np.minimum(3,a1.size)
    ind1 = np.random.choice(a1,Mnode1, replace=False)
    
    a0 = np.where(rhoc==0)[-1]
    Mnode0 = np.minimum(2,a0.size)
    ind0 = np.random.choice(a0,Mnode0, replace=False)
    Npar = np.concatenate((ind0, ind1))
    
    
    #for inode in np.arange(Nnode):
    for inode in Npar:
        for ipar in np.arange(1,5): # 1 or 2 or 3 or 4
        
            xp = xc.copy()
            yp = yc.copy()
            zp = zc.copy()
            rhop = rhoc.copy()
            
            if ipar == 1:
                xp[inode] = cauchy_dist(xc[inode],0.2,0,1,xc[inode])
                if np.isclose(xc[inode] , xp[inode])==1: continue
                
            elif ipar == 2:
                yp[inode] = cauchy_dist(yc[inode],0.2,0,1,yc[inode])
                if np.isclose(yc[inode] , yp[inode])==1: continue
            
            elif ipar == 3:
                zp[inode] = cauchy_dist(zc[inode],0.2,zn_min,1,zc[inode])
                if np.isclose(zc[inode] , zp[inode])==1: continue
        
            else:
                if rhoc[inode]<0:
                    rhop[inode] = cauchy_dist(rhoc[inode],0.04,rho_salt_min,rho_salt_max,rhoc[inode])
                    if np.isclose(rhoc[inode] , rhop[inode])==1: continue
                elif rhoc[inode]>0:
                    rhop[inode] = cauchy_dist(rhoc[inode],0.04,rho_base_min,rho_base_max,rhoc[inode])
                    if np.isclose(rhoc[inode] , rhop[inode])==1: continue
        
            
         
        
            if ipar<=3:
 
                logic_sed = ((xp[inode]<minX) or (xp[inode]>maxX) or (yp[inode]<minY) or (yp[inode]>maxY) or (zp[inode]<minZ))and(rhoc[inode]!=0)
                logic_salt = ((xp[inode]>=minX) and (xp[inode]<=maxX) and (yp[inode]>=minY) and (yp[inode]<=maxY) and (zp[inode]>=minZ) and (zp[inode]<=maxZ))and(rhoc[inode]>=0)
                logic_base = ((xp[inode]>=minX) and (xp[inode]<=maxX) and (yp[inode]>=minY) and (yp[inode]<=maxY) and (zp[inode]>maxZ))and(rhoc[inode]<=0)
            
       
            
                if logic_sed==1 or logic_salt==1 or logic_base==1:
                    r = np.random.rand()
                    rhop[inode] = logic_salt*(rho_salt_min+r*dsalt)+logic_base*(rho_base_min+r*dbase).copy()
            
            rhop = rhop.astype('float32')    
            LogLp = Log_Likelihood(Kernel_Grv,Kernel_Mag,dg_obs,dT_obs,xp,yp,zp,rhop,ARgc_2D,ARTc_2D,XnYnZn)[0]
            MHP = np.exp((LogLp - LogLc)/T)
            if np.random.rand()<=MHP:
                    LogLc = LogLp
                    xc = xp.copy()
                    yc = yp.copy()
                    zc = zp.copy()
                    rhoc = rhop.copy()
            
    ### Hyper Parameters
    for ipar in np.arange(1,7): # 1 or 2 or 3 or 4 or 5 or 6
        rhop = rhoc.copy()
        X1p = X1c.copy()
        X2p = X2c.copy()
        Y1p = Y1c.copy()
        Y2p = Y2c.copy()
        Z1p = Z1c.copy()
        Z2p = Z2c.copy()
        
        if ipar==1:
            X1p = cauchy_dist(X1c,0.2,0,1,X1c)
            if np.isclose(X1c , X1p)==1: continue
        elif ipar==2:
            X2p = cauchy_dist(X2c,0.2,0,1,X2c)
            if np.isclose(X2c , X2p)==1: continue
        elif ipar==3:
            Y1p = cauchy_dist(Y1c,0.2,0,1,Y1c)
            if np.isclose(Y1c , Y1p)==1: continue
        elif ipar==4:
            Y2p = cauchy_dist(Y2c,0.2,0,1,Y2c)
            if np.isclose(Y2c , Y2p)==1: continue
        elif ipar==5:
            Z1p = cauchy_dist(Z1c,0.2,zn_min,1,Z1c)
            if np.isclose(Z1c , Z1p)==1: continue
        else:
            Z2p = cauchy_dist(Z2c,0.2,zn_min,1,Z2c)
            if np.isclose(Z2c , Z2p)==1: continue
        
        
        minX = np.minimum(X1p,X2p)
        maxX = np.maximum(X1p,X2p)
        minY = np.minimum(Y1p,Y2p)
        maxY = np.maximum(Y1p,Y2p)
        minZ = np.minimum(Z1p,Z2p)
        maxZ = np.maximum(Z1p,Z2p)
        
        logic_sed = np.logical_and(np.logical_or.reduce((xc<minX,xc>maxX,yc<minY,yc>maxY,zc<minZ)),rhoc!=0)
        logic_salt = np.logical_and(np.logical_and.reduce((xc>=minX , xc<=maxX, yc>=minY , yc<=maxY , zc>=minZ , zc<=maxZ)),rhoc>=0)
        logic_base = np.logical_and(np.logical_and.reduce((xc>=minX , xc<=maxX, yc>=minY , yc<=maxY , zc>maxZ)),rhoc<=0)

        r = np.random.rand()
        rhop[logic_sed] = 0
        rhop[logic_salt] = rho_salt_min+r*dsalt
        rhop[logic_base] = rho_base_min+r*dbase
        
        rhop = rhop.astype('float32')

        LogLp = Log_Likelihood(Kernel_Grv,Kernel_Mag,dg_obs,dT_obs,xc,yc,zc,rhop,ARgc_2D,ARTc_2D,XnYnZn)[0]
        MHP = np.exp((LogLp - LogLc)/T)
        if np.random.rand()<=MHP:
            LogLc = LogLp
            rhoc = rhop.copy()
            X1c = X1p
            X2c = X2p
            Y1c = Y1p
            Y2c = Y2p
            Z1c = Z1p
            Z2c = Z2p
            
    XYZLinec = np.hstack((X1c, X2c, Y1c, Y2c, Z1c, Z2c))
    return [LogLc,XYZLinec,xc,yc,zc,rhoc]