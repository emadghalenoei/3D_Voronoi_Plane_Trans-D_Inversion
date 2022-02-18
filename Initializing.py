# -*- coding: utf-8 -*-
"""
Created on Thu Nov 19 23:22:13 2020

@author: emadg
"""
import numpy as np
from Log_Likelihood import Log_Likelihood
from Chain2xyz import Chain2xyz
# import os
# import sys

def Initializing(Chain,XnYnZn,globals_par,Kernel_Grv,Kernel_Mag,dg_obs,dT_obs,Chain_MaxL,loaddesk):
    
    Kmin = int(globals_par[0,0])
    Kmax = int(globals_par[0,1])
    rho_salt_min = globals_par[1,0]
    rho_salt_max = globals_par[1,1]
    rho_base_min = globals_par[2,0]
    rho_base_max = globals_par[2,1]
    KminAR = int(globals_par[3,0])
    KmaxAR = int(globals_par[3,1])
    zn_min = globals_par[4,0]
    
    if loaddesk == 0:

        Nnode = Kmin
        xc = np.random.rand(Nnode).astype('float32')
        yc = np.random.rand(Nnode).astype('float32')
        zc = (zn_min + np.random.rand(Nnode)*(1-zn_min)).astype('float32')
        X1c = 0.3
        X2c = 0.9
        Y1c = 0.3
        Y2c = 0.9
        Z1c = 0.5
        Z2c = 0.8
        XYZLinec = np.array([X1c, X2c, Y1c, Y2c, Z1c, Z2c])
        r = np.random.rand(Nnode).astype('float32')
        logic_salt = np.logical_and.reduce((xc>=X1c , xc<=X2c , yc>=Y1c , yc<=Y2c , zc>=Z1c , zc<=Z2c)).astype('float32')
        logic_base = np.logical_and.reduce((xc>=X1c , xc<=X2c , yc>=Y1c , yc<=Y2c , zc>Z2c)).astype('float32')

        rhoc = logic_salt*(rho_salt_min+r*(rho_salt_max-rho_salt_min))+(logic_base)*(rho_base_min+r*(rho_base_max-rho_base_min)) 
        rhoc = rhoc.astype('float32')
        
        ARgc_2D =  np.zeros((4,4)).copy() 
        ARTc_2D =  np.zeros((4,4)).copy() 
        
        
       
        
    else:
        [xc, yc, zc, rhoc, k_AR, XYZLinec, ARgc, ARTc]= Chain2xyz(Chain_MaxL)
        ARgc_2D =  np.zeros((4,4))
        ARTc_2D =  np.zeros((4,4))
    
    LogLc = Log_Likelihood(Kernel_Grv,Kernel_Mag,dg_obs,dT_obs,xc,yc,zc,rhoc,ARgc_2D,ARTc_2D,XnYnZn)[0]
    
    ARgc = ARgc_2D.flatten() 
    ARTc = ARTc_2D.flatten() 
    
    Chain[0] = LogLc
    Chain[1] = np.size(xc)
    Chain[2:8] = np.hstack((0, 0, 0, 0, 0, 0))
    Chain[8:14] = XYZLinec.copy()
    Chain[14:30] = ARgc.copy()  
    Chain[30:46] = ARTc.copy() 
    Chain[46:46+np.size(xc)*4] = np.concatenate((xc,yc,zc,rhoc)).copy()
    
    return Chain

    
    