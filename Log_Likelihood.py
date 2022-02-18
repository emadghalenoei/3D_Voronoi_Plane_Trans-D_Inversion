# -*- coding: utf-8 -*-
"""
Created on Thu Nov 19 23:06:46 2020

@author: emadg
"""

import numpy as np
import faiss 
from numpy.fft  import fft2, ifft2
from Model_compressor import Model_compressor

# from profile_each_line import profile_each_line
#import math
# @profile_each_line
def Log_Likelihood(Kernel_Grv,Kernel_Mag,dg_obs,dT_obs,x,y,z,rho,ARg,ART,XnYnZn):
    
    TrainPoints = np.column_stack((x,y,z)).copy()
    index = faiss.IndexFlatL2(3)
    index.add(TrainPoints)
    D, I = index.search(XnYnZn, 1)     # actual search    
    DensityModel = rho[I[:,0]].copy() 
#     tf = DensityModel!=0
#     rg = dg_obs - (Kernel_Grv[:,tf] @ DensityModel[tf])

    SusModel = DensityModel/50.
    SusModel[DensityModel<0.2]=0
#     tf = SusModel!=0
#     rT = dT_obs - (Kernel_Mag[:,tf] @ SusModel[tf]) #(nT)

    [dg_pre, dT_pre] = Model_compressor(DensityModel,SusModel,Kernel_Grv,Kernel_Mag) 
    rg = dg_obs - dg_pre
    rT = dT_obs - dT_pre
    
    N = len(rg)
    SqN = int(np.sqrt(N))

    sigma_rg = np.linalg.norm(rg)/SqN 
    sigma_rT = np.linalg.norm(rT)/SqN
    
    if sigma_rg<0.1: sigma_rg = 0.1
    if sigma_rT<0.1: sigma_rT = 0.1
    
    
    rg_2D = rg.reshape((SqN,SqN))
    rg_2D_padd = np.pad(rg_2D, [(3,3),(3,3)])  
    da_g = np.real(ifft2(fft2(rg_2D_padd)*fft2(ARg, s=rg_2D_padd.shape)))
    
    rT_2D = rT.reshape((SqN,SqN))
    rT_2D_padd = np.pad(rT_2D, [(3,3),(3,3)])  
    da_T = np.real(ifft2(fft2(rT_2D_padd)*fft2(ART, s=rT_2D_padd.shape)))
    
    uncor_g_padd = rg_2D_padd - da_g
    uncor_T_padd = rT_2D_padd - da_T
    
    uncor_g = uncor_g_padd[3:-3,3:-3].copy()
    uncor_T = uncor_T_padd[3:-3,3:-3].copy()
    
    LogL = -N*np.log(sigma_rg*sigma_rT) - (0.5*np.sum((uncor_g/sigma_rg)**2)) - (0.5*np.sum((uncor_T/sigma_rT)**2))
    
    return LogL, DensityModel, SusModel, rg, rT, sigma_rg, sigma_rT, uncor_g, uncor_T

#     LogL = 0.
#     rg = 1.
#     return LogL, rg

