# -*- coding: utf-8 -*-
"""
Created on Thu Nov 19 23:22:13 2020

@author: emadg
"""
import numpy as np
from scipy.sparse import csr_matrix
import pywt

def Kernel_compressor(Ndatapoints,CX,CY,CZ,Kernel_Grv,Kernel_Mag,fpath_loaddesk):
    
    wname = 'db4'
    wv = pywt.Wavelet(wname)
    Nlevel = 2
    thrg = 0.001
    thrT = 0.7
    Wmode = 'periodization'
    
    Gkernel = np.zeros((Ndatapoints*Ndatapoints,CX*CY*CZ))
    Mkernel = np.zeros((Ndatapoints*Ndatapoints,CX*CY*CZ))

    for irow in np.arange(Ndatapoints*Ndatapoints):

        Kernelsplit = Kernel_Grv[irow,:].copy()
        Gi = Kernelsplit.reshape((CX,CY,CZ))
        Gi_coeff  = pywt.wavedecn(Gi, wv, mode= Wmode, level=Nlevel)
        Gi_3D_coeff = pywt.coeffs_to_array(Gi_coeff)[0]
        Gi_3D_coeff[abs(Gi_3D_coeff)<thrg] = 0
        Gi_3D_coeff_row = Gi_3D_coeff.reshape((1,CX*CY*CZ))
        Gkernel[irow,:] = Gi_3D_coeff_row

        Kernelsplit = Kernel_Mag[irow,:].copy()
        Mi = Kernelsplit.reshape((CX,CY,CZ))
        Mi_coeff  = pywt.wavedecn(Mi, wv, mode= Wmode, level=Nlevel)
        Mi_3D_coeff = pywt.coeffs_to_array(Mi_coeff)[0]
        Mi_3D_coeff[abs(Mi_3D_coeff)<thrT] = 0
        Mi_3D_coeff_row = Mi_3D_coeff.reshape((1,CX*CY*CZ))
        Mkernel[irow,:] = Mi_3D_coeff_row   



    Gkernelsp = csr_matrix(Gkernel)
    Mkernelsp = csr_matrix(Mkernel)

    
    return Gkernelsp, Mkernelsp
    
    
    
    