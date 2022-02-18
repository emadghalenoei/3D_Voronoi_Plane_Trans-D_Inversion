# -*- coding: utf-8 -*-
"""
Created on Fri Nov 27 22:13:56 2020

@author: emadg
"""

import numpy as np
import matplotlib.pyplot as plt
from Chain2xyz import Chain2xyz
from Log_Likelihood import Log_Likelihood

def Imshow_residuals(ChainKeep,Ndatapoints,Kernel_Grv,Kernel_Mag,dg_obs,dT_obs,XnYnZn,fpath,figname,fignum):
    
    ind = np.argsort(ChainKeep[:,0])[::-1]
    Chain_maxL = ChainKeep[ind[0]].copy()
    [x, y, z, rho, k_AR, XYZLine, ARg, ART]= Chain2xyz(Chain_maxL).copy()
    
    ARg_2D = ARg.reshape((4,4))
    ART_2D = ART.reshape((4,4))
    
    [rg, rT] = Log_Likelihood(Kernel_Grv,Kernel_Mag,dg_obs,dT_obs,x,y,z,rho,ARg_2D,ART_2D,XnYnZn)[3:5]
    
    dg_pre = dg_obs - rg
    dT_pre = dT_obs - rT
    
    dg_pre_reshape = dg_pre.reshape((Ndatapoints,Ndatapoints))
    dT_pre_reshape = dT_pre.reshape((Ndatapoints,Ndatapoints))
    dg_obs_reshape = dg_obs.reshape((Ndatapoints,Ndatapoints))
    dT_obs_reshape = dT_obs.reshape((Ndatapoints,Ndatapoints))
    
    fig, axs = plt.subplots(2, 2)
    
    axs[0, 0].plot(dg_obs_reshape[10,:],'k-')
    axs[0, 0].plot(dg_pre_reshape[10,:],'r.-')
    
    axs[0, 1].plot(dT_obs_reshape[20,:],'k-')
    axs[0, 1].plot(dT_pre_reshape[20,:],'r.-')
    
    axs[1, 0].plot(dg_obs_reshape[:,10],'k-')
    axs[1, 0].plot(dg_pre_reshape[:,10],'r.-')
    
    axs[1, 1].plot(dT_obs_reshape[:,20],'k-')
    axs[1, 1].plot(dT_pre_reshape[:,20],'r.-')
    
    plt.show()

    #fig.savefig(fpath+'/'+figname+'.png')
    fig.savefig(fpath+'/'+figname+str(fignum)+'.pdf')
    plt.close(fig)    # close the figure window