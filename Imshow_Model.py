# -*- coding: utf-8 -*-
"""
Created on Fri Nov 27 22:13:56 2020

@author: emadg
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Nov 19 22:41:20 2020

@author: emadg
"""

import numpy as np
import matplotlib.pyplot as plt

def Imshow_Model(x1,x2,z1,z2,globals_par,ModelGrid,fpath,figname):
    fig = plt.figure()
    rho_salt_min = globals_par[1,0]
    rho_base_max = globals_par[2,1]
    Xticklabels = np.linspace(x1,x2,5)
    Zticklabels = np.linspace(z2,z1,5)

    plt.imshow(ModelGrid,interpolation='none',
           vmin=rho_salt_min, vmax=rho_base_max,extent=(x1,x2,z2,z1), aspect=(x2-x1)/z2)
    plt.xticks(Xticklabels)
    plt.yticks(Zticklabels)
            
    plt.xlabel("X Profile (m)")
    plt.ylabel("Depth (m)")

    cbar = plt.colorbar()
    plt.set_cmap('bwr')
    cbar.ax.set_ylabel('density contrast ($\mathregular{g/cm^{3}}$)')
    # plt.pause(0.00001)
    # plt.draw()
    fig.savefig(fpath+'/'+figname+'.png')
    fig.savefig(fpath+'/'+figname+'.pdf')
    plt.close(fig)    # close the figure window