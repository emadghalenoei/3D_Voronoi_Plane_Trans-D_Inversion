# -*- coding: utf-8 -*-
"""
Created on Thu Dec 10 18:27:42 2020

@author: emadg
"""

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

def Imshow_Data(x1,x2,y1,y2,DataGrid,fpath,figname):
    fig = plt.figure()
    Xticklabels = np.linspace(x1,x2,5)
    Yticklabels = np.linspace(y1,y2,5)

    plt.imshow(DataGrid,interpolation='none',
           extent=(x1,x2,y1,y2), aspect=(x2-x1)/(y2-y1))
    plt.xticks(Xticklabels)
    plt.yticks(Yticklabels)
            
    plt.xlabel("X Profile (km)")
    plt.ylabel("Y Profile (km)")

    cbar = plt.colorbar()
    plt.set_cmap('jet')
    cbar.ax.set_ylabel('Gravity Anomaly (mGal)')
    # plt.pause(0.00001)
    # plt.draw()
    fig.savefig(fpath+'/'+figname+'.png')
    fig.savefig(fpath+'/'+figname+'.pdf')
    plt.close(fig)    # close the figure window