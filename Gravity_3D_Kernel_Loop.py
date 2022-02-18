# -*- coding: utf-8 -*-
"""
Created on Mon Dec  7 02:10:03 2020

@author: emadg
"""
import numpy as np
import math
def Gravity_3D_Kernel_Loop(X,Y,Z,XS,YS):
    dx = abs(X[0,0,1]-X[0,0,0])
    dy = abs(Y[0,1,0]-Y[0,0,0])
    dz = abs(Z[1,0,0]-Z[0,0,0])
    
    cGrav = 6.67408e-11
    CX = np.size(X, axis=2)    # Prisms along x
    CY = np.size(Y, axis=1)    # Prisms along y
    CZ = np.size(Z, axis=0)    # Prisms along z
    CTOT=CX*CY*CZ              # Total number of prisms
    
    A = np.ones((XS.size,CTOT))
    row = -1
    
    for xs in np.arange(np.size(XS, axis=1)):   # 0 to CX-1
        for ys in np.arange(np.size(YS, axis=0)):
            row = row + 1
            x0 = XS[ys,xs]
            y0 = YS[ys,xs]
            col = -1
            x = np.zeros(2)
            y = np.zeros(2)
            z = np.zeros(2)
            for cz in np.arange(CZ):
                for cx in np.arange(CX):
                    for cy in np.arange(CY):
                        col = col + 1
                        x[0] = x0 - (X[cz,cy,cx]-dx/2)
                        y[0] = y0 - (Y[cz,cy,cx]-dy/2)
                        z[0] =  0 - (Z[cz,cy,cx]-dz/2)
                        x[1] = x0 - (X[cz,cy,cx]+dx/2)
                        y[1] = y0 - (Y[cz,cy,cx]+dy/2)
                        z[1] =  0 - (Z[cz,cy,cx]+dz/2)
                        Effect = 0
                        for i in np.arange(2): # 0 to 1
                            for j in np.arange(2): # 0 to 1
                                for k in np.arange(2): # 0 to 1
                                    rijk = np.sqrt(np.power(x[i],2)+np.power(y[j],2)+np.power(z[k],2))
                                    ijk = np.power(-1,i+1)*np.power(-1,j+1)*np.power(-1,k+1)
                                    arg1 = np.arctan2(x[i]*y[j],z[k]*rijk)
                                    arg1 = arg1 + 2*math.pi*(arg1<0)
                                    arg2 = rijk + y[j]
                                    arg3 = rijk + x[i]
                                    arg2 = np.log(arg2)
                                    arg3 = np.log(arg3)
                                    Effect = Effect + ijk*(z[k]*arg1 -x[i]*arg2 - y[j]*arg3)
                                
                        A[row,col] = Effect
                        
    A = cGrav * A
    return A
                                    
    
    