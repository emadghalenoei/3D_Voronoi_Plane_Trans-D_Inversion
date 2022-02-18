# -*- coding: utf-8 -*-
"""
Created on Tue Dec  8 21:58:24 2020

@author: emadg
"""

import numpy as np
import math
def Magnetic_3D_Kernel_Expanded_MD(X,Y,Z,XS,YS,dx,dy,dz,I,D,Fe):
    D0 = D
    I0 = I
    theta = 0
    
    mi = I0
    md = D0
    [mx,my,mz] = dircos(mi,md,theta)
    
    fi = I
    fd = D
    [fx,fy,fz] = dircos(fi,fd,theta)
    
    a12 = mx*fy + my*fx
    a13 = mx*fz + mz*fx
    a23 = my*fz + mz*fy
    
    JT = Fe
    
    CX = np.size(X, axis=1)    # Prisms along x
    CY = np.size(Y, axis=0)    # Prisms along y
    CZ = 1
    CTOT=CX*CY*CZ              # Total number of prisms
    
    A = np.ones((XS.size,CTOT))
    
    Ximat = np.diag(XS.flatten()) @ A
    Yimat = np.diag(YS.flatten()) @ A
    Xjmat = A @ np.diag(X.flatten())
    Yjmat = A @ np.diag(Y.flatten())
    Zjmat = A @ np.diag(Z.flatten())
    
    shiftx = (dx/2)*A
    shifty = (dy/2)*A
    shiftz = (dz/2)*A
    
    X1 = Ximat - (Xjmat-shiftx)
    X2 = Ximat - (Xjmat+shiftx)
    Y1 = Yimat - (Yjmat-shifty)
    Y2 = Yimat - (Yjmat+shifty)
    Z1 =  0    - (Zjmat-shiftz)
    Z2 =  0    - (Zjmat+shiftz)
    
    # Distance and Angles
    
    X1p = np.power(X1,2)
    X2p = np.power(X2,2)
    Y1p = np.power(Y1,2)
    Y2p = np.power(Y2,2)
    Z1p = np.power(Z1,2)
    Z2p = np.power(Z2,2)
    
    R111 = np.power(X1p+Y1p+Z1p,0.5)
    R112 = np.power(X1p+Y1p+Z2p,0.5)
    R121 = np.power(X1p+Y2p+Z1p,0.5)
    R122 = np.power(X1p+Y2p+Z2p,0.5)
    R211 = np.power(X2p+Y1p+Z1p,0.5)
    R212 = np.power(X2p+Y1p+Z2p,0.5)
    R221 = np.power(X2p+Y2p+Z1p,0.5)
    R222 = np.power(X2p+Y2p+Z2p,0.5)
    
    #### the only non-zero argument is arg6 :)
    ATAN111 = np.arctan2(X1*Y1,  Z1*R111)
    ATAN112 = np.arctan2(X1*Y1,  Z2*R112)
    ATAN121 = np.arctan2(X1*Y2,  Z1*R121)
    ATAN122 = np.arctan2(X1*Y2,  Z2*R122)
    ATAN211 = np.arctan2(X2*Y1,  Z1*R211)
    ATAN212 = np.arctan2(X2*Y1,  Z2*R212)
    ATAN221 = np.arctan2(X2*Y2,  Z1*R221)
    ATAN222 = np.arctan2(X2*Y2,  Z2*R222)

    ATAN111 = ATAN111 + 2*math.pi*(ATAN111<0)
    ATAN112 = ATAN112 + 2*math.pi*(ATAN112<0)
    ATAN121 = ATAN121 + 2*math.pi*(ATAN121<0)
    ATAN122 = ATAN122 + 2*math.pi*(ATAN122<0)
    ATAN211 = ATAN211 + 2*math.pi*(ATAN211<0)
    ATAN212 = ATAN212 + 2*math.pi*(ATAN212<0)
    ATAN221 = ATAN221 + 2*math.pi*(ATAN221<0)
    ATAN222 = ATAN222 + 2*math.pi*(ATAN222<0)

    ARG6 = -ATAN111 + ATAN112 + ATAN121 - ATAN122 + ATAN211 - ATAN212 - ATAN221 + ATAN222
    ARG6 = mz*fz*ARG6
    A = ARG6
    A = JT* A
    return A
    


def dircos(incl,decl,azim):
    a = math.cos(incl)*math.cos(decl-azim)
    b = math.cos(incl)*math.sin(decl-azim)
    c = math.sin(incl)
    return a,b,c
    

    
    