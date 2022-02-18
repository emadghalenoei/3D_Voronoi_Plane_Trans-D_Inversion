# -*- coding: utf-8 -*-
"""
Created on Tue Dec  8 13:20:56 2020

@author: emadg
"""

import numpy as np
import math
def Gravity_3D_Kernel_Expanded(X,Y,Z,XS,YS):
    dx = abs(X[0,0,1]-X[0,0,0])
    dy = abs(Y[0,1,0]-Y[0,0,0])
    dz = abs(Z[1,0,0]-Z[0,0,0])
    
    cGrav = 6.67408e-11
    CX = np.size(X, axis=2)    # Prisms along x
    CY = np.size(Y, axis=1)    # Prisms along y
    CZ = np.size(Z, axis=0)    # Prisms along z
    CTOT=CX*CY*CZ              # Total number of prisms
    
    A = np.ones((XS.size,CTOT))
    
    # XT = np.transpose(X, (1, 2, 0))
    # YT = np.transpose(Y, (1, 2, 0))
    # ZT = np.transpose(Z, (1, 2, 0))

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
    
    ATAN111 = np.arctan2(X1*Y1,Z1*R111)
    ATAN112 = np.arctan2(X1*Y1,Z2*R112)
    ATAN121 = np.arctan2(X1*Y2,Z1*R121)
    ATAN122 = np.arctan2(X1*Y2,Z2*R122)
    ATAN211 = np.arctan2(X2*Y1,Z1*R211)
    ATAN212 = np.arctan2(X2*Y1,Z2*R212)
    ATAN221 = np.arctan2(X2*Y2,Z1*R221)
    ATAN222 = np.arctan2(X2*Y2,Z2*R222)
    
    ATAN111 = ATAN111 + 2*math.pi*(ATAN111<0)
    ATAN112 = ATAN112 + 2*math.pi*(ATAN112<0)
    ATAN121 = ATAN121 + 2*math.pi*(ATAN121<0)
    ATAN122 = ATAN122 + 2*math.pi*(ATAN122<0)
    ATAN211 = ATAN211 + 2*math.pi*(ATAN211<0)
    ATAN212 = ATAN212 + 2*math.pi*(ATAN212<0)
    ATAN221 = ATAN221 + 2*math.pi*(ATAN221<0)
    ATAN222 = ATAN222 + 2*math.pi*(ATAN222<0)
    
    Term111 = Z1* ATAN111 - X1* np.log(R111+Y1) - Y1* np.log(R111+X1)
    Term112 = Z2* ATAN112 - X1* np.log(R112+Y1) - Y1* np.log(R112+X1)
    Term121 = Z1* ATAN121 - X1* np.log(R121+Y2) - Y2* np.log(R121+X1)
    Term122 = Z2* ATAN122 - X1* np.log(R122+Y2) - Y2* np.log(R122+X1)
    Term211 = Z1* ATAN211 - X2* np.log(R211+Y1) - Y1* np.log(R211+X2)
    Term212 = Z2* ATAN212 - X2* np.log(R212+Y1) - Y1* np.log(R212+X2)
    Term221 = Z1* ATAN221 - X2* np.log(R221+Y2) - Y2* np.log(R221+X2)
    Term222 = Z2* ATAN222 - X2* np.log(R222+Y2) - Y2* np.log(R222+X2)
    
    A = - Term111 + Term112 + Term121 - Term122 + Term211 - Term212 - Term221 + Term222
    A = cGrav* A
    return A
    
    
    
    
    
    
    
    