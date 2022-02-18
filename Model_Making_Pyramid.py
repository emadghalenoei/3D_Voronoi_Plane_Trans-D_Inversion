# -*- coding: utf-8 -*-
"""
Created on Thu Jul 29 21:57:11 2021

@author: emadg
"""

import numpy as np
import alphashape

def Model_Making_Pyramid(Xn_3D,Yn_3D,Zn_3D,XnYnZn):

    A = np.array([0.6,0.6,0.5]) 
    B = np.array([0.45,0.75,0.8]) 
    C = np.array([0.45,0.45,0.8]) 
    D = np.array([0.75,0.45,0.8]) 
    E = np.array([0.75,0.75,0.8]) 

    xc = np.array([A[0],B[0],C[0],D[0],E[0]])
    yc = np.array([A[1],B[1],C[1],D[1],E[1]])
    zc = np.array([A[2],B[2],C[2],D[2],E[2]])


    TrainPoints = np.column_stack((xc,yc,zc)).copy()
    hull = alphashape.alphashape(TrainPoints,0.1)
    Salt1 = hull.contains(XnYnZn)
    
    CX = Xn_3D.shape[0]
    CY = Xn_3D.shape[0]
    CZ = Xn_3D.shape[0]
    
    Salt1 = Salt1.reshape(CX,CY,CZ)
    Salt1 = Salt1*-0.25
    ###########################################################
    logic_salt2 = (Xn_3D>0.25) & (Xn_3D<0.45) & (Yn_3D>0.25) & (Yn_3D<0.65) & (Zn_3D>0.6) & (Zn_3D<=0.8)
    Salt2 = np.zeros(np.shape(Xn_3D))
    Salt2[logic_salt2] = -0.35

    logic_base1 = (Xn_3D>0.15) & (Xn_3D<0.85) & (Yn_3D>0.15) & (Yn_3D<0.85) & (Zn_3D>0.9)
    Base1 = np.zeros(np.shape(Xn_3D))
    Base1[logic_base1] = 0.35

    logic_base2 = (Xn_3D>0.2) & (Xn_3D<0.8) & (Yn_3D>0.2) & (Yn_3D<0.8) & (Zn_3D>0.8) & (Zn_3D<=0.9)
    Base2 = np.zeros(np.shape(Xn_3D))
    Base2[logic_base2] = 0.25
      
    
    TrueDensityModel = Base1 + Base2 + Salt1 + Salt2
    TrueSUSModel = TrueDensityModel/50
    TrueSUSModel[TrueDensityModel<0.2]=0
    return TrueDensityModel,TrueSUSModel


