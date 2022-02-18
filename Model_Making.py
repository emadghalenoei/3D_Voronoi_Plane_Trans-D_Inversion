# -*- coding: utf-8 -*-
"""
Created on Mon Dec  7 02:00:08 2020

@author: emadg
"""

import numpy as np

def Model_Making(Xn_3D,Yn_3D,Zn_3D):
    logic_base1 = (Xn_3D>0.15) & (Xn_3D<0.85) & (Yn_3D>0.15) & (Yn_3D<0.85) & (Zn_3D>0.9)
    Base1 = np.zeros(np.shape(Xn_3D))
    Base1[logic_base1] = 0.35

    logic_base2 = (Xn_3D>0.25) & (Xn_3D<0.75) & (Yn_3D>0.25) & (Yn_3D<0.75) & (Zn_3D>0.75) & (Zn_3D<=0.9)
    Base2 = np.zeros(np.shape(Xn_3D))
    Base2[logic_base2] = 0.25

    logic_salt1 = (Xn_3D>0.3) & (Xn_3D<0.4) & (Yn_3D>0.3) & (Yn_3D<0.4) & (Zn_3D>0.40) & (Zn_3D<=0.75)
    Salt1 = np.zeros(np.shape(Xn_3D))
    Salt1[logic_salt1] = -0.25

    logic_salt2 = (Xn_3D>0.6) & (Xn_3D<0.7) & (Yn_3D>0.6) & (Yn_3D<0.7) & (Zn_3D>0.55) & (Zn_3D<=0.75)
    Salt2 = np.zeros(np.shape(Xn_3D))
    Salt2[logic_salt2] = -0.35

    TrueDensityModel = Base1 + Base2 + Salt1 + Salt2
    TrueSUSModel = TrueDensityModel/50
    TrueSUSModel[TrueDensityModel<0.2]=0
    return TrueDensityModel,TrueSUSModel
    