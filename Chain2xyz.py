# -*- coding: utf-8 -*-
"""
Created on Sat Nov 28 12:25:37 2020

@author: emadg
"""
def Chain2xyz(Chain):
    Nnode = int(Chain[1])
    k_AR = Chain[2:8].copy()
    XYZLine = Chain[8:14].copy()
    ARg = Chain[14:30].copy()
    ART = Chain[30:46].copy()
    x = Chain[46:46+Nnode].copy()
    y = Chain[46+Nnode:46+2*Nnode].copy()
    z = Chain[46+2*Nnode:46+3*Nnode].copy()
    rho = Chain[46+3*Nnode:46+4*Nnode].copy()
    return [x, y, z, rho, k_AR, XYZLine, ARg, ART]
