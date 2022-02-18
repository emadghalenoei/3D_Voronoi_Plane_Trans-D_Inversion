# -*- coding: utf-8 -*-
"""
Created on Sat Nov 28 12:44:27 2020

@author: emadg
"""
import numpy as np

def select_step_AR(Kmin,Kmax,Nnode,Prob):
    
    r = np.random.rand()
    
    if Nnode == Kmin: step = 91
        
    elif Nnode == Kmax: step= 92 if r <= Prob else 93

    else:
        if r<=Prob:
            step = 91
        elif r>Prob and r<=2*Prob:
            step = 92
        else:
            step = 93  
    return step