# -*- coding: utf-8 -*-
"""
Created on Sat Nov 28 19:02:14 2020

@author: emadg
"""
import numpy as np
def TEMPSWAP(T,TAB0):

    TAB = TAB0.copy()
    n=TAB0.shape[0]
    
    p = np.random.randint(0, n)
    q = np.random.randint(0, n)
    
    if T[p]!=T[q]:
        Prob=np.exp(((1/T[p])-(1/T[q]))*(TAB[q,0]-TAB[p,0]))
        if np.random.rand()<=Prob:
            TAB[p][:]=TAB0[q][:].copy()
            TAB[q][:]=TAB0[p][:].copy()
                
        
    return TAB
        
        
    