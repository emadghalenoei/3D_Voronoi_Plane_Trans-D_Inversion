# -*- coding: utf-8 -*-
"""
Created on Sat Nov 28 15:02:36 2020

@author: emadg
"""
import numpy as np
#import math
def cauchy_dist(x0,b,lower,upper,preference):
    
    x = x0+b*np.tan((3.14159265359)*(np.random.rand()-0.5))
    if x<lower or x>upper: x = preference
    return x