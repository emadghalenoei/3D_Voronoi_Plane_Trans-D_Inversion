import numpy as np
def mean_online(n,Xn,mX):
    ## mX is mean for previous n-1 samples and Xn is new sample
    if n == 1:
        m_online = Xn.copy()
    else:
        m_online = ((Xn+(n-1)*mX)/n).copy()
    
    return m_online