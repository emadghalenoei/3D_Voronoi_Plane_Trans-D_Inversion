import numpy as np

def std_online(n,mS,Xn,mX):
    ## mX is mean for previous n-1 samples and Xn is new sample
    ## mS is std for previous n-1 samples
    
    if n == 1:
        s_online = np.zeros(mS.shape)
    else:
        s_online = np.sqrt((((n-2)/(n-1))*(np.power(mS,2)))+((1/n)*np.power(Xn-mX,2))).copy()
    
    return s_online