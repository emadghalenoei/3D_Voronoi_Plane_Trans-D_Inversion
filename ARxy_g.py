import numpy as np
from Log_Likelihood import Log_Likelihood
from select_step_AR import select_step_AR
from cauchy_dist import cauchy_dist

def ARxy_g(k_ARc,XnYnZn,globals_par,AR_bounds,LogLc,xc,yc,zc,rhoc,ARgc,ARTc,T,Kernel_Grv,Kernel_Mag,dg_obs,dT_obs,bk_AR):
    
    kAR = int(k_ARc[2])
    step = select_step_AR(globals_par[3,0],globals_par[3,1],kAR,bk_AR[kAR])
    ARgc_2D = ARgc.reshape((4,4))
    ARTc_2D = ARTc.reshape((4,4))

    if step==91:
        AR_min = AR_bounds[kAR+1,0]
        AR_max = AR_bounds[kAR+1,1]
        arp = AR_min + np.random.rand() * (AR_max-AR_min)
        ARgp_2D = ARgc_2D.copy()
        ARgp_2D[kAR+1,kAR+1] = arp
        
        LogLp = Log_Likelihood(Kernel_Grv,Kernel_Mag,dg_obs,dT_obs,xc,yc,zc,rhoc,ARgp_2D,ARTc_2D,XnYnZn)[0]
        MHP = (bk_AR[kAR+1]/bk_AR[kAR]) * np.exp((LogLp - LogLc)/T)
        if np.random.rand()<=MHP:
            LogLc = LogLp
            ARgc_2D = ARgp_2D.copy()
            k_ARc[2] += 1 
            
    
    if step ==92:
        ARgp_2D = ARgc_2D.copy()
        ARgp_2D[kAR,kAR] = 0.
        LogLp = Log_Likelihood(Kernel_Grv,Kernel_Mag,dg_obs,dT_obs,xc,yc,zc,rhoc,ARgp_2D,ARTc_2D,XnYnZn)[0]
        MHP = (bk_AR[kAR-1]/bk_AR[kAR]) * np.exp((LogLp - LogLc)/T)
        if np.random.rand()<=MHP:
            LogLc = LogLp
            ARgc_2D = ARgp_2D.copy()
            k_ARc[2] -= 1 
            
        
    else:  
        for iar in np.arange(kAR):
            AR_min = AR_bounds[iar+1,0]
            AR_max = AR_bounds[iar+1,1]
            std_cauchy = abs(AR_max-AR_min)/40
            ARgp_2D = ARgc_2D.copy()
            ARgp_2D[iar+1,iar+1] = cauchy_dist(ARgc_2D[iar+1,iar+1],std_cauchy,AR_min,AR_max,ARgc_2D[iar+1,iar+1])
            if np.isclose(ARgc_2D[iar+1,iar+1] , ARgp_2D[iar+1,iar+1])==1: continue
                
            LogLp = Log_Likelihood(Kernel_Grv,Kernel_Mag,dg_obs,dT_obs,xc,yc,zc,rhoc,ARgp_2D,ARTc_2D,XnYnZn)[0]    
            MHP = np.exp((LogLp - LogLc)/T)
            
            if np.random.rand()<=MHP:
                LogLc = LogLp
                ARgc_2D = ARgp_2D.copy()           
    
    ARgc = ARgc_2D.flatten() 
    return [LogLc,ARgc,k_ARc]
        
