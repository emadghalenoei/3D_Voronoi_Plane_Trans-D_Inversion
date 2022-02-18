import numpy as np
from Log_Likelihood import Log_Likelihood
from select_step_AR import select_step_AR
from cauchy_dist import cauchy_dist

def ARxy_T(k_ARc,XnYnZn,globals_par,AR_bounds,LogLc,xc,yc,zc,rhoc,ARgc,ARTc,T,Kernel_Grv,Kernel_Mag,dg_obs,dT_obs,bk_AR):
    
    kAR = int(k_ARc[5])
    step = select_step_AR(globals_par[3,0],globals_par[3,1],kAR,bk_AR[kAR])
    ARgc_2D = ARgc.reshape((4,4))
    ARTc_2D = ARTc.reshape((4,4))

    if step==91:
        AR_min = AR_bounds[kAR+1,0]
        AR_max = AR_bounds[kAR+1,1]
        arp = AR_min + np.random.rand() * (AR_max-AR_min)
        ARTp_2D = ARTc_2D.copy()
        ARTp_2D[kAR+1,kAR+1] = arp
        
        LogLp = Log_Likelihood(Kernel_Grv,Kernel_Mag,dg_obs,dT_obs,xc,yc,zc,rhoc,ARgc_2D,ARTp_2D,XnYnZn)[0]
        MHP = (bk_AR[kAR+1]/bk_AR[kAR]) * np.exp((LogLp - LogLc)/T)
        if np.random.rand()<=MHP:
            LogLc = LogLp
            ARTc_2D = ARTp_2D.copy()
            k_ARc[5] += 1 
            
    
    if step ==92:
        ARTp_2D = ARTc_2D.copy()
        ARTp_2D[kAR,kAR] = 0.
        LogLp = Log_Likelihood(Kernel_Grv,Kernel_Mag,dg_obs,dT_obs,xc,yc,zc,rhoc,ARgc_2D,ARTp_2D,XnYnZn)[0]
        MHP = (bk_AR[kAR-1]/bk_AR[kAR]) * np.exp((LogLp - LogLc)/T)
        if np.random.rand()<=MHP:
            LogLc = LogLp
            ARTc_2D = ARTp_2D.copy()
            k_ARc[5] -= 1 
            
        
    else:  
        for iar in np.arange(kAR):
            AR_min = AR_bounds[iar+1,0]
            AR_max = AR_bounds[iar+1,1]
            std_cauchy = abs(AR_max-AR_min)/40
            ARTp_2D = ARTc_2D.copy()
            ARTp_2D[iar+1,iar+1] = cauchy_dist(ARTc_2D[iar+1,iar+1],std_cauchy,AR_min,AR_max,ARTc_2D[iar+1,iar+1])
            if np.isclose(ARTc_2D[iar+1,iar+1] , ARTp_2D[iar+1,iar+1])==1: continue
                
            LogLp = Log_Likelihood(Kernel_Grv,Kernel_Mag,dg_obs,dT_obs,xc,yc,zc,rhoc,ARgc_2D,ARTp_2D,XnYnZn)[0]    
            MHP = np.exp((LogLp - LogLc)/T)
            
            if np.random.rand()<=MHP:
                LogLc = LogLp
                ARTc_2D = ARTp_2D.copy()           
    
    ARTc = ARTc_2D.flatten() 
    return [LogLc,ARTc,k_ARc]
        
