# -*- coding: utf-8 -*-
"""
Created on Fri Nov 27 16:14:42 2020

@author: emadg
"""
from mpi4py import MPI
import os
import sys
import shutil
import numpy as np
import scipy.sparse
#from scipy.interpolate import griddata
from Model_Making import Model_Making
from Gravity_3D_Kernel_Expanded_MD import Gravity_3D_Kernel_Expanded_MD
from Magnetic_3D_Kernel_Expanded_MD import Magnetic_3D_Kernel_Expanded_MD
from Imshow_residuals import Imshow_residuals
import matplotlib.pyplot as plt
from Initializing import Initializing
from Chain2xyz import Chain2xyz
from select_step import select_step
from birth import birth
from death import death
from move import move
from datetime import datetime
# from scipy.signal import convolve2d
# from scipy.linalg import toeplitz
from ARx_g import ARx_g
from ARy_g import ARy_g
from ARxy_g import ARxy_g
from ARx_T import ARx_T
from ARy_T import ARy_T
from ARxy_T import ARxy_T
from Kernel_compressor import Kernel_compressor
from scipy.sparse import csr_matrix
from Model_Making_Pyramid import Model_Making_Pyramid


comm = MPI.COMM_WORLD
rank = comm.Get_rank()
status = MPI.Status()
Nchain = comm.Get_size()-1 # No. MCMC chains or MPI threads


plt.close('all')

Ndatapoints = 32         # Number of total data in 1D array shape
CX = 64                 
CY = 64                 
CZ = 64                 


XnYnZn = np.zeros(CX*CY*CZ).astype('float32')
dg_obs = np.zeros(Ndatapoints*Ndatapoints,dtype=float)
dT_obs = np.zeros(Ndatapoints*Ndatapoints,dtype=float)
Kernel_Grv = np.zeros((Ndatapoints*Ndatapoints,CX*CY*CZ)).astype('float32')
Kernel_Mag = np.zeros((Ndatapoints*Ndatapoints,CX*CY*CZ)).astype('float32')
Gkernelsp = csr_matrix(Kernel_Grv)
Mkernelsp = csr_matrix(Kernel_Mag)
globals_par = np.zeros((5,2))
globals_xyz = np.zeros((3,2))
AR_bounds = np.zeros((4,2))

Kmin = 6
Kmax = 120
KminAR = 0
KmaxAR = 3
Chain = np.zeros(46+Kmax*4).astype('float32')

NT1 = int(np.floor((Nchain+1)/2))       # number of chains with T=1
dt = 1.2                                # ratio between temperature levels
TempLevels=np.arange(Nchain-NT1,0,-1)   # define Temp Levels
Temp = np.hstack((pow(dt,TempLevels),np.ones(NT1)))

NKEEP = 1000          # dump a binary file to desk every NKEEP records
NMCMC = 100000*NKEEP   #number of random walks
    
if rank == 0:
    
    loaddesk = 0
    
    fpath_loaddesk = os.getcwd()+'/loaddesk'

    daytime = datetime.now().strftime("%Y_%m_%d-%I_%M_%S_%p")
    fpath = os.getcwd()+'//'+daytime
    if os.path.exists(fpath) and os.path.isdir(fpath):
        shutil.rmtree(fpath)
    os.mkdir(fpath)
    
    fpath_Restart = fpath+'/Restart'
    os.mkdir(fpath_Restart)
    
    fpath_PDF = fpath+'/PDF'
    os.mkdir(fpath_PDF)
    
    fpath_bnfiles = fpath+'/BinFiles'
    os.mkdir(fpath_bnfiles) 
    
    
    ikeep = 0 #counter when writing to output files
    ChainKeep = np.zeros((NKEEP,Chain.size)).astype('float32')
    
    ChainAll = np.zeros((Nchain,Chain.size)).astype('float32') # ChainAll keeps the latest Chain of each source, for restart program...
    ChainHistory = np.empty((0,Chain.size)).astype('float32')
    
    if loaddesk == 1:
        ChainAll_loaddesk = np.load(fpath_loaddesk+'//'+'ChainAll.npy') # use this, if the latest result exists
        #ChainAll[:10] = ChainAll[-10:]
        iload = ChainAll_loaddesk.shape[0]-1
        for irow in np.arange(Nchain-1,-1,-1):
            ChainAll[irow,:] = ChainAll_loaddesk[iload,:].copy()
            iload -= 1
            if iload == -1:
                iload = ChainAll_loaddesk.shape[0]-1
                
                

    Gravity_Data = np.loadtxt('GRV_2D_Data.txt').astype('float32')
    Magnetic_Data = np.loadtxt('RTP_2D_Data.txt').astype('float32')

    xs = np.linspace(Gravity_Data[0,0],Gravity_Data[-1,0],Ndatapoints)
    ys = np.flip(np.linspace(Gravity_Data[0,1],Gravity_Data[-1,1], Ndatapoints),0)
    # ys = np.linspace(Gravity_Data[0,1],Gravity_Data[-1,1], Ndatapoints)
    XS,YS = np.meshgrid(xs,ys)
#     GRV_obs = griddata(Gravity_Data[:,0:2], Gravity_Data[:,2], (XS, YS), method='cubic')
#     RTP_obs = griddata(Magnetic_Data[:,0:2], Magnetic_Data[:,2], (XS, YS), method='cubic')
    
    # model space
    Z0 = 0                
    ZEND = 10000          
    Pad_Length = 5000

    xmodel = np.linspace(np.amin(XS)-Pad_Length,np.amax(XS)+Pad_Length,CX)
    ymodel = np.flip(np.linspace(np.amin(YS)-Pad_Length,np.amax(YS)+Pad_Length,CY))
    zmodel = np.linspace(Z0,ZEND,CZ)
    Y, Z, X = np.meshgrid(ymodel, zmodel, xmodel)
    dx = abs(X[0,0,1]-X[0,0,0])
    dy = abs(Y[0,1,0]-Y[0,0,0])
    dz = abs(Z[1,0,0]-Z[0,0,0])
    
    x_min=np.min(X)-dx/2
    x_max=np.max(X)+dx/2
    y_min=np.min(Y)-dy/2
    y_max=np.max(Y)+dy/2
    z_min=np.min(Z)-dz/2
    z_max=np.max(Z)+dz/2
    
    Xn_3D = np.divide(X-x_min,x_max-x_min)
    Yn_3D = np.divide(Y-y_min,y_max-y_min)
    Zn_3D = np.divide(Z-z_min,z_max-z_min)
    
        
    Xn = Xn_3D.flatten()
    Yn = Yn_3D.flatten()
    Zn = Zn_3D.flatten()
    XnYnZn = np.column_stack((Xn,Yn,Zn)).astype('float32')

    #[TrueDensityModel,TrueSUSModel] = Model_Making(Xn_3D,Yn_3D,Zn_3D)
    [TrueDensityModel,TrueSUSModel] = Model_Making_Pyramid(Xn_3D,Yn_3D,Zn_3D,XnYnZn)
    fig = plt.figure()
    plt.imshow(TrueDensityModel[:,30,:])
    plt.show()
    figname = 'True_PMD_Model'
    fig.savefig(fpath_Restart+'/'+figname+'.pdf')
    plt.close(fig)
    
    
#     TrueDensityModel = np.load(fpath_loaddesk+'//'+'TrueDensityModel.npy')
#     TrueSUSModel = np.load(fpath_loaddesk+'//'+'TrueSUSModel.npy')
    
    

    
    I = 90 * (3.14159265359/180)
    D = 0 * (3.14159265359/180)
    Fe = 43314  #(nT)

    for iz in np.arange(CZ):
        c1 = iz*CX*CY
        c2 = (iz+1)*CX*CY
        Kernel_Grv[:,c1:c2] = Gravity_3D_Kernel_Expanded_MD(X[iz,:,:],Y[iz,:,:],Z[iz,:,:],XS,YS,dx,dy,dz)
        Kernel_Mag[:,c1:c2] = Magnetic_3D_Kernel_Expanded_MD(X[iz,:,:],Y[iz,:,:],Z[iz,:,:],XS,YS,dx,dy,dz,I,D,Fe)

    Kernel_Grv = Kernel_Grv*1e8
    Kernel_Grv = Kernel_Grv.astype('float32')
    Kernel_Mag = Kernel_Mag.astype('float32')
    
    Kernel_Grv_str = fpath_loaddesk +'//'+'Kernel_Grv'+'.npy'
    np.save(Kernel_Grv_str, Kernel_Grv)
    Kernel_Mag_str = fpath_loaddesk +'//'+'Kernel_Mag'+'.npy'
    np.save(Kernel_Mag_str, Kernel_Mag)
    
    [Gkernelsp, Mkernelsp] = Kernel_compressor(Ndatapoints,CX,CY,CZ,Kernel_Grv,Kernel_Mag,fpath_loaddesk)
    
    Gkernelsp_str = fpath_loaddesk +'//'+'Gkernelsp.npz'
    scipy.sparse.save_npz(Gkernelsp_str, Gkernelsp)
    Mkernelsp_str = fpath_loaddesk +'//'+'Mkernelsp.npz'
    scipy.sparse.save_npz(Mkernelsp_str, Mkernelsp)

    Kernel_Grv = np.load(fpath_loaddesk+'//'+'Kernel_Grv.npy') # use this, if the latest result exists 
    Kernel_Mag = np.load(fpath_loaddesk+'//'+'Kernel_Mag.npy') # use this, if the latest result exists
    
    Gkernelsp_str = fpath_loaddesk +'//'+'Gkernelsp.npz'
    Gkernelsp = scipy.sparse.load_npz(Gkernelsp_str)

    Mkernelsp_str = fpath_loaddesk +'//'+'Mkernelsp.npz'
    Mkernelsp = scipy.sparse.load_npz(Mkernelsp_str)


    TrueDensityModelVec = TrueDensityModel.flatten()
    TrueSUSModelVec = TrueSUSModel.flatten()

    dg_true = Kernel_Grv @ TrueDensityModelVec
    dT_true = Kernel_Mag @ TrueSUSModelVec

    # Adding noise
    
    AR_original_g = np.array([[0., 0.6, -0.5, 0],[0.4, -0.2, 0., 0.],[0,0,0,0],[0,0,0,0]])
    AR_original_T = np.zeros((4,4))
    

    noise_g_level = 0.04
    sigma_g_original = noise_g_level*max(abs(dg_true))
    uncorr_noise_g_original = sigma_g_original*np.random.randn(Ndatapoints,Ndatapoints).astype('float32')
    
    kcol = AR_original_g.shape[1]-1
    krow = AR_original_g.shape[0]-1
    u_padd = np.pad(uncorr_noise_g_original, [(krow, krow),(kcol,kcol)]) 
    
    r_padd = np.zeros(u_padd.shape)  
    
    AR_original_g_flip = np.flipud(np.fliplr(AR_original_g))    # Flip the kernel
    
    for m in np.arange(krow,r_padd.shape[0]):
        for n in np.arange(kcol,r_padd.shape[1]):
            r_padd[m,n] = (AR_original_g_flip*r_padd[m-krow:m+1,n-kcol:n+1]).sum() + u_padd[m,n]
        
    corr_noise_g_original = r_padd[krow:-krow,kcol:-kcol].copy()    
    
    dg_obs = dg_true + (corr_noise_g_original.flatten())
    
    
#     if loaddesk == 1:
#         dg_obs = np.load(fpath_loaddesk+'//'+'dg_obs.npy') # use this, if the latest result exists 

    noise_T_level = 0.06
    sigma_T_original = noise_T_level*max(abs(dT_true))
    noise_T_original = sigma_T_original*np.random.randn(Ndatapoints*Ndatapoints)
    dT_obs = dT_true + noise_T_original
    
#     if loaddesk == 1:
#         dT_obs = np.load(fpath_loaddesk+'//'+'dT_obs.npy') # use this, if the latest result exists
    
    
    rho_salt_min = -0.4
    rho_salt_max = -0.2
    rho_base_min = 0.2
    rho_base_max = 0.4
    sus_base_max = rho_base_max/50
    zn_min = 0./z_max
    
    globals_par = np.matrix([[Kmin, Kmax], [rho_salt_min, rho_salt_max],
                             [rho_base_min, rho_base_max], [KminAR, KmaxAR], [zn_min, sus_base_max]])
    globals_xyz = np.matrix([[x_min, x_max], [y_min, y_max], [z_min, z_max]])
    
    AR_bounds =  np.matrix([[0., 0.], [-0.85, 0.9], [-0.85, 0.1], [-0.25, 0.25]]) # AR0, AR1, AR2, AR3
    
    for ichain in np.arange(1,Nchain+1): #sources (ranks) 1,2,...,Nchain
        
        Chain[:] = 0.
        
        #ind = np.argsort(ChainAll[:,0])[::-1]
        #Chain_MaxL = ChainAll[ind[0]].copy()
        Chain_MaxL = ChainAll[ichain-1,:]
        
        if ichain<=9:
            loaddesk = 0
        else:
            loaddesk = 1
        
        Chain = Initializing(Chain,XnYnZn,globals_par,Gkernelsp,Mkernelsp,dg_obs,dT_obs,Chain_MaxL,loaddesk).copy()
        comm.Send(Chain, dest=ichain, tag=ichain)   
        
        
    # Save important arrays for posterior process
    ChainAll = np.zeros((Nchain,Chain.size)).astype('float32')
    ChainAll_str = fpath_Restart+'//'+'ChainAll.npy'
    np.save(ChainAll_str, ChainAll)
    
    Kernel_Grv_str = fpath_Restart+'//'+'Kernel_Grv.npy'
    np.save(Kernel_Grv_str, Kernel_Grv)
    
    Kernel_Mag_str = fpath_Restart+'//'+'Kernel_Mag.npy'
    np.save(Kernel_Mag_str, Kernel_Mag)
    
    dg_obs_str = fpath_Restart+'//'+'dg_obs.npy'
    np.save(dg_obs_str, dg_obs)
    
    dT_obs_str = fpath_Restart+'//'+'dT_obs.npy'
    np.save(dT_obs_str, dT_obs)
    
    XnYnZn_str = fpath_Restart+'//'+'XnYnZn.npy'
    np.save(XnYnZn_str, XnYnZn)
    
    globals_par_str = fpath_Restart+'//'+'globals_par.npy'
    np.save(globals_par_str, globals_par)
    
    globals_xyz_str = fpath_Restart+'//'+'globals_xyz.npy'
    np.save(globals_xyz_str, globals_xyz)
    
    AR_bounds_str = fpath_Restart+'//'+'AR_bounds.npy'
    np.save(AR_bounds_str, AR_bounds)
    
    TrueDensityModel_str = fpath_Restart+'//'+'TrueDensityModel.npy'
    np.save(TrueDensityModel_str, TrueDensityModel)
    
    TrueSUSModel_str = fpath_Restart+'//'+'TrueSUSModel.npy'
    np.save(TrueSUSModel_str, TrueSUSModel)
    
    AR_original_g_str = fpath_Restart+'//'+'AR_original_g.npy'
    np.save(AR_original_g_str, AR_original_g)
    
    AR_original_T_str = fpath_Restart+'//'+'AR_original_T.npy'
    np.save(AR_original_T_str, AR_original_T)
    
    ChainHistory_str = fpath_Restart+'//'+'ChainHistory.npy'
    #np.save(ChainHistory_str, ChainHistory)
    
    Gkernelsp_str = fpath_Restart +'//'+'Gkernelsp.npz'
    scipy.sparse.save_npz(Gkernelsp_str, Gkernelsp)
    
    Mkernelsp_str = fpath_Restart +'//'+'Mkernelsp.npz'
    scipy.sparse.save_npz(Mkernelsp_str, Mkernelsp)
          

if rank>0:
    comm.Recv(Chain, source=0, tag=rank)

XnYnZn = comm.bcast(XnYnZn, root=0)    
dg_obs = comm.bcast(dg_obs, root=0)
dT_obs = comm.bcast(dT_obs, root=0)
# Kernel_Grv = comm.bcast(Kernel_Grv, root=0)
# Kernel_Mag = comm.bcast(Kernel_Mag, root=0)
Kernel_Grv = comm.bcast(Gkernelsp, root=0)
Kernel_Mag = comm.bcast(Mkernelsp, root=0)
globals_par = comm.bcast(globals_par, root=0)
globals_xyz = comm.bcast(globals_xyz, root=0)
AR_bounds = comm.bcast(AR_bounds, root=0)

comm.Barrier()



if rank > 0: 
    T = Temp[rank-1]
else:
    T = 1

#print("rank: ",rank, ", Chain: ",Chain[0:8])
#sys.stdout.flush()

#### Inversion

########################################################################
## workers

if rank > 0:
    
    bk_Nodes = 0.3 # probability from M(k) to M(k+1)
    bk_AR = np.array([1., 0.3, 0.3, 0.3])   # probability from M(k) to M(k+1)

    for imcmc in np.arange(1,NMCMC+1):  # 1 to NMCMC
        LogLc = Chain[0].copy()
        [xc, yc, zc, rhoc, k_ARc, XYZLinec, ARgc, ARTc] = Chain2xyz(Chain)
        
        step = select_step(globals_par[0,0],globals_par[0,1],np.size(xc),bk_Nodes)

        if step==91:
            [LogLc,xc,yc,zc,rhoc] = birth(XnYnZn,globals_par,LogLc,XYZLinec,xc,yc,zc,rhoc,
                                                  T,Kernel_Grv,Kernel_Mag,dg_obs,dT_obs,ARgc,ARTc)
        elif step==92:
            [LogLc,xc,yc,zc,rhoc] = death(XnYnZn,LogLc,xc,yc,zc,rhoc,
                                                   T,Kernel_Grv,Kernel_Mag,dg_obs,dT_obs,ARgc,ARTc)
        else:
            [LogLc,XYZLinec,xc,yc,zc,rhoc] = move(XnYnZn,globals_par,LogLc,XYZLinec,xc,yc,zc,rhoc,
                                                               T,Kernel_Grv,Kernel_Mag,dg_obs,dT_obs,ARgc,ARTc)
        
        if imcmc>2000 and imcmc % 10 == 0:
            
            iAR = np.random.randint(6)
            
            if iAR == 0: ### kx_g
                [LogLc,ARgc,k_ARc] = ARx_g(k_ARc,XnYnZn,globals_par,AR_bounds,LogLc,xc,yc,zc,rhoc,
                                           ARgc,ARTc,T,Kernel_Grv,Kernel_Mag,dg_obs,dT_obs,bk_AR)
                
            
            elif iAR == 1: ### ky_g
                [LogLc,ARgc,k_ARc] = ARy_g(k_ARc,XnYnZn,globals_par,AR_bounds,LogLc,xc,yc,zc,rhoc,
                                           ARgc,ARTc,T,Kernel_Grv,Kernel_Mag,dg_obs,dT_obs,bk_AR)
            
            elif iAR == 2: ### kxy_g
                [LogLc,ARgc,k_ARc] = ARxy_g(k_ARc,XnYnZn,globals_par,AR_bounds,LogLc,xc,yc,zc,rhoc,
                                           ARgc,ARTc,T,Kernel_Grv,Kernel_Mag,dg_obs,dT_obs,bk_AR)
            
            elif iAR == 3: ### kx_T
                [LogLc,ARTc,k_ARc] = ARx_T(k_ARc,XnYnZn,globals_par,AR_bounds,LogLc,xc,yc,zc,rhoc,
                                           ARgc,ARTc,T,Kernel_Grv,Kernel_Mag,dg_obs,dT_obs,bk_AR)
            
            elif iAR == 4: ### ky_T
                [LogLc,ARTc,k_ARc] = ARy_T(k_ARc,XnYnZn,globals_par,AR_bounds,LogLc,xc,yc,zc,rhoc,
                                           ARgc,ARTc,T,Kernel_Grv,Kernel_Mag,dg_obs,dT_obs,bk_AR)
                
            elif iAR == 5: ### kxy_T    
                [LogLc,ARTc,k_ARc] = ARxy_T(k_ARc,XnYnZn,globals_par,AR_bounds,LogLc,xc,yc,zc,rhoc,
                                           ARgc,ARTc,T,Kernel_Grv,Kernel_Mag,dg_obs,dT_obs,bk_AR)
        
        
        
        Chain[:] = 0.
        Chain[0] = LogLc
        Chain[1] = np.size(xc)
        Chain[2:8] = k_ARc.copy()
        Chain[8:14] = XYZLinec.copy()
        Chain[14:30] = ARgc.copy()  
        Chain[30:46] = ARTc.copy() 
        Chain[46:46+np.size(xc)*4] = np.concatenate((xc,yc,zc,rhoc)).copy()
        
        
        
#         if imcmc % 100 ==0:
#             print("rank: ",rank,", T: ","%.2f" %T, ", Iteration: ",imcmc, ", LogL: ","%.2f" %Chain[0], ", k: ",Chain[1])
#             sys.stdout.flush()
        
        ## Sending model to Master 
        comm.Send(Chain, dest=0, tag=rank)
        
        ## Receiving back from Master
        Chain[:] = 0.
        comm.Recv(Chain, source=0, tag=MPI.ANY_TAG)       
        
        
## MASTER rank == 0
else:
    ifile = 0
    iraw = 0
    for imcmc in np.arange(1,NMCMC+1):  # 1 to NMCMC
        
        Chain[:] = 0.
        comm.Recv(Chain, source=MPI.ANY_SOURCE, tag=MPI.ANY_TAG,status=status)
        Chain_p = Chain.copy()
        source_p = status.source
        Tp = Temp[source_p-1].copy()
        LogLp = Chain_p[0].copy()  
        
        Chain[:] = 0.
        comm.Recv(Chain, source=MPI.ANY_SOURCE, tag=MPI.ANY_TAG,status=status)
        Chain_q = Chain.copy()
        source_q = status.source
        Tq = Temp[source_q-1].copy()
        LogLq = Chain_q[0].copy()     
                
        if Tp != Tq:
            Prob=np.exp(((1/Tp)-(1/Tq))*(LogLq-LogLp))
            
            if np.random.rand()<=Prob:
                Chain_0 = Chain_p.copy()
                Chain_p = Chain_q.copy()
                Chain_q = Chain_0.copy()
        
        Chain = Chain_p.copy()
        comm.Send(Chain, dest=source_p, tag=rank)
        Chain = Chain_q.copy()
        comm.Send(Chain, dest=source_q, tag=rank)
        
        ChainAll[source_p-1,:] = Chain_p.copy()
        ChainAll[source_q-1,:] = Chain_q.copy()
        
        ## save to binary format
        
        if Tp == 1:  
            ChainKeep[ikeep,:] = Chain_p.copy()
            ikeep += 1
        if (Tq == 1) & (ikeep<NKEEP):
            ChainKeep[ikeep,:] = Chain_q.copy()
            ikeep += 1
        
        
        if ikeep == NKEEP:
            
            if np.sum(ChainAll[-10:,2:8]) == 0:
                imcmc_str = 'RAW_'+str(iraw)
                iraw += 1
            else:
                imcmc_str = 'STN_'+str(ifile)
                ifile += 1
                
            np.save(fpath_bnfiles+'//'+imcmc_str+'.npy', ChainKeep)
            np.save(ChainAll_str, ChainAll)
            Imshow_residuals(ChainAll,Ndatapoints,Kernel_Grv,Kernel_Mag,dg_obs,dT_obs,XnYnZn,fpath_PDF,'Residuals_',imcmc_str)
            
            ikeep = 0
            ChainKeep[:,:] = 0.
            
        