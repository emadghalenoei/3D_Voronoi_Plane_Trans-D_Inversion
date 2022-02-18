from mpi4py import MPI
import os
import sys
import shutil
import numpy as np
from Chain2xyz import Chain2xyz
import matplotlib.pyplot as plt
import time
from mean_online import mean_online
from std_online import std_online
from Log_Likelihood import Log_Likelihood
import scipy.sparse


comm = MPI.COMM_WORLD
rank = comm.Get_rank()
status = MPI.Status()
Nworkers = comm.Get_size()-1 # No. MCMC chains or MPI threads


file_name = '2021_08_09-09_47_27_PM'
    
fpath_Restart = os.getcwd()+'//'+file_name+'//'+'Restart'

fpath_output = os.getcwd()+'//'+file_name+'//'+'Output'
if rank == 0:
    if os.path.exists(fpath_output) and os.path.isdir(fpath_output):
        shutil.rmtree(fpath_output)
    os.mkdir(fpath_output)

dg_obs = np.load(fpath_Restart+'//'+'dg_obs.npy')
dT_obs = np.load(fpath_Restart+'//'+'dT_obs.npy')
XnYnZn = np.load(fpath_Restart+'//'+'XnYnZn.npy')
# Kernel_Grv = np.load(fpath_Restart+'//'+'Kernel_Grv.npy')
# Kernel_Mag = np.load(fpath_Restart+'//'+'Kernel_Mag.npy')
Kernel_Grv = scipy.sparse.load_npz(fpath_Restart +'//'+'Gkernelsp.npz')
Kernel_Mag = scipy.sparse.load_npz(fpath_Restart +'//'+'Mkernelsp.npz')
globals_par = np.load(fpath_Restart+'//'+'globals_par.npy')
globals_xyz = np.load(fpath_Restart+'//'+'globals_xyz.npy')
# TrueDensityModel = np.load(fpath_Restart+'//'+'TrueDensityModel.npy')
# TrueSUSModel = np.load(fpath_Restart+'//'+'TrueSUSModel.npy')



Kmin = int(globals_par[0,0])
Kmax = int(globals_par[0,1])
rho_salt_min = globals_par[1,0]
rho_salt_max = globals_par[1,1]
rho_base_min = globals_par[2,0]
rho_base_max = globals_par[2,1]
zn_min = globals_par[4,0]
sus_base_max = globals_par[4,1]
KminAR = int(globals_par[3,0])
KmaxAR = int(globals_par[3,1])

x_min = globals_xyz[0,0]
x_max = globals_xyz[0,1]
y_min = globals_xyz[1,0]
y_max = globals_xyz[1,1]
z_min = globals_xyz[2,0]
z_max = globals_xyz[2,1]

CX = 64
CY = 64
CZ = 64
Ndatapoints = np.size(dg_obs)

Xn_Grid = np.reshape(XnYnZn[:,0],(CZ,CY,CX)).copy()
Yn_Grid = np.reshape(XnYnZn[:,1],(CZ,CY,CX)).copy()
Zn_Grid = np.reshape(XnYnZn[:,2],(CZ,CY,CX)).copy()


# Ncol = 8+Kmax*4
# ndelete = 47000 # must be dividable by Nworkers (e.x. 10, 100, ...)
# Nchain = 1000   # numder of chains(rows) per each worker

# with open(ChainHistory_str, 'rb') as f:
    
#     Chainsplit = np.fromfile(f,count = ndelete*Ncol, dtype='float32')

#     for irank in np.arange(0,rank+1):    
#         #f.seek(0,1)
#         Chainsplit = np.fromfile(f,count = Nchain*Ncol, dtype='float32').reshape(Nchain,Ncol)

# print('rank: ', rank, ', number of chains: ', Chainsplit.shape[0])
# sys.stdout.flush()

Chainsplit = np.empty((0,46+Kmax*4)).astype('float32')
ndelete = 272
Nfile = 5
fpath_bin = os.getcwd()+'//'+file_name+'//'+'BinFiles'

for ifile in np.arange(0,Nfile):
    filestr = 'STN_'+str(ndelete + ifile + rank*Nfile)+'.npy'
    Chainsp = np.load(fpath_bin+'//'+filestr)
    Chainsplit = np.append(Chainsplit,Chainsp, axis=0)
    
    
    if ifile % 1 == 0:
        print('rank: ', rank, ', file name: ', filestr)
        sys.stdout.flush()


print('rank: ', rank, ', number of chains: ', Chainsplit.shape[0])
sys.stdout.flush()

Nchain = Chainsplit.shape[0]
        
LogLkeep = np.zeros(Nchain).copy()
Nnode = np.zeros(Nchain).astype(int)
sigmakeep_g = np.zeros(Nchain).copy()
sigmakeep_T = np.zeros(Nchain).copy()
rho_keep = np.zeros(1).copy()

grid_g_mean = np.zeros(Xn_Grid.shape).copy()
grid_g_std  = np.zeros(Xn_Grid.shape).copy()

grid_T_mean = np.zeros(Xn_Grid.shape).copy()
grid_T_std  = np.zeros(Xn_Grid.shape).copy()

data_g_mean = np.zeros((dg_obs.size,1))
data_g_std  = np.zeros((dg_obs.size))

data_T_mean = np.zeros((dg_obs.size,1))
data_T_std  = np.zeros((dg_obs.size))

k_AR_keep = np.zeros((Nchain,6)).astype(int)

ARx_g_keep = np.zeros((Nchain,KmaxAR)).copy()
ARy_g_keep = np.zeros((Nchain,KmaxAR)).copy()
ARxy_g_keep = np.zeros((Nchain,KmaxAR)).copy()
ARx_T_keep = np.zeros((Nchain,KmaxAR)).copy()
ARy_T_keep = np.zeros((Nchain,KmaxAR)).copy()
ARxy_T_keep = np.zeros((Nchain,KmaxAR)).copy()

for ichain in np.arange(Nchain):
    
        Chain = Chainsplit[ichain,:].copy()
        [x, y, z, rho, k_AR, XYZLine, ARg, ART]= Chain2xyz(Chain).copy()
        
        ARg_2D = ARg.reshape((4,4))
        ART_2D = ART.reshape((4,4))
        
        k_AR_keep[ichain,:] = k_AR.copy()
        ARx_g_keep[ichain,:] = ARg_2D[0,1:4].copy()
        ARy_g_keep[ichain,:] = ARg_2D[1:4,0].copy()
        ARxy_g_keep[ichain,:] = np.array([ARg_2D[1,1],ARg_2D[2,2],ARg_2D[3,3]]).copy()
        ARx_T_keep[ichain,:] = ART_2D[0,1:4].copy()
        ARy_T_keep[ichain,:] = ART_2D[1:4,0].copy()
        ARxy_T_keep[ichain,:] = np.array([ART_2D[1,1],ART_2D[2,2],ART_2D[3,3]]).copy()
        
        [LogL, model_vec_g, model_vec_T, rg, rT, sigma_g, sigma_T, uncor_g, uncor_T] = Log_Likelihood(Kernel_Grv,Kernel_Mag,dg_obs,dT_obs,x,y,z,rho,ARg_2D,ART_2D,XnYnZn)
        
        
        LogLkeep[ichain] = Chain[0].copy()
        Nnode[ichain] = Chain[1].copy()       
        sigmakeep_g[ichain] = sigma_g
        sigmakeep_T[ichain] = sigma_T
        
        rho_keep = np.append(rho_keep,rho[rho != 0])
        
        gridi_g = model_vec_g.reshape((CZ,CY,CX)).copy()
        gridi_T = model_vec_T.reshape((CZ,CY,CX)).copy() 
                
        grid_g_mean = mean_online(ichain+1,gridi_g,grid_g_mean)
        grid_g_std = std_online(ichain+1,grid_g_std,gridi_g,grid_g_mean)
        
        grid_T_mean = mean_online(ichain+1,gridi_T,grid_T_mean)
        grid_T_std = std_online(ichain+1,grid_T_std,gridi_T,grid_T_mean)
        
        dg_pre = dg_obs - rg
        dT_pre = dT_obs - rT
        
        data_g_mean = mean_online(ichain+1,dg_pre,data_g_mean)
        data_g_std = std_online(ichain+1,data_g_std,dg_pre,data_g_mean)
        
        data_T_mean = mean_online(ichain+1,dT_pre,data_T_mean)
        data_T_std = std_online(ichain+1,data_T_std,dT_pre,data_T_mean) 
        
        
        
        
        if ichain % 100 == 0:
            print('rank: ', rank, ', ichain: ', ichain, ' from ', Nchain)
            sys.stdout.flush()

        
grid_g_mean_ToT = comm.reduce(grid_g_mean, op=MPI.SUM, root=0)
grid_g_std_ToT = comm.reduce(grid_g_std, op=MPI.SUM, root=0)

grid_T_mean_ToT = comm.reduce(grid_T_mean, op=MPI.SUM, root=0)
grid_T_std_ToT = comm.reduce(grid_T_std, op=MPI.SUM, root=0)

data_g_mean_ToT = comm.reduce(data_g_mean, op=MPI.SUM, root=0)
data_g_std_ToT = comm.reduce(data_g_std, op=MPI.SUM, root=0)

data_T_mean_ToT = comm.reduce(data_T_mean, op=MPI.SUM, root=0)
data_T_std_ToT = comm.reduce(data_T_std, op=MPI.SUM, root=0)


# gather all local arrays on process root, will return a list of numpy arrays
LogLkeep_TOT = comm.gather(LogLkeep, root=0)
Nnode_TOT = comm.gather(Nnode, root=0)
sigmakeep_g_TOT = comm.gather(sigmakeep_g, root=0)
sigmakeep_T_TOT = comm.gather(sigmakeep_T, root=0)
rho_keep = np.delete(rho_keep, 0).copy()
rho_keep_TOT = comm.gather(rho_keep, root=0)
k_AR_keep_TOT = comm.gather(k_AR_keep, root=0)
ARx_g_keep_TOT = comm.gather(ARx_g_keep, root=0)
ARy_g_keep_TOT = comm.gather(ARy_g_keep, root=0)
ARxy_g_keep_TOT = comm.gather(ARxy_g_keep, root=0)
ARx_T_keep_TOT = comm.gather(ARx_T_keep, root=0)
ARy_T_keep_TOT = comm.gather(ARy_T_keep, root=0)
ARxy_T_keep_TOT = comm.gather(ARxy_T_keep, root=0)
################################################################################################################
### save to desk

if rank == 0:
    
    # turn the list of arrays into a single array
    LogLkeep_TOT = np.concatenate(LogLkeep_TOT)
    Nnode_TOT = np.concatenate(Nnode_TOT)
    sigmakeep_g_TOT = np.concatenate(sigmakeep_g_TOT)
    sigmakeep_T_TOT = np.concatenate(sigmakeep_T_TOT)
    rho_keep_TOT = np.concatenate(rho_keep_TOT)
    k_AR_keep_TOT = np.concatenate(k_AR_keep_TOT)
    ARx_g_keep_TOT = np.concatenate(ARx_g_keep_TOT)
    ARy_g_keep_TOT = np.concatenate(ARy_g_keep_TOT)
    ARxy_g_keep_TOT = np.concatenate(ARxy_g_keep_TOT)
    ARx_T_keep_TOT = np.concatenate(ARx_T_keep_TOT)
    ARy_T_keep_TOT = np.concatenate(ARy_T_keep_TOT)
    ARxy_T_keep_TOT = np.concatenate(ARxy_T_keep_TOT)
    
    
    PMD_g = grid_g_mean_ToT/(Nworkers+1)
    PMD_g_str = fpath_output+'//'+'PMD_g.npy'
    np.save(PMD_g_str, PMD_g)
    
    STD_g = grid_g_std_ToT/(Nworkers+1)
    STD_g_str = fpath_output+'//'+'STD_g.npy'
    np.save(STD_g_str, STD_g)
    
    PMD_T = grid_T_mean_ToT/(Nworkers+1)
    PMD_T_str = fpath_output+'//'+'PMD_T.npy'
    np.save(PMD_T_str, PMD_T)
    
    STD_T = grid_T_std_ToT/(Nworkers+1)
    STD_T_str = fpath_output+'//'+'STD_T.npy'
    np.save(STD_T_str, STD_T)
    
    PMD_data_g = data_g_mean_ToT/(Nworkers+1)
    PMD_data_g_str = fpath_output+'//'+'PMD_data_g.npy'
    np.save(PMD_data_g_str, PMD_data_g)
    
    STD_data_g = data_g_std_ToT/(Nworkers+1)
    STD_data_g_str = fpath_output+'//'+'STD_data_g.npy'
    np.save(STD_data_g_str, STD_data_g)
    
    PMD_data_T = data_T_mean_ToT/(Nworkers+1)
    PMD_data_T_str = fpath_output+'//'+'PMD_data_T.npy'
    np.save(PMD_data_T_str, PMD_data_T)
    
    STD_data_T = data_T_std_ToT/(Nworkers+1)
    STD_data_T_str = fpath_output+'//'+'STD_data_T.npy'
    np.save(STD_data_T_str, STD_data_T)
    
    
    LogLkeep_str = fpath_output+'//'+'LogLkeep.npy'
    np.save(LogLkeep_str, LogLkeep_TOT)

    Nnode_str = fpath_output+'//'+'Nnode.npy'
    np.save(Nnode_str, Nnode_TOT)
    
    sigma_g_str = fpath_output+'//'+'sigma_g.npy'
    np.save(sigma_g_str, sigmakeep_g_TOT)
    
    sigma_T_str = fpath_output+'//'+'sigma_T.npy'
    np.save(sigma_T_str, sigmakeep_T_TOT)
    
    rho_keep_str = fpath_output+'//'+'rho_keep.npy'
    np.save(rho_keep_str, rho_keep_TOT)
    
    k_AR_keep_str = fpath_output+'//'+'k_AR_keep.npy'
    np.save(k_AR_keep_str, k_AR_keep_TOT)
    
    ARx_g_keep_str = fpath_output+'//'+'ARx_g_keep.npy'
    np.save(ARx_g_keep_str, ARx_g_keep_TOT)
    
    ARy_g_keep_str = fpath_output+'//'+'ARy_g_keep.npy'
    np.save(ARy_g_keep_str, ARy_g_keep_TOT)
    
    ARxy_g_keep_str = fpath_output+'//'+'ARxy_g_keep.npy'
    np.save(ARxy_g_keep_str, ARxy_g_keep_TOT)
    
    ARx_T_keep_str = fpath_output+'//'+'ARx_T_keep.npy'
    np.save(ARx_T_keep_str, ARx_T_keep_TOT)
    
    ARy_T_keep_str = fpath_output+'//'+'ARy_T_keep.npy'
    np.save(ARy_T_keep_str, ARy_T_keep_TOT)
    
    ARxy_T_keep_str = fpath_output+'//'+'ARxy_T_keep.npy'
    np.save(ARxy_T_keep_str, ARxy_T_keep_TOT)    
    

    print('The End')
    sys.stdout.flush()
    
MPI.Finalize
 