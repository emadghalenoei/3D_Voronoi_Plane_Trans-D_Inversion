import os
import sys
import shutil
import numpy as np
import matplotlib.pyplot as plt
import matplotlib        as mpl
import matplotlib.cm as cm
from scipy.stats import norm
from matplotlib import rc
from Chain2xyz import Chain2xyz
from scipy.spatial import Voronoi, voronoi_plot_2d
import scipy.sparse
from Log_Likelihood import Log_Likelihood
from scipy import signal
from mpl_toolkits.mplot3d import Axes3D
# import pyvista as pv
#######################################################################

file_name = '2021_08_09-09_47_27_PM'

fpath_Restart = os.getcwd()+'//'+file_name+'//'+'Restart'

dg_obs = np.load(fpath_Restart+'//'+'dg_obs.npy')
dT_obs = np.load(fpath_Restart+'//'+'dT_obs.npy')
XnYnZn = np.load(fpath_Restart+'//'+'XnYnZn.npy')
Kernel_Grv = scipy.sparse.load_npz(fpath_Restart +'//'+'Gkernelsp.npz')
Kernel_Mag = scipy.sparse.load_npz(fpath_Restart +'//'+'Mkernelsp.npz')
TrueDensityModel = np.load(fpath_Restart+'//'+'TrueDensityModel.npy')
TrueSUSModel = np.load(fpath_Restart+'//'+'TrueSUSModel.npy')
globals_par = np.load(fpath_Restart+'//'+'globals_par.npy')
globals_xyz = np.load(fpath_Restart+'//'+'globals_xyz.npy')
AR_parameters_original_g = np.load(fpath_Restart+'//'+'AR_original_g.npy')
AR_parameters_original_T = np.load(fpath_Restart+'//'+'AR_original_T.npy')
AR_bounds = np.load(fpath_Restart+'//'+'AR_bounds.npy')
ChainAll = np.load(fpath_Restart+'//'+'ChainAll.npy') 

WhiteBlueGreenYellowRed = np.loadtxt(os.getcwd()+'//'+'WhiteBlueGreenYellowRed.txt')
WBGR = np.ones((WhiteBlueGreenYellowRed.shape[0],4))
WBGR[:,:-1] = WhiteBlueGreenYellowRed.copy()
WBGR = mpl.colors.ListedColormap(WBGR, name='WBGR', N=WBGR.shape[0])
    
########################################################################
fpath_output = os.getcwd()+'//'+file_name+'//'+'Output'

LogLkeep = np.load(fpath_output+'//'+'LogLkeep.npy')
Nnode = np.load(fpath_output+'//'+'Nnode.npy')
PMD_g = np.load(fpath_output+'//'+'PMD_g.npy')
STD_g = np.load(fpath_output+'//'+'STD_g.npy')
PMD_T = np.load(fpath_output+'//'+'PMD_T.npy')
STD_T = np.load(fpath_output+'//'+'STD_T.npy')
rho_keep = np.load(fpath_output+'//'+'rho_keep.npy')
PMD_data_g = np.load(fpath_output+'//'+'PMD_data_g.npy')
STD_data_g = np.load(fpath_output+'//'+'STD_data_g.npy')
PMD_data_T = np.load(fpath_output+'//'+'PMD_data_T.npy')
STD_data_T = np.load(fpath_output+'//'+'STD_data_T.npy')
sigma_g = np.load(fpath_output+'//'+'sigma_g.npy')
sigma_T = np.load(fpath_output+'//'+'sigma_T.npy')
k_AR_keep = np.load(fpath_output+'//'+'k_AR_keep.npy')
ARx_g_keep = np.load(fpath_output+'//'+'ARx_g_keep.npy')
ARy_g_keep = np.load(fpath_output+'//'+'ARy_g_keep.npy')
ARxy_g_keep = np.load(fpath_output+'//'+'ARxy_g_keep.npy')
ARx_T_keep = np.load(fpath_output+'//'+'ARx_T_keep.npy')
ARy_T_keep = np.load(fpath_output+'//'+'ARy_T_keep.npy')
ARxy_T_keep = np.load(fpath_output+'//'+'ARxy_T_keep.npy')



Ndata = int(np.sqrt(PMD_data_g.size))


CI_g_Low  = PMD_g - 1.96 * STD_g
CI_g_High = PMD_g + 1.96 * STD_g
CI_g_Width = abs(CI_g_High-CI_g_Low)

CI_T_Low  = PMD_T - 1.96 * STD_T
CI_T_High = PMD_T + 1.96 * STD_T
CI_T_Width = abs(CI_T_High-CI_T_Low)

data_g_error = 1.96 * STD_data_g
data_T_error = 1.96 * STD_data_T

Gravity_Data = np.loadtxt('GRV_2D_Data.txt').astype('float32')
Magnetic_Data = np.loadtxt('RTP_2D_Data.txt').astype('float32')
xs = np.linspace(Gravity_Data[0,0],Gravity_Data[-1,0],Ndata)
ys = np.flip(np.linspace(Gravity_Data[0,1],Gravity_Data[-1,1], Ndata),0)

###############################################################################
fpath_plots = os.getcwd()+'//'+file_name+'//'+'Plots'
if os.path.exists(fpath_plots) and os.path.isdir(fpath_plots):
    shutil.rmtree(fpath_plots)
os.mkdir(fpath_plots)

##############################################################################

Kmin = int(globals_par[0,0])
Kmax = int(globals_par[0,1])
rho_salt_min = globals_par[1,0]
rho_salt_max = globals_par[1,1]
rho_base_min = globals_par[2,0]
rho_base_max = globals_par[2,1]

x_min = globals_xyz[0,0]
x_max = globals_xyz[0,1]
y_min = globals_xyz[1,0]
y_max = globals_xyz[1,1]
z_min = globals_xyz[2,0]
z_max = globals_xyz[2,1]

xs_plot = (xs - np.amin(xs))/1000.
ys_plot = (ys - np.amin(ys))/1000.

x1 = (x_min-np.amin(xs))/1000
x2 = (x_max-np.amin(xs))/1000
y1 = (y_min-np.amin(ys))/1000
y2 = (y_max-np.amin(ys))/1000
z1 = z_min/1000
z2 = z_max/1000

###############################################################################
### Plot LogL
fig, axe = plt.subplots(2, 1)
plt.rc('font', weight='normal')
plt.rc('xtick', labelsize=10)
plt.rc('ytick', labelsize=10)

axe[0].plot(LogLkeep, 'k-',linewidth=2) #row=0, col=0
axe[0].set_ylabel('Log Likelihood',fontweight="normal", fontsize = 10)
axe[0].set_xlabel('(a) rjMCMC STEP',fontweight="normal", fontsize = 10)

axe[0].xaxis.set_label_position("top")
axe[0].xaxis.tick_top()
# axe[0].get_xaxis().get_major_formatter().set_scientific(False)
plt.show()

axe[1].hist(Nnode, 13, density=True, color='0.5') 
axe[1].set_ylabel('pdf',fontweight="normal", fontsize = 10)
axe[1].set_xlabel('Number of Nodes',fontweight="normal", fontsize = 10)

# axe[1].yaxis.set_label_position("right")
# axe[1].yaxis.tick_right()
plt.show()

figname = 'LogL_NNode'
fig.savefig(fpath_plots+'/'+figname+'.pdf')
plt.close(fig)    # close the figure window


#########################################################################################
# Plot Sigma
fig, axe = plt.subplots(2, 1)

axe[0].plot(sigma_g, 'k-',linewidth=2) #row=0, col=0
axe[0].set(xlabel='rjMCMC STEP', ylabel='STD of gravity residuals (mGal)')
plt.show()

axe[1].plot(sigma_T, 'k-',linewidth=2) #row=1, col=0
axe[1].set(xlabel='rjMCMC STEP', ylabel='STD of magnetic residuals (nT)')
axe[1].yaxis.set_label_position("right")
axe[1].yaxis.tick_right()
plt.show()
figname = 'Sigma'
fig.savefig(fpath_plots+'/'+figname+'.pdf')
plt.close(fig)    # close the figure window

#########################################################################################
### Plot rho

rho_salt = rho_keep[rho_keep < 0]
rho_base = rho_keep[rho_keep > 0]

fig, axe = plt.subplots(2, 1)
plt.rc('font', weight='normal')
plt.rc('xtick', labelsize=10)
plt.rc('ytick', labelsize=10)

axe[0].hist(rho_salt, 8, density=True, color='0.5') 
axe[0].set_ylabel('pdf',fontweight="normal", fontsize = 10)
axe[0].set_xlabel('(a) density contrast of salt',fontweight="normal", fontsize = 10)
axe[0].locator_params(axis='x', nbins=5)

axe[0].xaxis.set_label_position("top")
axe[0].xaxis.tick_top()
axe[0].get_xaxis().get_major_formatter().set_scientific(False)
plt.show()

axe[1].hist(rho_base, 8, density=True, color='0.5') 
axe[1].set_ylabel('pdf',fontweight="normal", fontsize = 10)
axe[1].set_xlabel('(b) density contrast of basement',fontweight="normal", fontsize = 10)
axe[1].locator_params(axis='x', nbins=5)

# axe[1].yaxis.set_label_position("right")
# axe[1].yaxis.tick_right()
plt.show()

figname = 'rho_hist'
fig.savefig(fpath_plots+'/'+figname+'.pdf')
plt.close(fig) 

#########################################################################################

TF0_x_g = np.squeeze(k_AR_keep[:,0]==0)
TF1_x_g = np.squeeze(k_AR_keep[:,0]==1)
TF2_x_g = np.squeeze(k_AR_keep[:,0]==2)
TF3_x_g = np.squeeze(k_AR_keep[:,0]==3)

TF0_y_g = np.squeeze(k_AR_keep[:,1]==0)
TF1_y_g = np.squeeze(k_AR_keep[:,1]==1)
TF2_y_g = np.squeeze(k_AR_keep[:,1]==2)
TF3_y_g = np.squeeze(k_AR_keep[:,1]==3)

TF0_xy_g = np.squeeze(k_AR_keep[:,2]==0)
TF1_xy_g = np.squeeze(k_AR_keep[:,2]==1)
TF2_xy_g = np.squeeze(k_AR_keep[:,2]==2)
TF3_xy_g = np.squeeze(k_AR_keep[:,2]==3)



TF0_x_T = np.squeeze(k_AR_keep[:,3]==0)
TF1_x_T = np.squeeze(k_AR_keep[:,3]==1)
TF2_x_T = np.squeeze(k_AR_keep[:,3]==2)
TF3_x_T = np.squeeze(k_AR_keep[:,3]==3)

TF0_y_T = np.squeeze(k_AR_keep[:,4]==0)
TF1_y_T = np.squeeze(k_AR_keep[:,4]==1)
TF2_y_T = np.squeeze(k_AR_keep[:,4]==2)
TF3_y_T = np.squeeze(k_AR_keep[:,4]==3)

TF0_xy_T = np.squeeze(k_AR_keep[:,5]==0)
TF1_xy_T = np.squeeze(k_AR_keep[:,5]==1)
TF2_xy_T = np.squeeze(k_AR_keep[:,5]==2)
TF3_xy_T = np.squeeze(k_AR_keep[:,5]==3)


barval_x_g = [np.sum(TF0_x_g), np.sum(TF1_x_g), np.sum(TF2_x_g), np.sum(TF3_x_g)]/(np.sum(TF0_x_g)+np.sum(TF1_x_g)+np.sum(TF2_x_g)+np.sum(TF3_x_g)).copy()

barval_y_g = [np.sum(TF0_y_g), np.sum(TF1_y_g), np.sum(TF2_y_g), np.sum(TF3_y_g)]/(np.sum(TF0_y_g)+np.sum(TF1_y_g)+np.sum(TF2_y_g)+np.sum(TF3_y_g)).copy()

barval_xy_g = [np.sum(TF0_xy_g), np.sum(TF1_xy_g), np.sum(TF2_xy_g), np.sum(TF3_xy_g)]/(np.sum(TF0_xy_g)+np.sum(TF1_xy_g)+np.sum(TF2_xy_g)+np.sum(TF3_xy_g)).copy()

barval_x_T = [np.sum(TF0_x_T), np.sum(TF1_x_T), np.sum(TF2_x_T), np.sum(TF3_x_T)]/(np.sum(TF0_x_T)+np.sum(TF1_x_T)+np.sum(TF2_x_T)+np.sum(TF3_x_T)).copy()

barval_y_T = [np.sum(TF0_y_T), np.sum(TF1_y_T), np.sum(TF2_y_T), np.sum(TF3_y_T)]/(np.sum(TF0_y_T)+np.sum(TF1_y_T)+np.sum(TF2_y_T)+np.sum(TF3_y_T)).copy()

barval_xy_T = [np.sum(TF0_xy_T), np.sum(TF1_xy_T), np.sum(TF2_xy_T), np.sum(TF3_xy_T)]/(np.sum(TF0_xy_T)+np.sum(TF1_xy_T)+np.sum(TF2_xy_T)+np.sum(TF3_xy_T)).copy()

fs = 10

fig, axs = plt.subplots(3,2, sharex=False, sharey=False ,gridspec_kw={'wspace': 0, 'hspace': 0},figsize=(10, 5))
plt.rc('font', weight='normal')
plt.rc('xtick', labelsize=10)
plt.rc('ytick', labelsize=10)

bars = ('AR(0)', 'AR(1)', 'AR(2)', 'AR(3)')
y_pos = np.arange(len(bars))
axs[0,0].bar(y_pos, barval_x_g, color=['grey', 'grey', 'grey', 'grey'])
axs[0,0].set_xticks(y_pos)
axs[0,0].set_xticklabels(bars)
# axs[0,0].set_xlabel('(a) AR Gravity Orders',fontweight="normal", fontsize = 10)
axs[0,0].set_ylabel('pdf',fontweight="normal", fontsize = 10)
axs[0,0].text(3,0.6,'(a)', fontweight="normal", fontsize=fs)

axs[1,0].bar(y_pos, barval_y_g,color=['grey', 'grey', 'grey', 'grey'])
axs[1,0].set_xticks(y_pos)
axs[1,0].set_xticklabels(bars)
# axs[1,0].set_xlabel('(b) AR Magnetic Orders',fontweight="normal", fontsize = 10)
axs[1,0].set_ylabel('pdf',fontweight="normal", fontsize = 10)
# axs[1,0].yaxis.tick_right()
# axs[1,0].yaxis.set_label_position("right")
axs[1,0].text(3,0.6,'(b)', fontweight="normal", fontsize=fs)


axs[2,0].bar(y_pos, barval_xy_g,color=['grey', 'grey', 'grey', 'grey'])
axs[2,0].set_xticks(y_pos)
axs[2,0].set_xticklabels(bars)
axs[2,0].set_xlabel('AR Gravity Orders',fontweight="normal", fontsize = 10)
axs[2,0].set_ylabel('pdf',fontweight="normal", fontsize = 10)
# axs[2,0].yaxis.tick_right()
# axs[2,0].yaxis.set_label_position("right")
axs[2,0].text(3,0.6,'(c)', fontweight="normal", fontsize=fs)


axs[0,1].bar(y_pos, barval_x_T, color=['grey', 'grey', 'grey', 'grey'])
axs[0,1].set_xticks(y_pos)
axs[0,1].set_xticklabels(bars)
# axs[0,1].set_xlabel('(a) AR Gravity Orders',fontweight="normal", fontsize = 10)
axs[0,1].set_ylabel('pdf',fontweight="normal", fontsize = 10)
axs[0,1].yaxis.tick_right()
axs[0,1].yaxis.set_label_position("right")
axs[0,1].text(3,0.6,'(d)', fontweight="normal", fontsize=fs)


axs[1,1].bar(y_pos, barval_y_T,color=['grey', 'grey', 'grey', 'grey'])
axs[1,1].set_xticks(y_pos)
axs[1,1].set_xticklabels(bars)
# axs[1,1].set_xlabel('(b) AR Magnetic Orders',fontweight="normal", fontsize = 10)
axs[1,1].set_ylabel('pdf',fontweight="normal", fontsize = 10)
axs[1,1].yaxis.tick_right()
axs[1,1].yaxis.set_label_position("right")
axs[1,1].text(3,0.6,'(e)', fontweight="normal", fontsize=fs)


axs[2,1].bar(y_pos, barval_xy_T,color=['grey', 'grey', 'grey', 'grey'])
axs[2,1].set_xticks(y_pos)
axs[2,1].set_xticklabels(bars)
axs[2,1].set_xlabel('AR Magnetic Orders',fontweight="normal", fontsize = 10)
axs[2,1].set_ylabel('pdf',fontweight="normal", fontsize = 10)
axs[2,1].yaxis.tick_right()
axs[2,1].yaxis.set_label_position("right")    
axs[2,1].text(3,0.6,'(f)', fontweight="normal", fontsize=fs)

plt.show()
figname = 'AR_Order'
fig.savefig(fpath_plots+'/'+figname+'.pdf')
plt.close(fig)    # close the figure window


###################################################################################################

AR1_x_g = ARx_g_keep[TF1_x_g,0:1].copy()
AR2_x_g = ARx_g_keep[TF2_x_g,0:2].copy()
AR3_x_g = ARx_g_keep[TF3_x_g,0:3].copy()

AR1_y_g = ARy_g_keep[TF1_y_g,0:1].copy()
AR2_y_g = ARy_g_keep[TF2_y_g,0:2].copy()
AR3_y_g = ARy_g_keep[TF3_y_g,0:3].copy()

AR1_xy_g = ARxy_g_keep[TF1_xy_g,0:1].copy()
AR2_xy_g = ARxy_g_keep[TF2_xy_g,0:2].copy()
AR3_xy_g = ARxy_g_keep[TF3_xy_g,0:3].copy()

AR1_x_T = ARx_T_keep[TF1_x_T,0:1].copy()
AR2_x_T = ARx_T_keep[TF2_x_T,0:2].copy()
AR3_x_T = ARx_T_keep[TF3_x_T,0:3].copy()

AR1_y_T = ARy_T_keep[TF1_y_T,0:1].copy()
AR2_y_T = ARy_T_keep[TF2_y_T,0:2].copy()
AR3_y_T = ARy_T_keep[TF3_y_T,0:3].copy()

AR1_xy_T = ARxy_T_keep[TF1_xy_T,0:1].copy()
AR2_xy_T = ARxy_T_keep[TF2_xy_T,0:2].copy()
AR3_xy_T = ARxy_T_keep[TF3_xy_T,0:3].copy()

n_bins = 10
fs = 12

fig, axs = plt.subplots(9,2, sharex=False, sharey=False ,gridspec_kw={'wspace': 0.05, 'hspace': 0},figsize=(10, 10))
plt.rc('font', weight='normal')
plt.rc('xtick', labelsize=12)
plt.rc('ytick', labelsize=12)

axs[0,0].hist(AR1_x_g, density=True, histtype='step', fill=False, linewidth=2, ls='-', color='black') 
axs[0,0].axvline(x=AR_parameters_original_g[0,1], color='black', ls='--', lw=2)
axs[0,0].set_xlim([-1, 1])
axs[0,0].set_xticklabels([])
axs[0,0].set_ylabel('ARx(1)',fontweight="normal", fontsize = fs)
axs[0,0].text(-0.9,10,'(a)', fontsize=fs)


axs[1,0].hist(AR2_x_g[:,0], density=True, histtype='step', fill=False, linewidth=2, ls='-', color='black')
axs[1,0].hist(AR2_x_g[:,1], density=True, histtype='step', fill=False, linewidth=2, ls='--', color='red') 
axs[1,0].axvline(x=AR_parameters_original_g[0,1], color='black', ls='--', lw=2)
axs[1,0].axvline(x=AR_parameters_original_g[0,2], color='black', ls='--', lw=2)
axs[1,0].set_xlim([-1, 1])
axs[1,0].set_xticklabels([])
axs[1,0].set_ylabel('ARx(2)',fontweight="normal", fontsize = fs)
axs[1,0].text(-0.9,10,'(b)', fontsize=fs)


axs[2,0].hist(AR3_x_g[:,0], density=True, histtype='step', fill=False, linewidth=2, ls='-', color='black')
axs[2,0].hist(AR3_x_g[:,1], density=True, histtype='step', fill=False, linewidth=2, ls='--', color='red') 
axs[2,0].hist(AR3_x_g[:,2], density=True, histtype='step', fill=False, linewidth=2, ls='-.', color='m') 
axs[2,0].axvline(x=AR_parameters_original_g[0,1], color='black', ls='--', lw=2)
axs[2,0].axvline(x=AR_parameters_original_g[0,2], color='black', ls='--', lw=2)
axs[2,0].set_xlim([-1, 1])
axs[2,0].set_xticklabels([])
axs[2,0].set_ylabel('ARx(3)',fontweight="normal", fontsize = fs)
axs[2,0].text(-0.9,10,'(c)', fontsize=fs)


axs[3,0].hist(AR1_y_g, density=True, histtype='step', fill=False, linewidth=2, ls='-', color='black') 
axs[3,0].axvline(x=AR_parameters_original_g[1,0], color='black', ls='--', lw=2)
axs[3,0].set_xlim([-1, 1])
axs[3,0].set_xticklabels([])
# axs[3,0].set_xlabel('AR Coefficent',fontweight="normal", fontsize = 20)
axs[3,0].set_ylabel('ARy(1)',fontweight="normal", fontsize = fs)
axs[3,0].text(-0.9,10,'(d)', fontsize=fs)


axs[4,0].hist(AR2_y_g[:,0], density=True, histtype='step', fill=False, linewidth=2, ls='-', color='black')
axs[4,0].hist(AR2_y_g[:,1], density=True, histtype='step', fill=False, linewidth=2, ls='--', color='red') 
axs[4,0].axvline(x=AR_parameters_original_g[1,0], color='black', ls='--', lw=2)
axs[4,0].set_xlim([-1, 1])
axs[4,0].set_xticklabels([])
# axs[4,0].set_xlabel('AR Coefficent',fontweight="normal", fontsize = 20)
axs[4,0].set_ylabel('ARy(2)',fontweight="normal", fontsize = fs)
axs[4,0].text(-0.9,10,'(e)', fontsize=fs)


axs[5,0].hist(AR3_y_g[:,0], density=True, histtype='step', fill=False, linewidth=2, ls='-', color='black')
axs[5,0].hist(AR3_y_g[:,1], density=True, histtype='step', fill=False, linewidth=2, ls='--', color='red') 
axs[5,0].hist(AR3_y_g[:,2], density=True, histtype='step', fill=False, linewidth=2, ls='-.', color='m') 
axs[5,0].axvline(x=AR_parameters_original_g[1,0], color='black', ls='--', lw=2)
axs[5,0].set_xlim([-1, 1])
axs[5,0].set_xticklabels([])
# axs[5,0].set_xlabel('AR Coefficent',fontweight="normal", fontsize = 20)
axs[5,0].set_ylabel('ARy(3)',fontweight="normal", fontsize = fs)
axs[5,0].text(-0.9,10,'(f)', fontsize=fs)


axs[6,0].hist(AR1_xy_g, density=True, histtype='step', fill=False, linewidth=2, ls='-', color='black') 
axs[6,0].axvline(x=AR_parameters_original_g[1,1], color='black', ls='--', lw=2)
axs[6,0].set_xlim([-1, 1])
axs[6,0].set_xticklabels([])
# axs[6,0].set_xlabel('AR Coefficent',fontweight="normal", fontsize = 20)
axs[6,0].set_ylabel('ARxy(1)',fontweight="normal", fontsize = fs)
axs[6,0].text(-0.9,10,'(g)', fontsize=fs)


axs[7,0].hist(AR2_xy_g[:,0], density=True, histtype='step', fill=False, linewidth=2, ls='-', color='black')
axs[7,0].hist(AR2_xy_g[:,1], density=True, histtype='step', fill=False, linewidth=2, ls='--', color='red') 
axs[7,0].axvline(x=AR_parameters_original_g[1,1], color='black', ls='--', lw=2)
axs[7,0].set_xlim([-1, 1])
axs[7,0].set_xticklabels([])
# axs[7,0].set_xlabel('AR Coefficent',fontweight="normal", fontsize = 20)
axs[7,0].set_ylabel('ARxy(2)',fontweight="normal", fontsize = fs)
axs[7,0].text(-0.9,10,'(h)', fontsize=fs)


axs[8,0].hist(AR3_xy_g[:,0], density=True, histtype='step', fill=False, linewidth=2, ls='-', color='black')
axs[8,0].hist(AR3_xy_g[:,1], density=True, histtype='step', fill=False, linewidth=2, ls='--', color='red') 
axs[8,0].hist(AR3_xy_g[:,2], density=True, histtype='step', fill=False, linewidth=2, ls='-.', color='m') 
axs[8,0].axvline(x=AR_parameters_original_g[1,1], color='black', ls='--', lw=2)
axs[8,0].set_xlim([-1, 1])
axs[8,0].set_xlabel('AR Coefficent',fontweight="normal", fontsize = fs)
axs[8,0].set_ylabel('ARxy(3)',fontweight="normal", fontsize = fs)
axs[8,0].text(-0.9,10,'(i)', fontsize=fs)

#############
#############

axs[0,1].hist(AR1_x_T, density=True, histtype='step', fill=False, linewidth=2, ls='-', color='black') 
axs[0,1].axvline(x=AR_parameters_original_T[0,1], color='black', ls='--', lw=2)
axs[0,1].set_xlim([-1, 1])
axs[0,1].set_xticklabels([])
# axs[0,1].set_ylabel('ARx(1)',fontweight="normal", fontsize = fs)
axs[0,1].text(-0.9,10,'(a)', fontsize=fs)


axs[1,1].hist(AR2_x_T[:,0], density=True, histtype='step', fill=False, linewidth=2, ls='-', color='black')
axs[1,1].hist(AR2_x_T[:,1], density=True, histtype='step', fill=False, linewidth=2, ls='--', color='red') 
axs[1,1].axvline(x=AR_parameters_original_T[0,1], color='black', ls='--', lw=2)
# axs[1,1].axvline(x=AR_parameters_original_T[0,2], color='black', ls='--', lw=2)
axs[1,1].set_xlim([-1, 1])
axs[1,1].set_xticklabels([])
# axs[1,1].set_ylabel('ARx(2)',fontweight="normal", fontsize = fs)
axs[1,1].text(-0.9,10,'(b)', fontsize=fs)


axs[2,1].hist(AR3_x_T[:,0], density=True, histtype='step', fill=False, linewidth=2, ls='-', color='black')
axs[2,1].hist(AR3_x_T[:,1], density=True, histtype='step', fill=False, linewidth=2, ls='--', color='red') 
axs[2,1].hist(AR3_x_T[:,2], density=True, histtype='step', fill=False, linewidth=2, ls='-.', color='m') 
axs[2,1].axvline(x=AR_parameters_original_T[0,1], color='black', ls='--', lw=2)
# axs[2,1].axvline(x=AR_parameters_original_g[0,2], color='black', ls='--', lw=2)
axs[2,1].set_xlim([-1, 1])
axs[2,1].set_xticklabels([])
# axs[2,1].set_ylabel('ARx(3)',fontweight="normal", fontsize = fs)
axs[2,1].text(-0.9,10,'(c)', fontsize=fs)


axs[3,1].hist(AR1_y_T, density=True, histtype='step', fill=False, linewidth=2, ls='-', color='black') 
axs[3,1].axvline(x=AR_parameters_original_T[1,0], color='black', ls='--', lw=2)
axs[3,1].set_xlim([-1, 1])
axs[3,1].set_xticklabels([])
# axs[3,0].set_xlabel('AR Coefficent',fontweight="normal", fontsize = 20)
# axs[3,1].set_ylabel('ARy(1)',fontweight="normal", fontsize = fs)
axs[3,1].text(-0.9,10,'(d)', fontsize=fs)


axs[4,1].hist(AR2_y_T[:,0], density=True, histtype='step', fill=False, linewidth=2, ls='-', color='black')
axs[4,1].hist(AR2_y_T[:,1], density=True, histtype='step', fill=False, linewidth=2, ls='--', color='red') 
axs[4,1].axvline(x=AR_parameters_original_T[1,0], color='black', ls='--', lw=2)
axs[4,1].set_xlim([-1, 1])
axs[4,1].set_xticklabels([])
# axs[4,0].set_xlabel('AR Coefficent',fontweight="normal", fontsize = 20)
# axs[4,0].set_ylabel('ARy(2)',fontweight="normal", fontsize = fs)
axs[4,0].text(-0.9,10,'(e)', fontsize=fs)


axs[5,1].hist(AR3_y_T[:,0], density=True, histtype='step', fill=False, linewidth=2, ls='-', color='black')
axs[5,1].hist(AR3_y_T[:,1], density=True, histtype='step', fill=False, linewidth=2, ls='--', color='red') 
axs[5,1].hist(AR3_y_T[:,2], density=True, histtype='step', fill=False, linewidth=2, ls='-.', color='m') 
axs[5,1].axvline(x=AR_parameters_original_T[1,0], color='black', ls='--', lw=2)
axs[5,1].set_xlim([-1, 1])
axs[5,1].set_xticklabels([])
# axs[5,0].set_xlabel('AR Coefficent',fontweight="normal", fontsize = 20)
# axs[5,0].set_ylabel('ARy(3)',fontweight="normal", fontsize = fs)
axs[5,0].text(-0.9,10,'(f)', fontsize=fs)


axs[6,1].hist(AR1_xy_T, density=True, histtype='step', fill=False, linewidth=2, ls='-', color='black') 
axs[6,1].axvline(x=AR_parameters_original_T[1,1], color='black', ls='--', lw=2)
axs[6,1].set_xlim([-1, 1])
axs[6,1].set_xticklabels([])
# axs[6,0].set_xlabel('AR Coefficent',fontweight="normal", fontsize = 20)
# axs[6,1].set_ylabel('ARxy(1)',fontweight="normal", fontsize = fs)
axs[6,1].text(-0.9,10,'(g)', fontsize=fs)


axs[7,1].hist(AR2_xy_T[:,0], density=True, histtype='step', fill=False, linewidth=2, ls='-', color='black')
axs[7,1].hist(AR2_xy_T[:,1], density=True, histtype='step', fill=False, linewidth=2, ls='--', color='red') 
axs[7,1].axvline(x=AR_parameters_original_T[1,1], color='black', ls='--', lw=2)
axs[7,1].set_xlim([-1, 1])
axs[7,1].set_xticklabels([])
# axs[7,0].set_xlabel('AR Coefficent',fontweight="normal", fontsize = 20)
# axs[7,1].set_ylabel('ARxy(2)',fontweight="normal", fontsize = fs)
axs[7,1].text(-0.9,10,'(h)', fontsize=fs)


axs[8,1].hist(AR3_xy_T[:,0], density=True, histtype='step', fill=False, linewidth=2, ls='-', color='black')
axs[8,1].hist(AR3_xy_T[:,1], density=True, histtype='step', fill=False, linewidth=2, ls='--', color='red') 
axs[8,1].hist(AR3_xy_T[:,2], density=True, histtype='step', fill=False, linewidth=2, ls='-.', color='m') 
axs[8,1].axvline(x=AR_parameters_original_T[1,1], color='black', ls='--', lw=2)
axs[8,1].set_xlim([-1, 1])
axs[8,1].set_xlabel('AR Coefficent',fontweight="normal", fontsize = fs)
# axs[8,1].set_ylabel('ARxy(3)',fontweight="normal", fontsize = fs)
axs[8,1].text(-0.9,10,'(i)', fontsize=fs)


plt.show()
figname = 'AR_Hist'
fig.savefig(fpath_plots+'/'+figname+'.pdf')
plt.close(fig)    # close the figure window

# ##########################################################################################
# ##########################################################################################
##########################################################################################
### Plot Model
    
fig, axs = plt.subplots(3,2, sharex=False, sharey=False ,gridspec_kw={'wspace': 0, 'hspace': 0},figsize=(10, 8))
plt.rc('font', weight='normal')
plt.rc('xtick', labelsize=12)
plt.rc('ytick', labelsize=12)

pos00 = axs[0,0].get_position() # get the original position
pos10 = axs[1,0].get_position() # get the original position 
pos20 = axs[2,0].get_position() # get the original position 
pos01 = axs[0,1].get_position() # get the original position
pos11 = axs[1,1].get_position() # get the original position
pos21 = axs[2,1].get_position() # get the original position
pos00.x0 += 0.1  # or use: pos00 = [pos00.x0 + 0.1, pos00.y0 ,  pos00.width, pos00.height] 
pos10.x0 += 0.1
pos20.x0 += 0.1
pos01.x1 -= 0.1
pos11.x1 -= 0.1
pos21.x1 -= 0.1
axs[0,0].set_position(pos00) # set a new position
axs[1,0].set_position(pos10) # set a new position
axs[2,0].set_position(pos20) # set a new position
axs[0,1].set_position(pos01) # set a new position
axs[1,1].set_position(pos11) # set a new position
axs[2,1].set_position(pos21) # set a new position

iz = 55
iy = 30
ix = 30

im00 = axs[0,0].imshow(TrueDensityModel[iz,:,:],interpolation='none',
       vmin=rho_salt_min, vmax=rho_base_max, extent=(x1,x2,y1,y2), aspect='auto', cmap='seismic')
axs[0,0].text(0.1,.9,'(a)',horizontalalignment='center',transform=axs[0,0].transAxes, fontweight="normal", fontsize = 12)
axs[0,0].set_ylabel('Depth (km)',fontweight="normal", fontsize = 12)
# axs[0,0].set_xticklabels([])
axs[0,0].yaxis.tick_left()
axs[0,0].tick_params(axis="x",direction="in")
# axs[0,0].tick_params(axis='both', labelsize=15)


im10 = axs[1,0].imshow(TrueDensityModel[:,iy,:],interpolation='none',
       vmin=rho_salt_min, vmax=rho_base_max, extent=(x1,x2,z2,z1), aspect='auto', cmap='seismic')
axs[1,0].text(0.1,.9,'(b)',horizontalalignment='center',transform=axs[1,0].transAxes, fontweight="normal", fontsize = 12)
axs[1,0].set_ylabel('Depth (km)',fontweight="normal", fontsize = 12)
# axs[1,0].set_xticklabels([])
axs[1,0].yaxis.tick_left()
axs[1,0].tick_params(axis="x",direction="in")

CI_g_Width_NaN = CI_g_Width.copy()
CI_g_Width_NaN[CI_g_Width==0] = np.NaN
NewJet = cm.jet
NewJet.set_bad("white")
im20 = axs[2,0].imshow(TrueDensityModel[:,:,ix],interpolation='none',
       vmin=rho_salt_min, vmax=rho_base_max, extent=(y1,y2,z2,z1), aspect='auto', cmap='seismic')
axs[2,0].text(0.1,.9,'(c)',horizontalalignment='center',transform=axs[2,0].transAxes, fontweight="normal", fontsize = 12)
axs[2,0].set_ylabel('Depth (km)',fontweight="normal", fontsize = 12)
axs[2,0].set_xlabel('Distance (km)',fontweight="normal", fontsize = 12)
axs[2,0].yaxis.tick_left()

im01 = axs[0,1].imshow(PMD_g[iz,:,:],interpolation='none',
       vmin=rho_salt_min, vmax=rho_base_max, extent=(x1,x2,y1,y2), aspect='auto', cmap='seismic')
axs[0,1].text(0.1,.9,'(d)',horizontalalignment='center',transform=axs[0,1].transAxes, fontweight="normal", fontsize = 12)
# axs[0,1].set_xticklabels([])
axs[0,1].tick_params(axis="y",direction="in")
axs[0,1].tick_params(axis="x",direction="in")

im11 = axs[1,1].imshow(PMD_g[:,iy,:],interpolation='none',
       vmin=rho_salt_min, vmax=rho_base_max, extent=(x1,x2,z2,z1), aspect='auto', cmap='seismic')
axs[1,1].text(0.1,.9,'(e)',horizontalalignment='center',transform=axs[1,1].transAxes, fontweight="normal", fontsize = 12)
# axs[1,1].set_xticklabels([])
axs[1,1].tick_params(axis="y",direction="in")
axs[1,1].tick_params(axis="x",direction="in")

im21 = axs[2,1].imshow(PMD_g[:,:,ix],interpolation='none',
       vmin=rho_salt_min, vmax=rho_base_max, extent=(y1,y2,z2,z1), aspect='auto', cmap='seismic')
axs[2,1].text(0.1,.9,'(f)',horizontalalignment='center',transform=axs[2,1].transAxes, fontweight="normal", fontsize = 12)
axs[2,1].set_xlabel('Distance (km)',fontweight="normal", fontsize = 12)
axs[2,1].tick_params(axis="y",direction="in")

cbar_pos_density = fig.add_axes([0.1, 0.4, 0.03, 0.45]) 
cbar_density = plt.colorbar(im00, ax=axs[0,0] ,shrink=0.3, cax = cbar_pos_density,
                    orientation='vertical', ticklocation = 'left')
cbar_density.ax.tick_params(labelsize=12)
cbar_density.set_label(label = 'Density Contrast ($\mathregular{g/cm^{3}}$)', weight='normal')


# cbar_pos_jet = fig.add_axes([0.1, 0.1, 0.03, 0.2]) 
# cbar_jet = plt.colorbar(im20, ax=axs[2,0] ,shrink=0.3,  cax = cbar_pos_jet,
#                     orientation='vertical', ticklocation = 'left')
# cbar_jet.ax.tick_params(labelsize=12)
# cbar_jet.set_label(label='95% CI Widths ($\mathregular{g/cm^{3}}$)', weight='normal')


# cbar_pos_sus = fig.add_axes([0.85, 0.4, 0.03, 0.45]) 
# cbar_sus = plt.colorbar(im11, ax=axs[1,1] ,shrink=0.3, cax = cbar_pos_sus,
#                     orientation='vertical', ticklocation = 'right')
# cbar_sus.ax.tick_params(labelsize=12)
# cbar_sus.set_label(label='Susceptibility (SI)', weight='normal')

# cbar_pos_95_sus = fig.add_axes([0.85, 0.1, 0.03, 0.2]) 
# cbar_95_sus = plt.colorbar(im11, ax=axs[1,1] ,shrink=0.3, cax = cbar_pos_95_sus,
#                     orientation='vertical', ticklocation = 'right')
# cbar_95_sus.ax.tick_params(labelsize=12)
# cbar_95_sus.set_label(label='95% CI Widths (SI)', weight='normal')

# for ax in axs.flat:
#     ax.label_outer()

plt.show()
figname = 'True_PMD_Model'
fig.savefig(fpath_plots+'/'+figname+'.pdf')
plt.close(fig)    # close the figure window
#########################################################################################
### Plot Data Fit

fig, axe = plt.subplots(1, 2)
plt.rc('font', weight='normal')
plt.rc('xtick', labelsize=12)
plt.rc('ytick', labelsize=12)

dg_obs_Grid = dg_obs.reshape((Ndata,Ndata))
dT_obs_Grid = dT_obs.reshape((Ndata,Ndata))

PMD_data_g_Grid = PMD_data_g.reshape((Ndata,Ndata))
PMD_data_T_Grid = PMD_data_T.reshape((Ndata,Ndata))

data_g_error_Grid = data_g_error.reshape((Ndata,Ndata))
data_T_error_Grid = data_T_error.reshape((Ndata,Ndata))

axe[0].fill_between(xs_plot, PMD_data_g_Grid[15,:]-data_g_error_Grid[15,:], PMD_data_g_Grid[15,:]+data_g_error_Grid[15,:],facecolor='0.5')
# axe[0].plot(dis,PMD_data_g_Grid[15,:], 'b--',linewidth=2) #row=0, col=0
axe[0].plot(xs_plot,dg_obs_Grid[15,:], 'k.',linewidth=2) #row=0, col=0
axe[0].set_xlabel('Distance (km)',fontweight="normal", fontsize = 8)
axe[0].set_ylabel('Gravity (mGal)',fontweight="normal", fontsize = 8)
axe[0].text(-5,5.4,'(a)', fontweight="normal", fontsize = 12)
plt.show()

axe[1].fill_between(xs_plot, PMD_data_T_Grid[15,:]-data_T_error_Grid[15,:], PMD_data_T_Grid[15,:]+data_T_error_Grid[15,:],facecolor='0.5')
# axe[1].plot(dis,PMD_data_T_Grid[15,:], 'b--',linewidth=2) #row=0, col=0
axe[1].plot(xs_plot,dT_obs_Grid[15,:], 'k.',linewidth=1) #row=0, col=0
axe[1].set_xlabel('Distance (km)',fontweight="normal", fontsize = 8)
axe[1].set_ylabel('Magnetic (nT)',fontweight="normal", fontsize = 8)
axe[1].yaxis.set_label_position("right")
axe[1].yaxis.tick_right()
axe[1].text(-5,52,'(b)', fontweight="normal", fontsize = 12)

plt.show()
figname = 'Data_Fit'
fig.savefig(fpath_plots+'/'+figname+'.pdf')
plt.close(fig)    # close the figure window
#######################################################################################
#######################################################################################
### Plot Multiple Model Slices
     
fs = 15

fig, axs = plt.subplots(5,4, sharex=False, sharey=False ,gridspec_kw={'wspace':0.0 , 'hspace': 0.0},figsize=(10, 8))
plt.rc('font', weight='normal')
plt.rc('xtick', labelsize=12)
plt.rc('ytick', labelsize=12)

# pos00 = axs[0,0].get_position() # get the original position
# pos10 = axs[1,0].get_position() # get the original position 
# pos20 = axs[2,0].get_position() # get the original position
# pos30 = axs[3,0].get_position() # get the original position 
# pos40 = axs[4,0].get_position() # get the original position 
# pos01 = axs[0,1].get_position() # get the original position
# pos11 = axs[1,1].get_position() # get the original position
# pos21 = axs[2,1].get_position() # get the original position
# pos31 = axs[3,1].get_position() # get the original position
# pos41 = axs[4,1].get_position() # get the original position
# pos00.x0 += 0.1  # or use: pos00 = [pos00.x0 + 0.1, pos00.y0 ,  pos00.width, pos00.height] 
# pos10.x0 += 0.1
# pos20.x0 += 0.1
# pos30.x0 += 0.1
# pos40.x0 += 0.1
# pos01.x1 -= 0.1
# pos11.x1 -= 0.1
# pos21.x1 -= 0.1
# pos31.x1 -= 0.1
# pos41.x1 -= 0.1
# axs[0,0].set_position(pos00) # set a new position
# axs[1,0].set_position(pos10) # set a new position
# axs[2,0].set_position(pos20) # set a new position
# axs[3,0].set_position(pos30) # set a new position
# axs[4,0].set_position(pos40) # set a new position
# axs[0,1].set_position(pos01) # set a new position
# axs[1,1].set_position(pos11) # set a new position
# axs[2,1].set_position(pos21) # set a new position
# axs[3,1].set_position(pos31) # set a new position
# axs[4,1].set_position(pos41) # set a new position

ix0 = 20
ix1 = 25
ix2 = 30
ix3 = 35
ix4 = 40

iy0 = 20
iy1 = 25
iy2 = 30
iy3 = 35
iy4 = 40

im00 = axs[0,0].imshow(TrueDensityModel[:,iy0,:],interpolation='none',
       vmin=rho_salt_min, vmax=rho_base_max, extent=(x1,x2,z2,z1), aspect='auto', cmap='seismic')
axs[0,0].text(0.1,.8,'(a)',horizontalalignment='center',transform=axs[0,0].transAxes, fontweight="normal", fontsize = fs)
# axs[0,0].set_ylabel('Depth (km)',fontweight="normal", fontsize = 12)
axs[0,0].set_xticklabels([])
axs[0,0].yaxis.tick_left()
axs[0,0].tick_params(axis="x",direction="in")
plt.setp(axs[0,0], yticks=[2.5, 5, 7.5, 10], yticklabels=['2.5', '5', '7.5', '10'])
plt.setp(axs[0,0], xticks=[0, 20, 40, 60])
# axs[0,0].tick_params(axis='both', labelsize=15)


im10 = axs[1,0].imshow(TrueDensityModel[:,iy1,:],interpolation='none',
       vmin=rho_salt_min, vmax=rho_base_max, extent=(x1,x2,z2,z1), aspect='auto', cmap='seismic')
axs[1,0].text(0.1,.8,'(b)',horizontalalignment='center',transform=axs[1,0].transAxes, fontweight="normal", fontsize = fs)
# axs[1,0].set_ylabel('Depth (km)',fontweight="normal", fontsize = 12)
axs[1,0].set_xticklabels([])
axs[1,0].yaxis.tick_left()
axs[1,0].tick_params(axis="x",direction="in")
plt.setp(axs[1,0], yticks=[2.5, 5, 7.5, 10], yticklabels=['2.5', '5', '7.5', '10'])
plt.setp(axs[1,0], xticks=[0, 20, 40, 60])
# axs[1,0].text(0.1,.9,str(),horizontalalignment='center',transform=axs[1,0].transAxes, fontweight="normal", fontsize = fs)



im20 = axs[2,0].imshow(TrueDensityModel[:,iy2,:],interpolation='none',
       vmin=rho_salt_min, vmax=rho_base_max, extent=(x1,x2,z2,z1), aspect='auto', cmap='seismic')
axs[2,0].text(0.1,.8,'(c)',horizontalalignment='center',transform=axs[2,0].transAxes, fontweight="normal", fontsize = fs)
# axs[2,0].set_ylabel('Depth (km)',fontweight="normal", fontsize = 12)
axs[2,0].set_xticklabels([])
axs[2,0].yaxis.tick_left()
axs[2,0].tick_params(axis="x",direction="in")
# axs[2,0].text(-0.2,0,'Depth (km)',horizontalalignment='center',transform=axs[2,0].transAxes, fontweight="normal", fontsize = fs, rotation = 90)
plt.setp(axs[2,0], yticks=[2.5, 5, 7.5, 10], yticklabels=['2.5', '5', '7.5', '10'])
plt.setp(axs[2,0], xticks=[0, 20, 40, 60])

im30 = axs[3,0].imshow(TrueDensityModel[:,iy3,:],interpolation='none',
       vmin=rho_salt_min, vmax=rho_base_max, extent=(x1,x2,z2,z1), aspect='auto', cmap='seismic')
axs[3,0].text(0.1,.8,'(d)',horizontalalignment='center',transform=axs[3,0].transAxes, fontweight="normal", fontsize = fs)
# axs[3,0].set_ylabel('Depth (km)',fontweight="normal", fontsize = 12)
axs[3,0].set_xticklabels([])
axs[3,0].yaxis.tick_left()
axs[3,0].tick_params(axis="x",direction="in")
plt.setp(axs[3,0], yticks=[2.5, 5, 7.5, 10], yticklabels=['2.5', '5', '7.5', '10'])
plt.setp(axs[3,0], xticks=[0, 20, 40, 60])

im40 = axs[4,0].imshow(TrueDensityModel[:,iy4,:],interpolation='none',
       vmin=rho_salt_min, vmax=rho_base_max, extent=(x1,x2,z2,z1), aspect='auto', cmap='seismic')
axs[4,0].text(0.1,.8,'(e)',horizontalalignment='center',transform=axs[4,0].transAxes, fontweight="normal", fontsize = fs)
# axs[4,0].set_ylabel('Depth (km)',fontweight="normal", fontsize = 12)
# axs[1,0].set_xticklabels([])
# axs[4,0].yaxis.tick_left()
axs[4,0].tick_params(axis="x",direction="in")
axs[4,0].set_xlabel('Easting (km)',fontweight="normal", fontsize = fs)
plt.setp(axs[4,0], yticks=[2.5, 5, 7.5, 10], yticklabels=['2.5', '5', '7.5', '10'])
plt.setp(axs[4,0], xticks=[0, 20, 40, 60] , xticklabels=['0', '20', '40', '60'])

im01 = axs[0,1].imshow(PMD_g[:,iy0,:],interpolation='none',
       vmin=rho_salt_min, vmax=rho_base_max, extent=(x1,x2,z2,z1), aspect='auto', cmap='seismic')
axs[0,1].text(0.1,.8,'(f)',horizontalalignment='center',transform=axs[0,1].transAxes, fontweight="normal", fontsize = fs)
# axs[0,0].set_ylabel('Depth (km)',fontweight="normal", fontsize = 12)
axs[0,1].set_xticklabels([])
axs[0,1].set_yticklabels([])
axs[0,1].yaxis.tick_left()
axs[0,1].tick_params(axis="x",direction="in")
plt.setp(axs[0,1], yticks=[2.5, 5, 7.5, 10])
plt.setp(axs[0,1], xticks=[0, 20, 40, 60])
# axs[0,0].tick_params(axis='both', labelsize=15)


im11 = axs[1,1].imshow(PMD_g[:,iy1,:],interpolation='none',
       vmin=rho_salt_min, vmax=rho_base_max, extent=(x1,x2,z2,z1), aspect='auto', cmap='seismic')
axs[1,1].text(0.1,.8,'(g)',horizontalalignment='center',transform=axs[1,1].transAxes, fontweight="normal", fontsize = fs)
# axs[1,0].set_ylabel('Depth (km)',fontweight="normal", fontsize = 12)
axs[1,1].set_xticklabels([])
axs[1,1].set_yticklabels([])
axs[1,1].yaxis.tick_left()
axs[1,1].tick_params(axis="x",direction="in")
plt.setp(axs[1,1], yticks=[2.5, 5, 7.5, 10])
plt.setp(axs[1,1], xticks=[0, 20, 40, 60])
# axs[1,0].text(0.1,.9,str(),horizontalalignment='center',transform=axs[1,0].transAxes, fontweight="normal", fontsize = fs)



im21 = axs[2,1].imshow(PMD_g[:,iy2,:],interpolation='none',
       vmin=rho_salt_min, vmax=rho_base_max, extent=(x1,x2,z2,z1), aspect='auto', cmap='seismic')
axs[2,1].text(0.1,.8,'(h)',horizontalalignment='center',transform=axs[2,1].transAxes, fontweight="normal", fontsize = fs)
# axs[2,0].set_ylabel('Depth (km)',fontweight="normal", fontsize = 12)
axs[2,1].set_xticklabels([])
axs[2,1].set_yticklabels([])
axs[2,1].yaxis.tick_left()
axs[2,1].tick_params(axis="x",direction="in")
axs[2,1].text(-0.35,0,'Depth (km)',horizontalalignment='center',transform=axs[2,0].transAxes, fontweight="normal", fontsize = fs, rotation = 90)
plt.setp(axs[2,1], yticks=[2.5, 5, 7.5, 10])
plt.setp(axs[2,1], xticks=[0, 20, 40, 60])

im31 = axs[3,1].imshow(PMD_g[:,iy3,:],interpolation='none',
       vmin=rho_salt_min, vmax=rho_base_max, extent=(x1,x2,z2,z1), aspect='auto', cmap='seismic')
axs[3,1].text(0.1,.8,'(i)',horizontalalignment='center',transform=axs[3,1].transAxes, fontweight="normal", fontsize = fs)
# axs[3,0].set_ylabel('Depth (km)',fontweight="normal", fontsize = 12)
axs[3,1].set_xticklabels([])
axs[3,1].set_yticklabels([])
axs[3,1].yaxis.tick_left()
axs[3,1].tick_params(axis="x",direction="in")
plt.setp(axs[3,0], yticks=[2.5, 5, 7.5, 10])
plt.setp(axs[3,1], xticks=[0, 20, 40, 60])

im41 = axs[4,1].imshow(PMD_g[:,iy4,:],interpolation='none',
       vmin=rho_salt_min, vmax=rho_base_max, extent=(x1,x2,z2,z1), aspect='auto', cmap='seismic')
axs[4,1].text(0.1,.8,'(j)',horizontalalignment='center',transform=axs[4,1].transAxes, fontweight="normal", fontsize = fs)
# axs[4,0].set_ylabel('Depth (km)',fontweight="normal", fontsize = 12)
# axs[1,0].set_xticklabels([])
axs[4,1].set_yticklabels([])
axs[4,1].yaxis.tick_left()
axs[4,1].tick_params(axis="x",direction="in")
axs[4,1].set_xlabel('Easting (km)',fontweight="normal", fontsize = fs)
plt.setp(axs[4,1], yticks=[2.5, 5, 7.5, 10])
plt.setp(axs[4,1], xticks=[0, 20, 40, 60], xticklabels=['0', '20', '40', '60'])
#########################
im02 = axs[0,2].imshow(TrueDensityModel[:,:,ix0],interpolation='none',
       vmin=rho_salt_min, vmax=rho_base_max, extent=(y1,y2,z2,z1), aspect='auto', cmap='seismic')
axs[0,2].text(0.1,.8,'(k)',horizontalalignment='center',transform=axs[0,2].transAxes, fontweight="normal", fontsize = fs)
# axs[0,1].set_ylabel('Depth (km)',fontweight="normal", fontsize = 12)
axs[0,2].yaxis.tick_left()
axs[0,2].set_yticklabels([])
axs[0,2].set_xticklabels([])
plt.setp(axs[0,2], yticks=[2.5, 5, 7.5, 10])
plt.setp(axs[0,2], xticks=[0, 20, 40, 60])

im12 = axs[1,2].imshow(np.fliplr(TrueDensityModel[:,:,ix1]),interpolation='none',
       vmin=rho_salt_min, vmax=rho_base_max, extent=(y1,y2,z2,z1), aspect='auto', cmap='seismic')
axs[1,2].text(0.1,.8,'(l)',horizontalalignment='center',transform=axs[1,2].transAxes, fontweight="normal", fontsize = fs)
# axs[1,1].set_ylabel('Depth (km)',fontweight="normal", fontsize = 12)
# axs[0,2].set_xlabel('Distance (km)',fontweight="normal", fontsize = 12)
axs[1,2].yaxis.tick_left()
axs[1,2].set_yticklabels([])
axs[1,2].set_xticklabels([])
plt.setp(axs[1,2], yticks=[2.5, 5, 7.5, 10])
plt.setp(axs[1,2], xticks=[0, 20, 40, 60])

im22 = axs[2,2].imshow(np.fliplr(TrueDensityModel[:,:,ix2]),interpolation='none',
       vmin=rho_salt_min, vmax=rho_base_max, extent=(y1,y2,z2,z1), aspect='auto', cmap='seismic')
axs[2,2].text(0.1,.8,'(m)',horizontalalignment='center',transform=axs[2,2].transAxes, fontweight="normal", fontsize = fs)
# axs[2,1].set_ylabel('Depth (km)',fontweight="normal", fontsize = 12)
# axs[0,3].set_xlabel('Distance (km)',fontweight="normal", fontsize = 12)
axs[2,2].yaxis.tick_left()
axs[2,2].set_yticklabels([])
axs[2,2].set_xticklabels([])
plt.setp(axs[2,2], yticks=[2.5, 5, 7.5, 10])
plt.setp(axs[2,2], xticks=[0, 20, 40, 60])

im32 = axs[3,2].imshow(np.fliplr(TrueDensityModel[:,:,ix3]),interpolation='none',
       vmin=rho_salt_min, vmax=rho_base_max, extent=(y1,y2,z2,z1), aspect='auto', cmap='seismic')
axs[3,2].text(0.1,.8,'(n)',horizontalalignment='center',transform=axs[3,2].transAxes, fontweight="normal", fontsize = fs)
# axs[3,1].set_ylabel('Depth (km)',fontweight="normal", fontsize = 12)
# axs[0,4].set_xlabel('Distance (km)',fontweight="normal", fontsize = 12)
axs[3,2].yaxis.tick_left()
axs[3,2].set_yticklabels([])
axs[3,2].set_xticklabels([])
plt.setp(axs[3,2], yticks=[2.5, 5, 7.5, 10])
plt.setp(axs[3,2], xticks=[0, 20, 40, 60])

im42 = axs[4,2].imshow(np.fliplr(TrueDensityModel[:,:,ix4]),interpolation='none',
       vmin=rho_salt_min, vmax=rho_base_max, extent=(y1,y2,z2,z1), aspect='auto', cmap='seismic')
axs[4,2].text(0.1,.8,'(o)',horizontalalignment='center',transform=axs[4,2].transAxes, fontweight="normal", fontsize = fs)
# axs[4,1].set_ylabel('Depth (km)',fontweight="normal", fontsize = 12)
axs[4,2].set_xlabel('Northing (km)',fontweight="normal", fontsize = fs)
axs[4,2].yaxis.tick_left()
axs[4,2].set_yticklabels([])
# axs[4,1].set_xticklabels([])
plt.setp(axs[4,2], yticks=[2.5, 5, 7.5, 10])
plt.setp(axs[4,2], xticks=[0, 20, 40, 60], xticklabels=['0', '20', '40', '60'])

im03 = axs[0,3].imshow(PMD_g[:,:,ix0],interpolation='none',
       vmin=rho_salt_min, vmax=rho_base_max, extent=(y1,y2,z2,z1), aspect='auto', cmap='seismic')
axs[0,3].text(0.1,.8,'(p)',horizontalalignment='center',transform=axs[0,3].transAxes, fontweight="normal", fontsize = fs)
# axs[0,1].set_ylabel('Depth (km)',fontweight="normal", fontsize = 12)
axs[0,3].yaxis.tick_left()
axs[0,3].set_yticklabels([])
axs[0,3].set_xticklabels([])
plt.setp(axs[0,3], yticks=[2.5, 5, 7.5, 10])
plt.setp(axs[0,3], xticks=[0, 20, 40, 60])

im13 = axs[1,3].imshow(np.fliplr(PMD_g[:,:,ix1]),interpolation='none',
       vmin=rho_salt_min, vmax=rho_base_max, extent=(y1,y2,z2,z1), aspect='auto', cmap='seismic')
axs[1,3].text(0.1,.8,'(q)',horizontalalignment='center',transform=axs[1,3].transAxes, fontweight="normal", fontsize = fs)
# axs[1,1].set_ylabel('Depth (km)',fontweight="normal", fontsize = 12)
# axs[0,2].set_xlabel('Distance (km)',fontweight="normal", fontsize = 12)
axs[1,3].yaxis.tick_left()
axs[1,3].set_yticklabels([])
axs[1,3].set_xticklabels([])
plt.setp(axs[1,3], yticks=[2.5, 5, 7.5, 10])
plt.setp(axs[1,3], xticks=[0, 20, 40, 60])

im23 = axs[2,3].imshow(np.fliplr(PMD_g[:,:,ix2]),interpolation='none',
       vmin=rho_salt_min, vmax=rho_base_max, extent=(y1,y2,z2,z1), aspect='auto', cmap='seismic')
axs[2,3].text(0.1,.8,'(r)',horizontalalignment='center',transform=axs[2,3].transAxes, fontweight="normal", fontsize = fs)
# axs[2,1].set_ylabel('Depth (km)',fontweight="normal", fontsize = 12)
# axs[0,3].set_xlabel('Distance (km)',fontweight="normal", fontsize = 12)
axs[2,3].yaxis.tick_left()
axs[2,3].set_yticklabels([])
axs[2,3].set_xticklabels([])
plt.setp(axs[2,3], yticks=[2.5, 5, 7.5, 10])
plt.setp(axs[2,3], xticks=[0, 20, 40, 60])

im33 = axs[3,3].imshow(np.fliplr(PMD_g[:,:,ix3]),interpolation='none',
       vmin=rho_salt_min, vmax=rho_base_max, extent=(y1,y2,z2,z1), aspect='auto', cmap='seismic')
axs[3,3].text(0.1,.8,'(s)',horizontalalignment='center',transform=axs[3,3].transAxes, fontweight="normal", fontsize = fs)
# axs[3,1].set_ylabel('Depth (km)',fontweight="normal", fontsize = 12)
# axs[0,4].set_xlabel('Distance (km)',fontweight="normal", fontsize = 12)
axs[3,3].yaxis.tick_left()
axs[3,3].set_yticklabels([])
axs[3,3].set_xticklabels([])
plt.setp(axs[3,1], yticks=[2.5, 5, 7.5, 10])
plt.setp(axs[3,1], xticks=[0, 20, 40, 60])

im43 = axs[4,3].imshow(np.fliplr(PMD_g[:,:,ix4]),interpolation='none',
       vmin=rho_salt_min, vmax=rho_base_max, extent=(y1,y2,z2,z1), aspect='auto', cmap='seismic')
axs[4,3].text(0.1,.8,'(t)',horizontalalignment='center',transform=axs[4,3].transAxes, fontweight="normal", fontsize = fs)
# axs[4,1].set_ylabel('Depth (km)',fontweight="normal", fontsize = 12)
axs[4,3].set_xlabel('Northing (km)',fontweight="normal", fontsize = fs)
axs[4,3].yaxis.tick_left()
axs[4,3].set_yticklabels([])
# axs[4,1].set_xticklabels([])
plt.setp(axs[4,3], yticks=[2.5, 5, 7.5, 10])
plt.setp(axs[4,3], xticks=[0, 20, 40, 60], xticklabels=['0', '20', '40', '60'])




cbar_pos_density = fig.add_axes([0.3, 0.9, 0.45, 0.03]) 
cbar_density = plt.colorbar(im00, ax=axs[0,0] ,shrink=0.3,  cax = cbar_pos_density,
                    orientation='horizontal', ticklocation = 'top')
cbar_density.ax.tick_params(labelsize=fs)
cbar_density.set_label(label = 'Density Contrast ($\mathregular{g/cm^{3}}$)', weight='normal', fontsize = fs)


plt.show()
figname = 'PMD_Slice'
fig.savefig(fpath_plots+'/'+figname+'.pdf')
plt.close(fig)    # close the figure window

#######################################################################################
### Plot Multiple Data Slices

yg_lim2 = np.amax(PMD_data_g_Grid)+2
yg_lim1 = np.amin(PMD_data_g_Grid)-2

yT_lim2 = np.amax(PMD_data_T_Grid)+10
yT_lim1 = np.amin(PMD_data_T_Grid)-10

fs = 15
     
fig, axs = plt.subplots(5,4, sharex=False, sharey=False ,gridspec_kw={'wspace':0.0 , 'hspace': 0.0},figsize=(10, 8))
plt.rc('font', weight='normal')
plt.rc('xtick', labelsize=fs)
plt.rc('ytick', labelsize=fs)


pos00 = axs[0,0].get_position() # get the original position
pos10 = axs[1,0].get_position() # get the original position 
pos20 = axs[2,0].get_position() # get the original position
pos30 = axs[3,0].get_position() # get the original position 
pos40 = axs[4,0].get_position() # get the original position 
pos01 = axs[0,1].get_position() # get the original position
pos11 = axs[1,1].get_position() # get the original position
pos21 = axs[2,1].get_position() # get the original position
pos31 = axs[3,1].get_position() # get the original position
pos41 = axs[4,1].get_position() # get the original position
pos02 = axs[0,2].get_position() # get the original position
pos12 = axs[1,2].get_position() # get the original position 
pos22 = axs[2,2].get_position() # get the original position
pos32 = axs[3,2].get_position() # get the original position 
pos42 = axs[4,2].get_position() # get the original position 
pos03 = axs[0,3].get_position() # get the original position
pos13 = axs[1,3].get_position() # get the original position
pos23 = axs[2,3].get_position() # get the original position
pos33 = axs[3,3].get_position() # get the original position
pos43 = axs[4,3].get_position() # get the original position

dp = 0.015

pos00.x0 -= dp  # or use: pos00 = [pos00.x0 + 0.1, pos00.y0 ,  pos00.width, pos00.height] 
pos10.x0 -= dp
pos20.x0 -= dp
pos30.x0 -= dp
pos40.x0 -= dp
pos01.x0 -= dp
pos11.x0 -= dp
pos21.x0 -= dp
pos31.x0 -= dp
pos41.x0 -= dp

pos00.x1 -= dp  # or use: pos00 = [pos00.x0 + 0.1, pos00.y0 ,  pos00.width, pos00.height] 
pos10.x1 -= dp
pos20.x1 -= dp
pos30.x1 -= dp
pos40.x1 -= dp
pos01.x1 -= dp
pos11.x1 -= dp
pos21.x1 -= dp
pos31.x1 -= dp
pos41.x1 -= dp

pos02.x0 += dp  # or use: pos00 = [pos00.x0 + 0.1, pos00.y0 ,  pos00.width, pos00.height] 
pos12.x0 += dp
pos22.x0 += dp
pos32.x0 += dp
pos42.x0 += dp
pos03.x0 += dp
pos13.x0 += dp
pos23.x0 += dp
pos33.x0 += dp
pos43.x0 += dp

pos02.x1 += dp  # or use: pos00 = [pos00.x0 + 0.1, pos00.y0 ,  pos00.width, pos00.height] 
pos12.x1 += dp
pos22.x1 += dp
pos32.x1 += dp
pos42.x1 += dp
pos03.x1 += dp
pos13.x1 += dp
pos23.x1 += dp
pos33.x1 += dp
pos43.x1 += dp

axs[0,0].set_position(pos00) # set a new position
axs[1,0].set_position(pos10) # set a new position
axs[2,0].set_position(pos20) # set a new position
axs[3,0].set_position(pos30) # set a new position
axs[4,0].set_position(pos40) # set a new position
axs[0,1].set_position(pos01) # set a new position
axs[1,1].set_position(pos11) # set a new position
axs[2,1].set_position(pos21) # set a new position
axs[3,1].set_position(pos31) # set a new position
axs[4,1].set_position(pos41) # set a new position
axs[0,2].set_position(pos02) # set a new position
axs[1,2].set_position(pos12) # set a new position
axs[2,2].set_position(pos22) # set a new position
axs[3,2].set_position(pos32) # set a new position
axs[4,2].set_position(pos42) # set a new position
axs[0,3].set_position(pos03) # set a new position
axs[1,3].set_position(pos13) # set a new position
axs[2,3].set_position(pos23) # set a new position
axs[3,3].set_position(pos33) # set a new position
axs[4,3].set_position(pos43) # set a new position



ix0 = 2
ix1 = 8
ix2 = 15
ix3 = 22
ix4 = 28

iy0 = 2
iy1 = 8
iy2 = 15
iy3 = 22
iy4 = 18

# ylim1 = -2.5
# ylim2 = 4.2

axs[0,0].fill_between(xs_plot, PMD_data_g_Grid[iy0,:]-data_g_error_Grid[iy0,:], PMD_data_g_Grid[iy0,:]+data_g_error_Grid[iy0,:],facecolor='0.5')
axs[0,0].plot(xs_plot,dg_obs_Grid[iy0,:], 'k.',linewidth=2) #row=0, col=0
# axs[0,0].set_xlabel('Distance (km)',fontweight="normal", fontsize = 8)
# axs[0,0].set_ylabel('Gravity (mGal)',fontweight="normal", fontsize = 8)
axs[0,0].text(0,yg_lim2-3,'(a)', fontweight="normal", fontsize = fs)
axs[0,0].set_xticklabels([])
axs[0,0].set_ylim([yg_lim1, yg_lim2])
axs[0,0].yaxis.tick_left()
axs[0,0].tick_params(axis="x",direction="in")
# plt.setp(axs[0,0], xticks=[660, 680, 700])
# plt.setp(axs[0,0], yticks=[2.5, 5, 7.5, 10], yticklabels=['2.5', '5', '7.5', '10'])
# axs[0,0].tick_params(axis='both', labelsize=15)

axs[1,0].fill_between(xs_plot, PMD_data_g_Grid[iy1,:]-data_g_error_Grid[iy1,:], PMD_data_g_Grid[iy1,:]+data_g_error_Grid[iy1,:],facecolor='0.5')
axs[1,0].plot(xs_plot,dg_obs_Grid[iy1,:], 'k.',linewidth=2) #row=0, col=0
# axs[0,0].set_xlabel('Distance (km)',fontweight="normal", fontsize = 8)
# axs[1,0].set_ylabel('Gravity (mGal)',fontweight="normal", fontsize = 8)
axs[1,0].text(0,yg_lim2-3,'(b)', fontweight="normal", fontsize = fs)
axs[1,0].set_xticklabels([])
axs[1,0].yaxis.tick_left()
axs[1,0].tick_params(axis="x",direction="in")
axs[1,0].set_ylim([yg_lim1, yg_lim2])
# plt.setp(axs[1,0], xticks=[660, 680, 700])

axs[2,0].fill_between(xs_plot, PMD_data_g_Grid[iy2,:]-data_g_error_Grid[iy2,:], PMD_data_g_Grid[iy2,:]+data_g_error_Grid[iy2,:],facecolor='0.5')
axs[2,0].plot(xs_plot,dg_obs_Grid[iy2,:], 'k.',linewidth=2) #row=0, col=0
# axs[0,0].set_xlabel('Distance (km)',fontweight="normal", fontsize = 8)
# axs[2,0].set_ylabel('Gravity (mGal)'',fontweight="normal", fontsize = 8)
axs[2,0].text(0,yg_lim2-3,'(c)', fontweight="normal", fontsize = fs)
axs[2,0].set_xticklabels([])
axs[2,0].yaxis.tick_left()
axs[2,0].tick_params(axis="x",direction="in")
axs[2,0].set_ylim([yg_lim1, yg_lim2])
axs[2,0].text(-0.4,0,'Gravity (mGal)',transform=axs[2,0].transAxes, fontweight="normal", fontsize = fs, rotation = 90)

# plt.setp(axs[2,0], xticks=[660, 680, 700])

axs[3,0].fill_between(xs_plot, PMD_data_g_Grid[iy3,:]-data_g_error_Grid[iy3,:], PMD_data_g_Grid[iy3,:]+data_g_error_Grid[iy3,:],facecolor='0.5')
axs[3,0].plot(xs_plot,dg_obs_Grid[iy3,:], 'k.',linewidth=2) #row=0, col=0
# axs[0,0].set_xlabel('Distance (km)',fontweight="normal", fontsize = 8)
# axs[3,0].set_ylabel('Gravity (mGal)',fontweight="normal", fontsize = 8)
axs[3,0].text(0,yg_lim2-3,'(d)', fontweight="normal", fontsize = fs)
axs[3,0].set_xticklabels([])
axs[3,0].yaxis.tick_left()
axs[3,0].tick_params(axis="x",direction="in")
axs[3,0].set_ylim([yg_lim1, yg_lim2])
# plt.setp(axs[3,0], xticks=[660, 680, 700])

axs[4,0].fill_between(xs_plot, PMD_data_g_Grid[iy4,:]-data_g_error_Grid[iy4,:], PMD_data_g_Grid[iy4,:]+data_g_error_Grid[iy4,:],facecolor='0.5')
axs[4,0].plot(xs_plot,dg_obs_Grid[iy4,:], 'k.',linewidth=2) #row=0, col=0
axs[4,0].set_xlabel('Easting (km)',fontweight="normal", fontsize = fs)
# axs[4,0].set_ylabel('Gravity (mGal)',fontweight="normal", fontsize = 8)
axs[4,0].text(0,yg_lim2-3,'(e)', fontweight="normal", fontsize = fs)
# axs[4,0].set_xticklabels([])
axs[4,0].yaxis.tick_left()
axs[4,0].tick_params(axis="x",direction="in")
axs[4,0].set_ylim([yg_lim1, yg_lim2])
# plt.setp(axs[4,0], xticks=[660, 680, 700], xticklabels=['660', '680', '700'])
###########################################################
axs[0,1].fill_between(ys_plot, PMD_data_g_Grid[:,ix0]-data_g_error_Grid[:,ix0], PMD_data_g_Grid[:,ix0]+data_g_error_Grid[:,ix0],facecolor='0.5')
axs[0,1].plot(ys_plot,dg_obs_Grid[:,ix0], 'k.',linewidth=2) #row=0, col=0
# axs[0,0].set_xlabel('Distance (km)',fontweight="normal", fontsize = 8)
# axs[0,1].set_ylabel('Gravity (mGal)',fontweight="normal", fontsize = 8)
axs[0,1].text(0,yg_lim2-3,'(f)', fontweight="normal", fontsize = fs)
axs[0,1].set_xticklabels([])
axs[0,1].yaxis.tick_left()
axs[0,1].tick_params(axis="x",direction="in")
# plt.setp(axs[0,0], yticks=[2.5, 5, 7.5, 10], yticklabels=['2.5', '5', '7.5', '10'])
# axs[0,0].tick_params(axis='both', labelsize=15)
axs[0,1].set_ylim([yg_lim1, yg_lim2])
axs[0,1].set_yticklabels([])
# plt.setp(axs[0,1], xticks=[2730, 2710, 2690])

axs[1,1].fill_between(ys_plot, PMD_data_g_Grid[:,ix1]-data_g_error_Grid[:,ix1], PMD_data_g_Grid[:,ix1]+data_g_error_Grid[:,ix1],facecolor='0.5')
axs[1,1].plot(ys_plot,dg_obs_Grid[:,ix1], 'k.',linewidth=2) #row=0, col=0
# axs[0,0].set_xlabel('Distance (km)',fontweight="normal", fontsize = 8)
# axs[1,1].set_ylabel('Gravity (mGal)',fontweight="normal", fontsize = 8)
axs[1,1].text(0,yg_lim2-3,'(g)', fontweight="normal", fontsize = fs)
axs[1,1].set_xticklabels([])
axs[1,1].yaxis.tick_left()
axs[1,1].tick_params(axis="x",direction="in")
axs[1,1].set_ylim([yg_lim1, yg_lim2])
axs[1,1].set_yticklabels([])
# plt.setp(axs[1,1], xticks=[2730, 2710, 2690])

axs[2,1].fill_between(ys_plot, PMD_data_g_Grid[:,ix2]-data_g_error_Grid[:,ix2], PMD_data_g_Grid[:,ix2]+data_g_error_Grid[:,ix2],facecolor='0.5')
axs[2,1].plot(ys_plot,dg_obs_Grid[:,ix2], 'k.',linewidth=2) #row=0, col=0
# axs[0,0].set_xlabel('Distance (km)',fontweight="normal", fontsize = 8)
# axs[2,1].set_ylabel('Gravity (mGal)',fontweight="normal", fontsize = 8)
axs[2,1].text(0,yg_lim2-3,'(h)', fontweight="normal", fontsize = fs)
axs[2,1].set_xticklabels([])
axs[2,1].yaxis.tick_left()
axs[2,1].tick_params(axis="x",direction="in")
axs[2,1].set_ylim([yg_lim1, yg_lim2])
axs[2,1].set_yticklabels([])
# plt.setp(axs[2,1], xticks=[2730, 2710, 2690])

axs[3,1].fill_between(ys_plot, PMD_data_g_Grid[:,ix3]-data_g_error_Grid[:,ix3], PMD_data_g_Grid[:,ix3]+data_g_error_Grid[:,ix3],facecolor='0.5')
axs[3,1].plot(ys_plot,dg_obs_Grid[:,ix3], 'k.',linewidth=2) #row=0, col=0
# axs[0,0].set_xlabel('Distance (km)',fontweight="normal", fontsize = 8)
# axs[3,1].set_ylabel('Gravity (mGal)',fontweight="normal", fontsize = 8)
axs[3,1].text(0,yg_lim2-3,'(i)', fontweight="normal", fontsize = fs)
axs[3,1].set_xticklabels([])
axs[3,1].yaxis.tick_left()
axs[3,1].tick_params(axis="x",direction="in")
axs[3,1].set_ylim([yg_lim1, yg_lim2])
axs[3,1].set_yticklabels([])
# plt.setp(axs[3,1], xticks=[2730, 2710, 2690])

axs[4,1].fill_between(ys_plot, PMD_data_g_Grid[:,ix4]-data_g_error_Grid[:,ix4], PMD_data_g_Grid[:,ix4]+data_g_error_Grid[:,ix4],facecolor='0.5')
axs[4,1].plot(ys_plot,dg_obs_Grid[:,ix4], 'k.',linewidth=2) #row=0, col=0
# axs[0,0].set_xlabel('Distance (km)',fontweight="normal", fontsize = 8)
# axs[4,1].set_ylabel('Gravity (mGal)',fontweight="normal", fontsize = 8)
axs[4,1].text(0,yg_lim2-3,'(j)', fontweight="normal", fontsize = fs)
# axs[4,1].set_xticklabels([])
axs[4,1].yaxis.tick_left()
axs[4,1].tick_params(axis="x",direction="in")
axs[4,1].set_ylim([yg_lim1, yg_lim2])
axs[4,1].set_yticklabels([])
axs[4,1].set_xlabel('Northing (km)',fontweight="normal", fontsize = fs)
# plt.setp(axs[4,1], xticks=[2730, 2710, 2690], xticklabels=['2730', '2710', '2690'])
####################################################################################

# ylim1 = -25
# ylim2 = 130

axs[0,2].fill_between(xs_plot, PMD_data_T_Grid[iy0,:]-data_T_error_Grid[iy0,:], PMD_data_T_Grid[iy0,:]+data_T_error_Grid[iy0,:],facecolor='0.5')
axs[0,2].plot(xs_plot,dT_obs_Grid[iy0,:], 'k.',linewidth=2) #row=0, col=0
# axs[0,0].set_xlabel('Distance (km)',fontweight="normal", fontsize = 8)
# axs[0,2].set_ylabel('Gravity (mGal)',fontweight="normal", fontsize = 8)
axs[0,2].text(0,yT_lim2-30,'(k)', fontweight="normal", fontsize = fs)
axs[0,2].set_xticklabels([])
axs[0,2].set_ylim([yT_lim1, yT_lim2])
# axs[0,0].set_xlim([-2.5, 3])
axs[0,2].yaxis.tick_left()
axs[0,2].tick_params(axis="x",direction="in")
axs[0,2].yaxis.set_label_position("right")
# axs[0,2].yaxis.tick_right()
axs[0,2].set_yticklabels([])
# plt.setp(axs[0,2], xticks=[660, 680, 700])
plt.setp(axs[0,2], yticks=[0, 50, 100])
# plt.setp(axs[0,0], yticks=[2.5, 5, 7.5, 10], yticklabels=['2.5', '5', '7.5', '10'])
# axs[0,0].tick_params(axis='both', labelsize=15)

axs[1,2].fill_between(xs_plot, PMD_data_T_Grid[iy1,:]-data_T_error_Grid[iy1,:], PMD_data_T_Grid[iy1,:]+data_T_error_Grid[iy1,:],facecolor='0.5')
axs[1,2].plot(xs_plot,dT_obs_Grid[iy1,:], 'k.',linewidth=2) #row=0, col=0
# axs[0,0].set_xlabel('Distance (km)',fontweight="normal", fontsize = 8)
# axs[1,2].set_ylabel('Gravity (mGal)',fontweight="normal", fontsize = 8)
axs[1,2].text(0,yT_lim2-30,'(l)', fontweight="normal", fontsize = fs)
axs[1,2].set_xticklabels([])
axs[1,2].yaxis.tick_left()
axs[1,2].tick_params(axis="x",direction="in")
axs[1,2].set_ylim([yT_lim1, yT_lim2])
axs[1,2].yaxis.set_label_position("right")
# axs[1,2].yaxis.tick_right()
axs[1,2].set_yticklabels([])
# plt.setp(axs[1,2], xticks=[660, 680, 700])
plt.setp(axs[1,2], yticks=[0, 50, 100])

axs[2,2].fill_between(xs_plot, PMD_data_T_Grid[iy2,:]-data_T_error_Grid[iy2,:], PMD_data_T_Grid[iy2,:]+data_T_error_Grid[iy2,:],facecolor='0.5')
axs[2,2].plot(xs_plot,dT_obs_Grid[iy2,:], 'k.',linewidth=2) #row=0, col=0
# axs[0,0].set_xlabel('Distance (km)',fontweight="normal", fontsize = 8)
# axs[2,2].set_ylabel('Gravity (mGal)',fontweight="normal", fontsize = 8)
axs[2,2].text(0,yT_lim2-30,'(m)', fontweight="normal", fontsize = fs)
axs[2,2].set_xticklabels([])
axs[2,2].yaxis.tick_left()
axs[2,2].tick_params(axis="x",direction="in")
axs[2,2].set_ylim([yT_lim1, yT_lim2])
axs[2,2].yaxis.set_label_position("right")
# axs[2,2].yaxis.tick_right()
axs[2,2].set_yticklabels([])
# plt.setp(axs[2,2], xticks=[660, 680, 700])
plt.setp(axs[2,2], yticks=[0, 50, 100])

axs[3,2].fill_between(xs_plot, PMD_data_T_Grid[iy3,:]-data_T_error_Grid[iy3,:], PMD_data_T_Grid[iy3,:]+data_T_error_Grid[iy3,:],facecolor='0.5')
axs[3,2].plot(xs_plot,dT_obs_Grid[iy3,:], 'k.',linewidth=2) #row=0, col=0
# axs[0,0].set_xlabel('Distance (km)',fontweight="normal", fontsize = 8)
# axs[3,2].set_ylabel('Gravity (mGal)',fontweight="normal", fontsize = 8)
axs[3,2].text(0,yT_lim2-30,'(n)', fontweight="normal", fontsize = fs)
axs[3,2].set_xticklabels([])
axs[3,2].yaxis.tick_left()
axs[3,2].tick_params(axis="x",direction="in")
axs[3,2].set_ylim([yT_lim1, yT_lim2])
axs[3,2].yaxis.set_label_position("right")
# axs[3,2].yaxis.tick_right()
axs[3,2].set_yticklabels([])
# plt.setp(axs[3,2], xticks=[660, 680, 700])
plt.setp(axs[3,2], yticks=[0, 50, 100])

axs[4,2].fill_between(xs_plot, PMD_data_T_Grid[iy4,:]-data_T_error_Grid[iy4,:], PMD_data_T_Grid[iy4,:]+data_T_error_Grid[iy4,:],facecolor='0.5')
axs[4,2].plot(xs_plot,dT_obs_Grid[iy4,:], 'k.',linewidth=2) #row=0, col=0
# axs[0,0].set_xlabel('Distance (km)',fontweight="normal", fontsize = 8)
# axs[4,2].set_ylabel('Gravity (mGal)',fontweight="normal", fontsize = 8)
axs[4,2].text(0,yT_lim2-30,'(o)', fontweight="normal", fontsize = fs)
# axs[4,2].set_xticklabels([])
axs[4,2].yaxis.tick_left()
axs[4,2].tick_params(axis="x",direction="in")
axs[4,2].set_ylim([yT_lim1, yT_lim2])
axs[4,2].set_xlabel('Easting (km)',fontweight="normal", fontsize = fs)
axs[4,2].yaxis.set_label_position("right")
# axs[4,2].yaxis.tick_right()
axs[4,2].set_yticklabels([])
# plt.setp(axs[4,2], xticks=[660, 680, 700], xticklabels=['660', '680', '700'])
plt.setp(axs[4,2], yticks=[0, 50, 100])
###########################################################
axs[0,3].fill_between(ys_plot, PMD_data_T_Grid[:,ix0]-data_T_error_Grid[:,ix0], PMD_data_T_Grid[:,ix0]+data_T_error_Grid[:,ix0],facecolor='0.5')
axs[0,3].plot(ys_plot,dT_obs_Grid[:,ix0], 'k.',linewidth=2) #row=0, col=0
# axs[0,0].set_xlabel('Distance (km)',fontweight="normal", fontsize = 8)
# axs[0,3].set_ylabel('Gravity (mGal)',fontweight="normal", fontsize = 8)
axs[0,3].text(0,yT_lim2-30,'(p)', fontweight="normal", fontsize = fs)
axs[0,3].set_xticklabels([])
# axs[0,3].yaxis.tick_left()
axs[0,3].tick_params(axis="x",direction="in")
axs[0,3].set_ylim([yT_lim1, yT_lim2])
# axs[0,3].set_yticklabels([])
axs[0,3].yaxis.set_label_position("right")
axs[0,3].yaxis.tick_right()
# plt.setp(axs[0,3], xticks=[2730, 2710, 2690])
plt.setp(axs[0,3], yticks=[0, 50, 100], yticklabels=['0', '50', '100'])
# plt.setp(axs[0,0], yticks=[2.5, 5, 7.5, 10], yticklabels=['2.5', '5', '7.5', '10'])
# axs[0,0].tick_params(axis='both', labelsize=15)


axs[1,3].fill_between(ys_plot, PMD_data_T_Grid[:,ix1]-data_T_error_Grid[:,ix1], PMD_data_T_Grid[:,ix1]+data_T_error_Grid[:,ix1],facecolor='0.5')
axs[1,3].plot(ys_plot,dT_obs_Grid[:,ix1], 'k.',linewidth=2) #row=0, col=0
# axs[0,0].set_xlabel('Distance (km)',fontweight="normal", fontsize = 8)
# axs[1,3].set_ylabel('Gravity (mGal)',fontweight="normal", fontsize = 8)
axs[1,3].text(0,yT_lim2-30,'(q)', fontweight="normal", fontsize = fs)
axs[1,3].set_xticklabels([])
# axs[1,3].yaxis.tick_left()
axs[1,3].tick_params(axis="x",direction="in")
axs[1,3].set_ylim([yT_lim1, yT_lim2])
# axs[1,3].set_yticklabels([])
axs[1,3].yaxis.set_label_position("right")
axs[1,3].yaxis.tick_right()
# plt.setp(axs[1,3], xticks=[2730, 2710, 2690])
plt.setp(axs[1,3], yticks=[0, 50, 100], yticklabels=['0', '50', '100'])

axs[2,3].fill_between(ys_plot, PMD_data_T_Grid[:,ix2]-data_T_error_Grid[:,ix2], PMD_data_T_Grid[:,ix2]+data_T_error_Grid[:,ix2],facecolor='0.5')
axs[2,3].plot(ys_plot,dT_obs_Grid[:,ix2], 'k.',linewidth=2) #row=0, col=0
# axs[0,0].set_xlabel('Distance (km)',fontweight="normal", fontsize = 8)
# axs[2,3].set_ylabel('Gravity (mGal)',fontweight="normal", fontsize = 8)
axs[2,3].text(0,yT_lim2-30,'(r)', fontweight="normal", fontsize = fs)
axs[2,3].set_xticklabels([])
# axs[2,3].yaxis.tick_left()
axs[2,3].tick_params(axis="x",direction="in")
axs[2,3].set_ylim([yT_lim1, yT_lim2])
# axs[2,3].set_yticklabels([])
axs[2,3].yaxis.set_label_position("right")
axs[2,3].yaxis.tick_right()
axs[2,3].text(1.25,0,'DRTP (nT)', transform=axs[2,3].transAxes, fontweight="normal", fontsize = fs, rotation = 90)
# plt.setp(axs[2,3], xticks=[2730, 2710, 2690])
plt.setp(axs[2,3], yticks=[0, 50, 100], yticklabels=['0', '50', '100'])

axs[3,3].fill_between(ys_plot, PMD_data_T_Grid[:,ix3]-data_T_error_Grid[:,ix3], PMD_data_T_Grid[:,ix3]+data_T_error_Grid[:,ix3],facecolor='0.5')
axs[3,3].plot(ys_plot,dT_obs_Grid[:,ix3], 'k.',linewidth=2) #row=0, col=0
# axs[0,0].set_xlabel('Distance (km)',fontweight="normal", fontsize = 8)
# axs[3,3].set_ylabel('Gravity (mGal)',fontweight="normal", fontsize = 8)
axs[3,3].text(0,yT_lim2-30,'(s)', fontweight="normal", fontsize = fs)
axs[3,3].set_xticklabels([])
# axs[3,3].yaxis.tick_left()
axs[3,3].tick_params(axis="x",direction="in")
axs[3,3].set_ylim([yT_lim1, yT_lim2])
# axs[3,3].set_yticklabels([])
axs[3,3].yaxis.set_label_position("right")
axs[3,3].yaxis.tick_right()
# plt.setp(axs[3,3], xticks=[2730, 2710, 2690])
plt.setp(axs[3,3], yticks=[0, 50, 100], yticklabels=['0', '50', '100'])

axs[4,3].fill_between(ys_plot, PMD_data_T_Grid[:,ix4]-data_T_error_Grid[:,ix4], PMD_data_T_Grid[:,ix4]+data_T_error_Grid[:,ix4],facecolor='0.5')
axs[4,3].plot(ys_plot,dT_obs_Grid[:,ix4], 'k.',linewidth=2) #row=0, col=0
# axs[0,0].set_xlabel('Distance (km)',fontweight="normal", fontsize = 8)
# axs[4,3].set_ylabel('Gravity (mGal)',fontweight="normal", fontsize = 8)
axs[4,3].text(0,yT_lim2-30,'(t)', fontweight="normal", fontsize = fs)
# axs[4,3].set_xticklabels([])
# axs[4,3].yaxis.tick_left()
axs[4,3].tick_params(axis="x",direction="in")
axs[4,3].set_ylim([yT_lim1, yT_lim2])
axs[4,3].set_xlabel('Northing (km)',fontweight="normal", fontsize = fs)
axs[4,3].yaxis.set_label_position("right")
axs[4,3].yaxis.tick_right()
# plt.setp(axs[4,3], xticks=[2730, 2710, 2690], xticklabels=['2730', '2710', '2690'])
plt.setp(axs[4,3], yticks=[0, 50, 100], yticklabels=['0', '50', '100'])
# axs[4,3].set_yticklabels([])



plt.show()
figname = 'Data_Slice'
fig.savefig(fpath_plots+'/'+figname+'.pdf')
plt.close(fig)    # close the figure window

#########################################################################
### PLOT 2D ACF


[x, y, z, rho, k_AR, XYZLine, ARg, ART] = Chain2xyz(ChainAll[-1])
ARg_2D = ARg.reshape((4,4))
ART_2D = ART.reshape((4,4))
print('ARg_2D: ', ARg_2D)
print('ART_2D: ', ART_2D)

[LogL, model_vec_g, model_vec_T, rg, rT, sigma_g, sigma_T, uncor_g, uncor_T] = Log_Likelihood(Kernel_Grv,Kernel_Mag,dg_obs,dT_obs,x,y,z,rho,ARg_2D,ART_2D,XnYnZn)

rg_2D = rg.reshape((Ndata,Ndata))
rT_2D = rT.reshape((Ndata,Ndata))
#######################################################################
AC_rg = signal.correlate2d(rg_2D/np.std(rg_2D), rg_2D/np.std(rg_2D))
AC_rg = AC_rg/rg_2D.size
AC_ug = signal.correlate2d(uncor_g/np.std(uncor_g), uncor_g/np.std(uncor_g))
AC_ug = AC_ug/uncor_g.size

AC_rT = signal.correlate2d(rT_2D/np.std(rT_2D), rT_2D/np.std(rT_2D))
AC_rT = AC_rT/rT_2D.size
AC_uT = signal.correlate2d(uncor_T/np.std(uncor_T), uncor_T/np.std(uncor_T))
AC_uT = AC_uT/uncor_T.size
#################################################################################
fpath_bin = os.getcwd()+'//'+file_name+'//'+'BinFiles'
Chain_raw = np.load(fpath_bin+'//'+'RAW_1.npy')

[x, y, z, rho, k_AR, XYZLine, ARg, ART] = Chain2xyz(Chain_raw[-1])
ARg_2D = ARg.reshape((4,4))
ART_2D = ART.reshape((4,4))
print('ARg_raw: ', ARg_2D)
print('ART_raw: ', ART_2D)

[LogL, model_vec_g, model_vec_T, rg, rT, sigma_g, sigma_T, uncor_g, uncor_T] = Log_Likelihood(Kernel_Grv,Kernel_Mag,dg_obs,dT_obs,x,y,z,rho,ARg_2D,ART_2D,XnYnZn)

rg_raw = rg.reshape((Ndata,Ndata))
rT_raw = rT.reshape((Ndata,Ndata))
##################################################################################
AC_rg_raw = signal.correlate2d(rg_raw/np.std(rg_raw), rg_raw/np.std(rg_raw))
AC_rg_raw = AC_rg_raw/rg_raw.size

AC_rT_raw = signal.correlate2d(rT_raw/np.std(rT_raw), rT_raw/np.std(rT_raw))
AC_rT_raw = AC_rT_raw/rT_raw.size


# fig, axs = plt.subplots(2, 1)
# plt.rc('font', weight='normal')
# plt.rc('xtick', labelsize=10)
# plt.rc('ytick', labelsize=10)

fig = plt.figure(figsize=(10,6))
ax1 = fig.add_subplot(2, 2, 1, projection='3d')

# im0 = axs[0].imshow(AC_rg,cmap='Greys')

x_lag = np.linspace(-31, 31, 63)
y_lag = np.linspace(-31, 31, 63)

X_lag, Y_lag = np.meshgrid(x_lag, y_lag)

surf1 = ax1.plot_trisurf(X_lag.flatten(), Y_lag.flatten(), AC_rg.flatten(), cmap="jet", vmin = 0, vmax = 1)
# fig.colorbar(surf1, ax=ax1, shrink=0.5, aspect=5)
ax1.elev = 5

print('rg: ', np.amax(AC_rg))
print('ug: ', np.amax(AC_ug))

ax2 = fig.add_subplot(2, 2, 2, projection='3d')
surf2 = ax2.plot_trisurf(X_lag.flatten(), Y_lag.flatten(), AC_ug.flatten(), cmap="jet", vmin = 0, vmax = 1)
ax2.elev = 5

ax3 = fig.add_subplot(2, 2, 3)
#ax3.plot(y_lag,AC_rg[31,:],color='black',ls='-')
ax3.plot(y_lag,AC_rg_raw[31,:],color='black',ls='-')
ax3.plot(y_lag,AC_ug[31,:],color='black',ls='--')


ax4 = fig.add_subplot(2, 2, 4)
ax4.plot(y_lag,AC_rT_raw[31,:],color='black',ls='-')
ax4.plot(y_lag,AC_uT[31,:],color='black',ls='--')
# axe[0].set_ylabel('Log Likelihood',fontweight="normal", fontsize = 10)
# axe[0].set_xlabel('(a) rjMCMC STEP',fontweight="normal", fontsize = 10)

# axs[0].xaxis.set_label_position("top")
# axs[0].xaxis.tick_top()
# axe[0].get_xaxis().get_major_formatter().set_scientific(False)

# axs[1].imshow(AC_ug,cmap='Greys')
# axe[1].set_ylabel('pdf',fontweight="normal", fontsize = 10)
# axe[1].set_xlabel('(b) Number of Nodes',fontweight="normal", fontsize = 10)

# axe[1].yaxis.set_label_position("right")
# axe[1].yaxis.tick_right()
plt.show()

cbar_pos = fig.add_axes([0.1, 0.6, 0.02, 0.3]) 
cbar = plt.colorbar(surf1, ax=ax1 ,shrink=0.3, cax = cbar_pos,
                    orientation='vertical', ticklocation = 'left')

figname = 'ACF'
fig.savefig(fpath_plots+'/'+figname+'.pdf')
plt.close(fig)    # close the figure window

#########################################
# rg_2D_m = rg_2D-np.mean(rg_2D)            #subtract mean
# rg_2D_m = rg_2D_m/np.sqrt(np.sum(np.power(rg_2D_m, 2))) #normalize magnitude
#########################################
# f_rg = np.fft.fft2(rg_2D)
# AC_rg = np.fft.ifft2(np.abs(f_rg) ** 2)
# AC_rg = abs(np.fft.fftshift(AC_rg))

# f_ug = np.fft.fft2(uncor_g)
# AC_ug = np.fft.ifft2(np.abs(f_ug) ** 2)
# AC_ug = abs(np.fft.fftshift(AC_ug))
###########################################################

# AC_rg = signal.fftconvolve(rg_2D, rg_2D[::-1, ::-1])
# AC_ug = signal.fftconvolve(uncor_g, uncor_g[::-1, ::-1])
#########################################

######################################################################################

PMD_AR = np.zeros((4,4))
PMD_AR[0,1] = np.mean(AR3_x_g[:,0])
PMD_AR[0,2] = np.mean(AR3_x_g[:,1])
PMD_AR[0,3] = np.mean(AR3_x_g[:,2])
PMD_AR[1,0] = np.mean(AR3_y_g[:,0])
PMD_AR[2,0] = np.mean(AR3_y_g[:,1])
PMD_AR[3,0] = np.mean(AR3_y_g[:,2])
PMD_AR[1,1] = np.mean(AR3_xy_g[:,0])
PMD_AR[2,2] = np.mean(AR3_xy_g[:,1])
PMD_AR[3,3] = np.mean(AR3_xy_g[:,2])

# print(PMD_AR)
# print(0, np.mean(AR3_x_g[:,0]), np.mean(AR3_x_g[:,1]), np.mean(AR3_x_g[:,2]))
# print(np.mean(AR3_y_g[:,0]), np.mean(AR3_xy_g[:,0]),0, 0)
# print(np.mean(AR3_y_g[:,1]), 0, np.mean(AR3_xy_g[:,1]),0)
# print(np.mean(AR3_y_g[:,2]), 0, 0, np.mean(AR3_xy_g[:,2]))

# print(AR_parameters_original_g)
fig, axs = plt.subplots(1, 2)
plt.rc('font', weight='normal')
plt.rc('xtick', labelsize=10)
plt.rc('ytick', labelsize=10)

im0 = axs[0].imshow(AR_parameters_original_g,interpolation='none',
       vmin=-1, vmax=1, aspect='auto', cmap='jet')

x_txt = np.array([0,1,2,3])
y_txt = np.array([0,1,2,3])
X_txt, Y_txt = np.meshgrid(x_txt, y_txt)

for i in np.arange(4):
    for j in np.arange(4):
        axs[0].text(X_txt[i,j],Y_txt[i,j],np.array2string(AR_parameters_original_g[i,j],precision=3), fontweight="normal", fontsize = 10)
        plt.show()

im1 = axs[1].imshow(PMD_AR,interpolation='none',
       vmin=-1, vmax=1, aspect='auto', cmap='jet')

for i in np.arange(4):
    for j in np.arange(4):
        axs[1].text(X_txt[i,j],Y_txt[i,j],np.array2string(PMD_AR[i,j],precision=3), fontweight="normal", fontsize = 10)
        plt.show()

plt.show()

figname = 'PMD_AR'
fig.savefig(fpath_plots+'/'+figname+'.pdf')
plt.close(fig)    # close the figure window
