#%%
import numpy as np
import matplotlib.pyplot as plt
import WormTool 
from functions import *
# folder = 'RMDDV/20240204_25fps_6frames'
# centerline = np.loadtxt(folder + '/centerline.csv', delimiter=',').reshape(186,100,2).transpose(1,2,0)
# curvature = WormTool.timeseries.centerline2curvature(centerline) # (100,T)
# head_curvature = curvature[:15].mean(axis=0)
# np.savetxt(folder + '/curvature.csv', head_curvature)
# %%
def plot_RMDDV_curvature(folder, dt, smooth_window_s=1):
    curvature = np.loadtxt(folder + '/curvature.csv')
    forwards = np.loadtxt(folder + '/forward.csv', delimiter=',') # (N,2) forward intervals
    if len(forwards.shape) == 1:
        forwards = forwards.reshape(1,2)
    reversals = np.loadtxt(folder + '/reversal.csv', delimiter=',') # (N,2) reversal intervals
    if len(reversals.shape) == 1:
        reversals = reversals.reshape(1,2)
    RMDD = np.loadtxt(folder + '/RMDD.csv', delimiter=',')
    RMDV = np.loadtxt(folder + '/RMDV.csv', delimiter=',')

    smooth_window = round(int(smooth_window_s/dt)/6)*6
    curv_1st_der = smooth_1st_derivative(curvature, window_size=smooth_window)
    RMDD_1st_der = smooth_1st_derivative(RMDD, window_size=smooth_window)
    RMDV_1st_der = smooth_1st_derivative(RMDV, window_size=smooth_window)
    RMDD_2nd_der = smooth_2nd_derivative(RMDD, window_size=smooth_window)
    RMDV_2nd_der = smooth_2nd_derivative(RMDV, window_size=smooth_window)

    t = np.arange(len(curvature))*dt
    fig,ax = plt.subplots(3,1,figsize=(8,12),dpi=300)
    ax[0].plot(t,curvature,'k',label='curvature')
    for start, end in forwards:
        ax[0].axvspan(start*dt, end*dt, color='green', alpha=0.3, label='Forward' , lw=0)
        ax[1].axvspan(start*dt, end*dt, color='green', alpha=0.3, label='Forward' , lw=0)
        ax[2].axvspan(start*dt, end*dt, color='green', alpha=0.3, label='Forward' , lw=0)
    for start, end in reversals:
        ax[0].axvspan(start*dt, end*dt, color='red', alpha=0.3, label='Reversal' , lw=0)
        ax[1].axvspan(start*dt, end*dt, color='red', alpha=0.3, label='Reversal' , lw=0)
        ax[2].axvspan(start*dt, end*dt, color='red', alpha=0.3, label='Reversal' , lw=0)
    twin_ax0 = ax[0].twinx()
    twin_ax0.plot(t,RMDD,'m',label='RMDD')
    twin_ax0.plot(t,RMDV,'c',label='RMDV')
    ax[0].set_xlabel('Time (s)')
    ax[0].set_ylabel('Curvature')
    twin_ax0.set_ylabel('Ca2+')
    twin_ax0.spines['right'].set_visible(True)
    twin_ax0.legend()
    ax[1].plot(t,curv_1st_der,'k')
    twin_ax1 = ax[1].twinx()
    twin_ax1.plot(t,RMDD_1st_der,'m',label='RMDD')
    twin_ax1.plot(t,RMDV_1st_der,'c',label='RMDV')
    ax[1].set_xlabel('Time (s)')
    ax[1].set_ylabel('Curvature 1st Derivative')
    twin_ax1.set_ylabel('Ca2+ 1st Derivative')
    twin_ax1.spines['right'].set_visible(True)
    twin_ax1.legend()

    ax[2].plot(t,curv_1st_der,'k')
    twin_ax2 = ax[2].twinx()
    twin_ax2.plot(t,RMDD_2nd_der,'m',label='RMDD')
    twin_ax2.plot(t,RMDV_2nd_der,'c',label='RMDV')
    ax[2].set_xlabel('Time (s)')
    ax[2].set_ylabel('Curvature 1st Derivative')
    twin_ax2.set_ylabel('Ca2+ 2nd Derivative')
    twin_ax2.spines['right'].set_visible(True)
    twin_ax2.legend()

    plt.tight_layout()
    plt.show()

def curv_ca2_seg_correlation(folder, dt, corr_window_s=3, smooth_window_s=1):
    curvature = np.loadtxt(folder + '/curvature.csv')
    RMDD = np.loadtxt(folder + '/RMDD.csv', delimiter=',')
    RMDV = np.loadtxt(folder + '/RMDV.csv', delimiter=',')
    forwards = np.loadtxt(folder + '/forward.csv', delimiter=',') # (N,2) forward intervals
    if len(forwards.shape) == 1:
        forwards = forwards.reshape(1,2)
    reversals = np.loadtxt(folder + '/reversal.csv', delimiter=',') # (N,2) reversal intervals
    if len(reversals.shape) == 1:
        reversals = reversals.reshape(1,2)
    RMDD_curv_corr_ls = []
    RMDV_curv_corr_ls = []
    segments = np.concatenate([forwards,reversals],axis=0).astype(np.int32)
    for start, end in segments:
        curv_Ca2_1st_corr_ls, curv_Ca2_2nd_corr_ls, _, _ = curv_Ca2_correlation(curvature[start:end], RMDD[start:end], dt, corr_window_s=corr_window_s,smooth_window_s=smooth_window_s,plot=True)
        RMDD_curv_corr_ls.extend(curv_Ca2_1st_corr_ls)
        curv_Ca2_1st_corr_ls, curv_Ca2_2nd_corr_ls, _, _ = curv_Ca2_correlation(curvature[start:end], RMDV[start:end], dt, corr_window_s=corr_window_s,smooth_window_s=smooth_window_s,plot=True)
        RMDV_curv_corr_ls.extend(curv_Ca2_1st_corr_ls)
    return RMDD_curv_corr_ls, RMDV_curv_corr_ls

#%%
RMDD_curv_corr_ls = []
RMDV_curv_corr_ls = []
#%%
folder = 'RMDDV/20240204_25fps_6frames'
smooth_window_s = 1
corr_window_s = 5
plot_RMDDV_curvature(folder, 1/25*6, smooth_window_s=smooth_window_s)
RMDD_curv_corr, RMDV_curv_corr = curv_ca2_seg_correlation(folder, 1/25*6, corr_window_s=corr_window_s, smooth_window_s=smooth_window_s)
RMDD_curv_corr_ls.extend(RMDD_curv_corr)
RMDV_curv_corr_ls.extend(RMDV_curv_corr)
# %%
folder = 'RMDDV/20240424_50fps_5frames/stk2'
smooth_window_s = 1
corr_window_s = 5
plot_RMDDV_curvature(folder, 1/50*5, smooth_window_s=smooth_window_s)
RMDD_curv_corr, RMDV_curv_corr = curv_ca2_seg_correlation(folder, 1/50*5, corr_window_s=corr_window_s, smooth_window_s=smooth_window_s)
RMDD_curv_corr_ls.extend(RMDD_curv_corr)
RMDV_curv_corr_ls.extend(RMDV_curv_corr)
 # %%
folder = 'RMDDV/20240424_50fps_5frames/stk3'
smooth_window_s = 1
corr_window_s = 5
plot_RMDDV_curvature(folder, 1/50*5, smooth_window_s=smooth_window_s)
RMDD_curv_corr, RMDV_curv_corr = curv_ca2_seg_correlation(folder, 1/50*5, corr_window_s=corr_window_s, smooth_window_s=smooth_window_s)
RMDD_curv_corr_ls.extend(RMDD_curv_corr)
RMDV_curv_corr_ls.extend(RMDV_curv_corr)

#%%
plt.hist(RMDD_curv_corr_ls, bins=6, alpha=0.5)
plt.hist(RMDV_curv_corr_ls, bins=6, alpha=0.5)
plt.show()
# %%
# Save the RMDD_curv_corr_ls as csv file
np.savetxt('RMDD_curv_corr_ls.csv', RMDD_curv_corr_ls, delimiter=',')
np.savetxt('RMDV_curv_corr_ls.csv', RMDV_curv_corr_ls, delimiter=',')
# %%
