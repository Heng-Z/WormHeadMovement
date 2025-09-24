#%%
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal
from functions import *
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['font.size'] = 12

def plot_curv_Ca2(curv1, curv2, Ca2, dt, save_path=None, smooth_window_s=1):
    smooth_window = round(int(smooth_window_s/dt)/6)*6
    Ca2_1st_der = smooth_1st_derivative(Ca2,window_size=smooth_window)
    Ca2_2nd_der = smooth_2nd_derivative(Ca2,window_size=smooth_window)
    curv1_1st_der = smooth_1st_derivative(curv1,window_size=smooth_window)
    fig,ax = plt.subplots(2,1,figsize=(4,4),dpi=300)
    t = np.arange(len(curv1)) * dt
    
    # First subplot: curv1, curv2, and Ca2
    ax[0].plot(t, curv1, 'k', label='Curvature 12 points')
    # ax[0].plot(t, curv2, 'k--', label='Curvature 20 points')
    twin0 = ax[0].twinx()
    twin0.plot(t, Ca2, 'r', label='Ca2')
    
    # Red axis and ticks for the first subplot
    twin0.spines['right'].set_color('red')
    twin0.yaxis.label.set_color('red')
    twin0.tick_params(axis='y', colors='red')
    # ax[0].legend()
    twin0.set_ylabel('RMDL/R')
    ax[0].set_ylabel(r'Curvature $\kappa$ (1/L)')
    ax[0].set_title('Curvature and RMDL/R Ca2+ activity')
    twin0.spines['right'].set_visible(True)
    
    # ax[2] plot curv1_1st_der and Ca2_1st_der in twin axes
    ax[1].plot(t, curv1_1st_der, 'k', label='1st derivative of Curv 12pt',alpha=0.7)
    twin1 = ax[1].twinx()
    twin1.plot(t, Ca2_1st_der, 'm', label='1st derivative of Ca2')
    twin1.spines['right'].set_color('m')
    twin1.yaxis.label.set_color('m')
    twin1.tick_params(axis='y', colors='m')
    twin1.set_ylabel(r'$\frac{dz}{dt}$',rotation=0,fontsize=12)
    ax[1].set_ylabel(r'$\frac{d\kappa}{dt}$',rotation=0,fontsize=12)
    # ax[1].set_title('Derivatives of Curvature and Ca2')
    ax[1].set_xlabel('time (s)')
    twin1.set_ylim(-abs(np.max(Ca2_1st_der)),abs(np.max(Ca2_1st_der)))
    twin1.spines['right'].set_visible(True)
    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path)
    plt.show()

#%%
corr_window_s = 10
smooth_window_s = 1
stk345_dr = np.loadtxt('data/stk345_dr_r0.csv', delimiter=',')[1]
stk345_curv12 = np.loadtxt('data/stk345_12pts_curv_smooth.csv')*100
stk345_curv20 = np.loadtxt('data/stk345_20pts_curv_smooth.csv')*100
plot_curv_Ca2(stk345_curv12, stk345_curv20, stk345_dr, dt=1/5, save_path='RMDLR_curv_corr_example_plot.pdf',smooth_window_s=smooth_window_s)
# %%
corr_window_s = 3 # 3 seconds window for correlation
smooth_window_s = 1 # 1 second window for smoothing
stk1_dr = np.loadtxt('data/stk1_dr_r0.csv', delimiter=',')
stk1_curv12 = np.loadtxt('data/stk1_12pts_curv_smooth.csv')*100
stk1_curv20 = np.loadtxt('data/stk1_20pts_curv_smooth.csv')*100
plot_curv_Ca2(stk1_curv12, stk1_curv20, stk1_dr, dt=1/30, save_path='RMDLR_curv_corr_example_plot_stk1.pdf',smooth_window_s=smooth_window_s)
# %%
corr_window_s = 10 # 3 seconds window for correlation
smooth_window_s = 3 # 1 second window for smoothing
stk122_dr = np.loadtxt('data/stk122_dr_r0.csv', delimiter=',')[1]
stk122_curv12 = np.loadtxt('data/stk122_12pts_curv_smooth.csv')*100
stk122_curv20 = np.loadtxt('data/stk122_20pts_curv_smooth.csv')*100
plot_curv_Ca2(stk122_curv12, stk122_curv20, stk122_dr, dt=1/5, save_path='RMDLR_curv_corr_example_plot_stk122.svg',smooth_window_s=smooth_window_s)
# %%
