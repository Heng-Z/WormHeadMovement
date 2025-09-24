#%%
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal
from functions import *

def plot_curv_Ca2(curv1, curv2, Ca2, dt, save_path=None, smooth_window_s=1):
    smooth_window = round(int(smooth_window_s/dt)/6)*6
    Ca2_1st_der = smooth_1st_derivative(Ca2,window_size=smooth_window)
    Ca2_2nd_der = smooth_2nd_derivative(Ca2,window_size=smooth_window)
    curv1_1st_der = smooth_1st_derivative(curv1,window_size=smooth_window)
    fig,ax = plt.subplots(3,1,figsize=(5,9),dpi=300)
    t = np.arange(len(curv1)) * dt
    
    # First subplot: curv1, curv2, and Ca2
    ax[0].plot(t, curv1, 'k', label='Curvature 12 points')
    ax[0].plot(t, curv2, 'k--', label='Curvature 20 points')
    twin0 = ax[0].twinx()
    twin0.plot(t, Ca2, 'r', label='Ca2')
    
    # Red axis and ticks for the first subplot
    twin0.spines['right'].set_color('red')
    twin0.yaxis.label.set_color('red')
    twin0.tick_params(axis='y', colors='red')
    ax[0].legend()
    ax[0].set_xlabel('time (s)')
    twin0.set_ylabel('RMDL/R')
    ax[0].set_ylabel('Curvature (1/L)')
    ax[0].set_title('Curvature and RMDL/R Ca2+')
    twin0.spines['right'].set_visible(True)
    
    # Second subplot: curv1_1st_der and Ca2_2nd_der
    ax[1].plot(t, curv1_1st_der, 'k', label='1st derivative of Curv 12pt')
    twin1 = ax[1].twinx()
    twin1.plot(t, Ca2_2nd_der * 100, 'm', label='2nd derivative of Ca2')
    
    # Red axis and ticks for the second subplot
    twin1.spines['right'].set_color('m')
    twin1.yaxis.label.set_color('m')
    twin1.tick_params(axis='y', colors='m')
    # ax[1].legend()
    ax[1].set_xlabel('time (s)')
    twin1.set_ylabel('2nd Derivative of Ca2')
    ax[1].set_ylabel('1st Derivative of Curvature')
    ax[1].set_title('Derivatives of Curvature and Ca2')
    twin1.spines['right'].set_visible(True)
    
    # ax[2] plot curv1_1st_der and Ca2_1st_der in twin axes
    ax[2].plot(t, curv1_1st_der, 'k', label='1st derivative of Curv 12pt')
    twin2 = ax[2].twinx()
    twin2.plot(t, Ca2_1st_der, 'm', label='1st derivative of Ca2')
    twin2.spines['right'].set_color('m')
    twin2.yaxis.label.set_color('m')
    twin2.tick_params(axis='y', colors='m')
    twin2.set_ylabel('1st Derivative of Ca2')
    ax[2].set_ylabel('1st Derivative of Curvature')
    ax[2].set_title('Derivatives of Curvature and Ca2')
    twin2.spines['right'].set_visible(True)
    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path, dpi=300)
    plt.show()


#%%
curv_Ca2_1st_corr_pool = []
curv_Ca2_2nd_corr_pool = []
curv_Ca2_1st_lag_pool = []
curv_Ca2_2nd_lag_pool = []

# %%
corr_window_s = 3 # 3 seconds window for correlation
smooth_window_s = 2 # 1 second window for smoothing
stk1_dr = np.loadtxt('data/stk1_dr_r0.csv', delimiter=',')
stk1_curv12 = np.loadtxt('data/stk1_12pts_curv_smooth.csv')
stk1_curv20 = np.loadtxt('data/stk1_20pts_curv_smooth.csv')
plot_curv_Ca2(stk1_curv12, stk1_curv20, stk1_dr, dt=1/30,save_path='data/stk1_curv_Ca2.png',smooth_window_s=smooth_window_s)

curv_Ca2_1st_corr_ls, curv_Ca2_2nd_corr_ls, curv_Ca2_1st_lag_ls, curv_Ca2_2nd_lag_ls = curv_Ca2_correlation(stk1_curv12, stk1_dr, dt=1/30, corr_window_s=corr_window_s,smooth_window_s=smooth_window_s)
curv_Ca2_1st_corr_pool.extend(curv_Ca2_1st_corr_ls)
curv_Ca2_2nd_corr_pool.extend(curv_Ca2_2nd_corr_ls)
curv_Ca2_1st_lag_pool.extend(curv_Ca2_1st_lag_ls)
curv_Ca2_2nd_lag_pool.extend(curv_Ca2_2nd_lag_ls)
# %%
corr_window_s = 3 # 3 seconds window for correlation
smooth_window_s = 2 # 1 second window for smoothing
stk2_dr = np.loadtxt('data/stk2_dr_r0.csv', delimiter=',')
stk2_curv12 = np.loadtxt('data/stk2_12pts_curv_smooth.csv')
stk2_curv20 = np.loadtxt('data/stk2_20pts_curv_smooth.csv')
plot_curv_Ca2(stk2_curv12, stk2_curv20, stk2_dr, dt=1/30, save_path='data/stk2_curv_Ca2.png',smooth_window_s=smooth_window_s)
curv_Ca2_1st_corr_ls, curv_Ca2_2nd_corr_ls, curv_Ca2_1st_lag_ls, curv_Ca2_2nd_lag_ls = curv_Ca2_correlation(stk2_curv12, stk2_dr, dt=1/30, corr_window_s=corr_window_s,smooth_window_s=smooth_window_s)
curv_Ca2_1st_corr_pool.extend(curv_Ca2_1st_corr_ls)
curv_Ca2_2nd_corr_pool.extend(curv_Ca2_2nd_corr_ls)
curv_Ca2_1st_lag_pool.extend(curv_Ca2_1st_lag_ls)
curv_Ca2_2nd_lag_pool.extend(curv_Ca2_2nd_lag_ls)
# %%
corr_window_s = 10
smooth_window_s = 3
stk122_dr = np.loadtxt('data/stk122_dr_r0.csv', delimiter=',')[1]
stk122_curv12 = np.loadtxt('data/stk122_12pts_curv_smooth.csv')
stk122_curv20 = np.loadtxt('data/stk122_20pts_curv_smooth.csv')
plot_curv_Ca2(stk122_curv12, stk122_curv20, stk122_dr, dt=1/5, save_path='data/stk122_curv_Ca2.png',smooth_window_s=smooth_window_s)
curv_Ca2_1st_corr_ls, curv_Ca2_2nd_corr_ls, curv_Ca2_1st_lag_ls, curv_Ca2_2nd_lag_ls = curv_Ca2_correlation(stk122_curv12, stk122_dr, dt=1/5, corr_window_s=corr_window_s,smooth_window_s=smooth_window_s)
curv_Ca2_1st_corr_pool.extend(curv_Ca2_1st_corr_ls)
curv_Ca2_2nd_corr_pool.extend(curv_Ca2_2nd_corr_ls)
curv_Ca2_1st_lag_pool.extend(curv_Ca2_1st_lag_ls)
curv_Ca2_2nd_lag_pool.extend(curv_Ca2_2nd_lag_ls)
# %%
corr_window_s = 10
smooth_window_s = 3
stk345_dr = np.loadtxt('data/stk345_dr_r0.csv', delimiter=',')[1]
stk345_curv12 = np.loadtxt('data/stk345_12pts_curv_smooth.csv')
stk345_curv20 = np.loadtxt('data/stk345_20pts_curv_smooth.csv')
plot_curv_Ca2(stk345_curv12, stk345_curv20, stk345_dr, dt=1/5, save_path='data/stk345_curv_Ca2.png',smooth_window_s=smooth_window_s)
curv_Ca2_1st_corr_ls, curv_Ca2_2nd_corr_ls, curv_Ca2_1st_lag_ls, curv_Ca2_2nd_lag_ls = curv_Ca2_correlation(stk345_curv12, stk345_dr, dt=1/5, corr_window_s=corr_window_s,smooth_window_s=smooth_window_s)
curv_Ca2_1st_corr_pool.extend(curv_Ca2_1st_corr_ls)
curv_Ca2_2nd_corr_pool.extend(curv_Ca2_2nd_corr_ls)
curv_Ca2_1st_lag_pool.extend(curv_Ca2_1st_lag_ls)
curv_Ca2_2nd_lag_pool.extend(curv_Ca2_2nd_lag_ls)
# %%
fig,ax = plt.subplots(1,2,figsize=(4,3),dpi=300)
ax[0].hist(curv_Ca2_1st_corr_pool, bins=6, alpha=0.5)
ax[1].hist(curv_Ca2_2nd_corr_pool, bins=6, alpha=0.5)
ax[0].set_xlabel('Correlation Coefficient')
ax[1].set_xlabel('Correlation Coefficient')
ax[0].set_ylabel('Frequency')
ax[1].set_ylabel('Frequency')
ax[0].set_title('d[Ca2+]/dt and d[curvature]/dt')
ax[1].set_title('d^2[Ca2+]/dt^2 and d[curvature]/dt')
plt.tight_layout()
plt.show()
# %%
# Save the curv_Ca2_1st_corr_pool as csv file
np.savetxt('RMDLR_curv_corr_ls.csv', curv_Ca2_1st_corr_pool, delimiter=',')
# %%





