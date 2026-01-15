#%%
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal
from functions import *
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['font.size'] = 12


# Load the RMDL/R data

curvature_trials = []
Ca2_L_trials = []
Ca2_R_trials = []

curvature_segments = []
Ca2_L_segments = []
Ca2_R_segments = []

folder_ls = []

root = './'

# folder = root + 'RMDLR/w1/stk1'
# curvature = np.loadtxt(folder + '/curvature.csv')
# Ca2_RMDL = np.loadtxt(folder + '/RMDL/RMDL_r130.csv', delimiter=',')
# Ca2_RMDR = np.loadtxt(folder + '/RMDR/RMDR_r130.csv', delimiter=',')
# curvature_trials.append(curvature)
# Ca2_L_trials.append(Ca2_RMDL)
# Ca2_R_trials.append(Ca2_RMDR)
# folder_ls.append(folder)
# segment_idx = [0,-1]
# for i in range(len(segment_idx)-1):
#     curvature_segments.append(curvature[segment_idx[i]:segment_idx[i+1]])
#     Ca2_L_segments.append(Ca2_RMDL[segment_idx[i]:segment_idx[i+1]])
#     Ca2_R_segments.append(Ca2_RMDR[segment_idx[i]:segment_idx[i+1]])

folder = root + 'RMDLR/w1/stk2'
curvature = np.loadtxt(folder + '/curvature.csv')*50
Ca2_RMDL = np.loadtxt(folder + '/RMDL/RMDL_g180.csv', delimiter=',')
Ca2_RMDR = np.loadtxt(folder + '/RMDR/RMDR.csv', delimiter=',')
curvature_trials.append(curvature)
Ca2_L_trials.append(Ca2_RMDL)
Ca2_R_trials.append(Ca2_RMDR)
folder_ls.append(folder)
segment_idx = [0,350,650,-1]
for i in range(len(segment_idx)-1):
    curvature_segments.append(curvature[segment_idx[i]:segment_idx[i+1]])
    Ca2_L_segments.append(Ca2_RMDL[segment_idx[i]:segment_idx[i+1]])
    Ca2_R_segments.append(Ca2_RMDR[segment_idx[i]:segment_idx[i+1]])

folder = root + 'RMDLR/w2/stk1'
curvature = np.loadtxt(folder + '/curvature.csv')*50
Ca2_RMDL = np.loadtxt(folder + '/RMDL/RMDL.csv', delimiter=',')
Ca2_RMDR = np.loadtxt(folder + '/RMDR/RMDR.csv', delimiter=',')
curvature_trials.append(curvature)
Ca2_L_trials.append(Ca2_RMDL)
Ca2_R_trials.append(Ca2_RMDR)
folder_ls.append(folder)
segment_idx = [0,130,320,430,-1]
for i in range(len(segment_idx)-1):
    curvature_segments.append(curvature[segment_idx[i]:segment_idx[i+1]])
    Ca2_L_segments.append(Ca2_RMDL[segment_idx[i]:segment_idx[i+1]])
    Ca2_R_segments.append(Ca2_RMDR[segment_idx[i]:segment_idx[i+1]])


folder = root + 'RMDLR/w2/stk2'
curvature = np.loadtxt(folder + '/curvature.csv')*50
Ca2_RMDL = np.loadtxt(folder + '/RMDL/RMDL.csv', delimiter=',')
Ca2_RMDR = np.loadtxt(folder + '/RMDR/RMDR.csv', delimiter=',')
curvature_trials.append(curvature)
Ca2_L_trials.append(Ca2_RMDL)
Ca2_R_trials.append(Ca2_RMDR)
folder_ls.append(folder)
segment_idx = [0,-1]
for i in range(len(segment_idx)-1):
    curvature_segments.append(curvature[segment_idx[i]:segment_idx[i+1]])
    Ca2_L_segments.append(Ca2_RMDL[segment_idx[i]:segment_idx[i+1]])
    Ca2_R_segments.append(Ca2_RMDR[segment_idx[i]:segment_idx[i+1]])


folder = root + 'RMDLR/w3/stk1'
curvature = np.loadtxt(folder + '/curvature.csv')*50
Ca2_RMDL = np.loadtxt(folder + '/RMDL/RMDL_g200.csv', delimiter=',')
Ca2_RMDR = np.loadtxt(folder + '/RMDR/RMDR_g200.csv', delimiter=',')
curvature_trials.append(curvature)
Ca2_L_trials.append(Ca2_RMDL)
Ca2_R_trials.append(Ca2_RMDR)
folder_ls.append(folder)
segment_idx = [0,135,-1]
for i in range(len(segment_idx)-1):
    curvature_segments.append(curvature[segment_idx[i]:segment_idx[i+1]])
    Ca2_L_segments.append(Ca2_RMDL[segment_idx[i]:segment_idx[i+1]])
    Ca2_R_segments.append(Ca2_RMDR[segment_idx[i]:segment_idx[i+1]])


folder = root + 'RMDLR/RMDLR_202509/RMDL'
curvature = np.loadtxt(folder + '/curvature.csv')*50
Ca2_RMDL = np.loadtxt(folder + '/RMDL.csv')
Ca2_RMDR = np.zeros_like(curvature)
curvature_trials.append(curvature)
Ca2_L_trials.append(Ca2_RMDL)
Ca2_R_trials.append(Ca2_RMDR)
folder_ls.append(folder)
segment_idx = [0,-1]
for i in range(len(segment_idx)-1):
    curvature_segments.append(curvature[segment_idx[i]:segment_idx[i+1]])
    Ca2_L_segments.append(Ca2_RMDL[segment_idx[i]:segment_idx[i+1]])
    Ca2_R_segments.append(Ca2_RMDR[segment_idx[i]:segment_idx[i+1]])

folder = root + 'RMDLR/RMDLR_202509/RMDR'
curvature = np.loadtxt(folder + '/curvature.csv')*50    
Ca2_RMDL = np.zeros_like(curvature)
Ca2_RMDR = np.loadtxt(folder + '/RMDR.csv')
curvature_trials.append(curvature)
Ca2_L_trials.append(Ca2_RMDL)
Ca2_R_trials.append(Ca2_RMDR)
folder_ls.append(folder)
segment_idx = [0,-1]
for i in range(len(segment_idx)-1):
    curvature_segments.append(curvature[segment_idx[i]:segment_idx[i+1]])
    Ca2_L_segments.append(Ca2_RMDL[segment_idx[i]:segment_idx[i+1]])
    Ca2_R_segments.append(Ca2_RMDR[segment_idx[i]:segment_idx[i+1]])


def plot_curv_Ca2(curvature, Ca2_L, Ca2_R, dt, save_path=None, smooth_window_s=1):
    smooth_window = round(int(smooth_window_s/dt)/6)*6
    Ca2_L_1st_der = smooth_1st_derivative(Ca2_L, window_size=smooth_window)
    Ca2_R_1st_der = smooth_1st_derivative(Ca2_R, window_size=smooth_window)
    curvature_1st_der = smooth_1st_derivative(curvature, window_size=smooth_window)
    
    fig, ax = plt.subplots(1, 2, figsize=(6, 2.6), dpi=300, sharex=True)
    t = np.arange(len(curvature)) * dt
    
    # First subplot: original curvature, RMDL and RMDR Ca2
    ax[0].plot(t, curvature, 'k', label='Curvature')
    twin0 = ax[0].twinx()
    twin0.plot(t, Ca2_L, 'tab:red', label='RMDL')
    twin0.plot(t, Ca2_R, 'tab:green', label='RMDR')
    
    ax[0].set_ylabel(r'Curvature $\kappa$ (1/L)')
    twin0.set_ylabel('Ca2+ (a.u.)')
    ax[0].set_title('Curvature and RMDL/R Ca2+ activity')
    ax[0].set_xlabel('Time (s)')
    twin0.legend(loc='upper right', fontsize=10)
    
    # Second subplot: 1st derivatives
    ax[1].plot(t, curvature_1st_der, 'k', label=r'$d\kappa/dt$', alpha=0.7)
    twin1 = ax[1].twinx()
    twin1.plot(t, Ca2_L_1st_der, 'r', label='dRMDL/dt', alpha=0.7)
    twin1.plot(t, Ca2_R_1st_der, 'b', label='dRMDR/dt', alpha=0.7)
    
    ax[1].set_ylabel(r'$d\kappa/dt$')
    twin1.set_ylabel('dCa2+/dt')
    ax[1].set_xlabel('Time (s)')
    ax[1].set_title('1st Derivatives')
    twin1.legend(loc='upper right', fontsize=6)
    
    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path)
    else:
        plt.show()

def band_pass_filter(signal, dt, cutoff_freq, order=4):
    """
    Apply Butterworth band pass filter to signal.
    cutoff_freq should be a tuple: (low_freq, high_freq)
    """
    fs = 1.0 / dt  # sampling frequency
    nyq = 0.5 * fs  # Nyquist frequency
    normalized_cutoff = [cutoff_freq[0] / nyq, cutoff_freq[1] / nyq]
    b, a = scipy.signal.butter(order, normalized_cutoff, btype='band')
    return scipy.signal.filtfilt(b, a, signal)



def plot_curv_Ca2_band_pass(curvature, Ca2_L, Ca2_R, dt, save_path=None, band_pass_freq=(0.01,1)):
    # Apply low pass filter to all signals
    curvature_filt = band_pass_filter(curvature, dt, band_pass_freq)
    Ca2_L_filt = band_pass_filter(Ca2_L, dt, band_pass_freq)
    Ca2_R_filt = band_pass_filter(Ca2_R, dt, band_pass_freq)
    
    # Calculate 1st derivatives as simple temporal difference
    curvature_1st_der = np.diff(curvature_filt) / dt
    Ca2_L_1st_der = np.diff(Ca2_L_filt) / dt
    Ca2_R_1st_der = np.diff(Ca2_R_filt) / dt
    
    fig, ax = plt.subplots(2, 1, figsize=(12, 4), dpi=300, sharex=True)
    t = np.arange(len(curvature)) * dt
    t_der = t[:-1] + dt / 2  # time points for derivatives (centered)
    
    # First subplot: low-pass filtered curvature, RMDL and RMDR Ca2
    ax[0].plot(curvature_filt, 'k', label='Curvature')
    twin0 = ax[0].twinx()
    twin0.plot(Ca2_L_filt, 'r', label='RMDL')
    twin0.plot(Ca2_R_filt, 'b', label='RMDR')
    
    ax[0].set_ylabel(r'Curvature $\kappa$ (1/L)')
    twin0.set_ylabel('Ca2+ (a.u.)')
    ax[0].set_title(f'Low-pass filtered ({band_pass_freq} Hz)')
    twin0.legend(loc='upper right', fontsize=6)
    
    # Second subplot: 1st derivatives
    ax[1].plot( curvature_1st_der, 'k', label=r'$d\kappa/dt$', alpha=0.7)
    twin1 = ax[1].twinx()
    twin1.plot( Ca2_L_1st_der, 'r', label='dRMDL/dt', alpha=0.7)
    twin1.plot( Ca2_R_1st_der, 'b', label='dRMDR/dt', alpha=0.7)
    
    ax[1].set_ylabel(r'$d\kappa/dt$')
    twin1.set_ylabel('dCa2+/dt')
    ax[1].set_xlabel('Time (s)')
    ax[1].set_title('1st Derivatives')
    twin1.legend(loc='upper right', fontsize=6)
    
    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path)
    else:
        plt.show()

    
def plot_all_RMDLR(band_pass_freq=(0.01,0.2)):
    dt = 0.2
    for i in range(len(curvature_trials)):
        curvature = curvature_trials[i]
        Ca2_RMDL = Ca2_L_trials[i]
        Ca2_RMDR = Ca2_R_trials[i]
        folder = folder_ls[i]
        plot_curv_Ca2(curvature, Ca2_RMDL, Ca2_RMDR, dt, save_path=folder + '/RMDL_R_plot.pdf')
        plot_curv_Ca2_band_pass(curvature, Ca2_RMDL, Ca2_RMDR, dt, save_path=folder + '/RMDL_R_plot_lp.png', band_pass_freq=band_pass_freq)

def plot_phase_space(dt=0.2,band_pass_freq=0.2):
    colors = ['tab:red','tab:blue','tab:green','tab:purple','tab:orange','tab:brown']
    for i,curvature in enumerate(curvature_trials):
        curvature_filt = band_pass_filter(curvature, dt, band_pass_freq)
        curvature_1st_der = np.diff(curvature_filt) / dt
        plt.plot(curvature_filt[1:],curvature_1st_der,color=colors[i],label=f'Dataset {i+1}')
    plt.legend()
    plt.xlabel('Curvature')
    plt.ylabel('Curvature 1st Derivative')
    plt.title('Phase Space Plot')
    plt.show()

# plot_phase_space()

  
plot_all_RMDLR(band_pass_freq=(0.01,0.2))

# %%

def cross_corr(s1,s2,range=(-200,200)):
    # calculate normalized correlation
    corr = scipy.signal.correlate(s1, s2, mode='full')
    corr = corr / (np.linalg.norm(s1) * np.linalg.norm(s2))
    lags = scipy.signal.correlation_lags(len(s1), len(s2), mode='full').astype(np.int32)
    valid_range = (lags >= range[0]) & (lags <= range[1])
    corr = corr[valid_range]
    lags = lags[valid_range]
    max_idx = np.argmax(corr)
    min_idx = np.argmin(corr)
    if abs(lags[max_idx]) > abs(lags[min_idx]):
        opt_idx = min_idx
    else:
        opt_idx = max_idx
    corr_max = corr[opt_idx]
    lag = lags[opt_idx]
    return corr_max,lag, corr, lags

def correlation(curvature, Ca2_L, Ca2_R, dt, save_path=None, band_pass_freq=1.0,order=(0,0)):
    curvature_filt = band_pass_filter(curvature, dt, band_pass_freq)
    Ca2_L_filt = band_pass_filter(Ca2_L, dt, band_pass_freq)
    Ca2_R_filt = band_pass_filter(Ca2_R, dt, band_pass_freq)
    
    # Calculate 1st derivatives as simple temporal difference
    curvature_1st_der = np.diff(curvature_filt) / dt
    Ca2_L_1st_der = np.diff(Ca2_L_filt) / dt
    Ca2_R_1st_der = np.diff(Ca2_R_filt) / dt


    a1 = order[0]
    a2 = order[1]
    curvature_signal = (1-a1)*curvature_filt[1:] + a1*curvature_1st_der
    Ca2_L_signal = (1-a2)*Ca2_L_filt[1:] + a2*Ca2_L_1st_der
    Ca2_R_signal = (1-a2)*Ca2_R_filt[1:] + a2*Ca2_R_1st_der

    corr_max_L,lag_L,corr_L,lags_L = cross_corr(curvature_signal, Ca2_L_signal, range=(-100,100))
    corr_max_R,lag_R,corr_R,lags_R = cross_corr(curvature_signal, Ca2_R_signal, range=(-100,100))
    return corr_max_L,lag_L,corr_L,lags_L,corr_max_R,lag_R,corr_R,lags_R

# def correlation_all(freq=1,dt=0.2,order=(0,1)):
if True:
    freq=(0.01,0.3);dt=0.2;order=(0,1)
    corr_max_L_ls = []
    corr_max_R_ls = []
    lag_L_ls = []
    lag_R_ls = []
    corr_L_ls = []
    corr_R_ls = []
    lags_L_ls = []
    lags_R_ls = []
    for i in  range(len(curvature_segments)):
        curvature = curvature_segments[i]
        Ca2_RMDL = Ca2_L_segments[i]
        Ca2_RMDR = Ca2_R_segments[i]
        corr_max_L,lag_L,corr_L,lags_L,corr_max_R,lag_R,corr_R,lags_R = correlation(curvature, Ca2_RMDL, Ca2_RMDR, dt, band_pass_freq=freq, order=order)
        if (np.std(Ca2_RMDL)) >1e-2:
            corr_max_L_ls.append(corr_max_L)
            lag_L_ls.append(lag_L)
            corr_L_ls.append(corr_L)
            lags_L_ls.append(lags_L)
        if (np.std(Ca2_RMDR)) >1e-2:
            corr_max_R_ls.append(corr_max_R)
            lag_R_ls.append(lag_R)
            corr_R_ls.append(corr_R)
            lags_R_ls.append(lags_R)

    mode1 = []
    mode2 = []
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6, 2.8), sharex=True)
    
    for i in range(len(corr_L_ls)):
        L = len(corr_L_ls[i])
        if corr_L_ls[i][L//2+10] > 0:
            ax1.plot(lags_L_ls[i]*dt, corr_L_ls[i], 'tab:red')
            mode1.append(corr_L_ls[i])
        else:
            ax1.plot(lags_L_ls[i]*dt, corr_L_ls[i], linestyle='--', color='tab:red')
            mode2.append(corr_L_ls[i])

    mode1 = np.array(mode1)
    mode2 = np.array(mode2)
    mode1_mean = np.mean(mode1, axis=0)
    mode2_mean = np.mean(mode2, axis=0)
    mode1_std = np.std(mode1, axis=0)
    mode2_std = np.std(mode2, axis=0)
    print(f'RMDL num. of mode1: {len(mode1)}, num. of mode2: {len(mode2)}')
    # Plot mean ± std in second subplot
    if len(mode1) > 0:
        ax2.plot(lags_L_ls[0]*dt, mode1_mean, 'k', label='Mode 1')
        ax2.fill_between(lags_L_ls[0]*dt, mode1_mean-mode1_std, mode1_mean+mode1_std, color='k', alpha=0.2)
    if len(mode2) > 0:
        ax2.plot(lags_L_ls[0]*dt, mode2_mean, 'k', linestyle='--', label='Mode 2')
        ax2.fill_between(lags_L_ls[0]*dt, mode2_mean-mode2_std, mode2_mean+mode2_std, color='k', alpha=0.2)
    
    ax1.set_ylabel('Correlation Coefficient')
    ax1.set_title('RMDL(derivative) vs Curvature - Individual Curves')
    ax2.set_xlabel('Lags (s)')
    ax2.set_ylabel('Correlation Coefficient')
    ax2.set_title('RMDL(derivative) vs Curvature - Mean ± Std')
    ax2.legend()
    plt.tight_layout()
    plt.savefig('RMDL_K_corr_mode1_mode2.pdf')
    plt.show()

    mode1_R = []
    mode2_R = []
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6, 2.8), sharex=True)
    
    for i in range(len(corr_R_ls)):
        L = len(corr_R_ls[i])
        if corr_R_ls[i][L//2+10] < 0:
            ax1.plot(lags_R_ls[i]*dt, corr_R_ls[i], 'tab:blue')
            mode1_R.append(corr_R_ls[i])
        else:
            ax1.plot(lags_R_ls[i]*dt, corr_R_ls[i], linestyle='--', color='tab:blue')
            mode2_R.append(corr_R_ls[i])

    mode1_R = np.array(mode1_R)
    mode2_R = np.array(mode2_R)
    mode1_R_mean = np.mean(mode1_R, axis=0)
    mode2_R_mean = np.mean(mode2_R, axis=0)
    mode1_R_std = np.std(mode1_R, axis=0)
    mode2_R_std = np.std(mode2_R, axis=0)
    print(f'RMDR num. of mode1: {len(mode1_R)}, num. of mode2: {len(mode2_R)}')
    # Plot mean ± std in second subplot
    if len(mode1_R) > 0:
        ax2.plot(lags_R_ls[0]*dt, mode1_R_mean, 'k', label='Mode 1')
        ax2.fill_between(lags_R_ls[0]*dt, mode1_R_mean-mode1_R_std, mode1_R_mean+mode1_R_std, color='k', alpha=0.2)
    if len(mode2_R) > 0:
        ax2.plot(lags_R_ls[0]*dt, mode2_R_mean, 'k', linestyle='--', label='Mode 2')
        ax2.fill_between(lags_R_ls[0]*dt, mode2_R_mean-mode2_R_std, mode2_R_mean+mode2_R_std, color='k', alpha=0.2)
    
    ax1.set_ylabel('Correlation Coefficient')
    ax1.set_title('RMDR(derivative) vs Curvature - Individual Curves')
    ax2.set_xlabel('Lags (s)')
    ax2.set_ylabel('Correlation Coefficient')
    ax2.set_title('RMDR(derivative) vs Curvature - Mean ± Std')
    ax2.legend()
    plt.tight_layout()
    plt.savefig('RMDR_K_corr_mode1_mode2.pdf')
    plt.show()


    # return corr_max_L_ls,corr_max_R_ls,lag_L_ls,lag_R_ls,corr_L_ls,corr_R_ls,lags_L_ls,lags_R_ls

# correlation_all(freq=(0.01,0.3),dt=0.2,order=(0,1))

#%% Phase space plot

