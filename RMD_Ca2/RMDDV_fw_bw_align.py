#%%
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['font.size'] = 12
import scipy.signal
import glob
#%%

def extract_reversal_Ca2(folder,dt,before_s=10):
    # curvature = np.loadtxt(folder + '/curvature.csv')
    forwards = np.loadtxt(folder + '/forward.csv', delimiter=',') # (N,2) forward intervals
    if len(forwards.shape) == 1:
        forwards = forwards.reshape(1,2)
    forwards = forwards.astype(np.int32)
    reversals = np.loadtxt(folder + '/reversal.csv', delimiter=',') # (N,2) reversal intervals
    if len(reversals.shape) == 1:
        reversals = reversals.reshape(1,2)
    reversals = reversals.astype(np.int32)
    RMDD = np.loadtxt(folder + '/RMDD.csv', delimiter=',')
    RMDV = np.loadtxt(folder + '/RMDV.csv', delimiter=',')
    RMDD_seg_fw_bw_ls = []
    RMDV_seg_fw_bw_ls = []
    time_fw_bw_ls = []
    RMDD_seg_bw_fw_ls = []
    RMDV_seg_bw_fw_ls = []
    time_bw_fw_ls = []

    print('forwards:',forwards)
    print('reversals:',reversals)

    for reversal in reversals:
        # find nested forward before the reversal
        try:
            forward = forwards[np.where(forwards[:,1] <= reversal[0])][-1]
        except:
            continue
        start = max(forward[0],reversal[0] - int(before_s//dt))
        zero = reversal[0]
        end = zero + min(int(before_s//dt), reversal[1] - reversal[0])
        print(start,zero,end)
        time = np.arange(start,end)*dt
        time = time - time[zero-start]
        RMDD_seg = RMDD[start:end] - RMDD[zero]
        RMDV_seg = RMDV[start:end] - RMDV[zero]

        RMDD_seg_fw_bw_ls.append(RMDD_seg)
        RMDV_seg_fw_bw_ls.append(RMDV_seg)
        time_fw_bw_ls.append(time)

    for reversal in reversals:
        # find nested forward before the reversal
        try:
            forward = forwards[np.where(forwards[:,0] >= reversal[1])][0]
        except:
            continue
        start = max(reversal[0],reversal[1] - int(before_s//dt))
        zero = reversal[1]
        end = zero + min(int(20//dt), forward[1] - forward[0])
        print(start,zero,end)
        time = np.arange(start,end)*dt
        time = time - time[zero-start]
        RMDD_seg = RMDD[start:end] - RMDD[zero]
        RMDV_seg = RMDV[start:end] - RMDV[zero]

        RMDD_seg_bw_fw_ls.append(RMDD_seg)
        RMDV_seg_bw_fw_ls.append(RMDV_seg)
        time_bw_fw_ls.append(time)

    return [RMDD_seg_fw_bw_ls, RMDV_seg_fw_bw_ls, time_fw_bw_ls, RMDD_seg_bw_fw_ls, RMDV_seg_bw_fw_ls, time_bw_fw_ls]

#%%
RMDD_fw_bw_ls = []
RMDV_fw_bw_ls = []
time_fw_bw_ls = []

RMDD_bw_fw_ls = []
RMDV_bw_fw_ls = []
time_bw_fw_ls = []

#%%
folder_ls = ['RMDDV/20240424_50fps_5frames/stk2',
             'RMDDV/20240424_50fps_5frames/stk3']

folder_ls.extend(glob.glob('RMDDV/RMD_DV_new/stk*'))

dt_ls = np.ones(len(folder_ls))*1/50*5
# dt_ls[0] = 1/25*6
#%%
for folder,dt in zip(folder_ls,dt_ls):
    results = extract_reversal_Ca2(folder, dt)
    RMDD_fw_bw_ls.extend(results[0])
    RMDV_fw_bw_ls.extend(results[1])
    time_fw_bw_ls.extend(results[2])
    RMDD_bw_fw_ls.extend(results[3])
    RMDV_bw_fw_ls.extend(results[4])
    time_bw_fw_ls.extend(results[5])


# %%
fig,ax = plt.subplots(2,1,figsize=(2,4),dpi=300)
for i, time in enumerate(time_fw_bw_ls):
    ax[0].plot(time,RMDD_fw_bw_ls[i],'k',alpha=0.2)
    ax[0].plot(time,RMDV_fw_bw_ls[i],'g',alpha=0.2)
ax[0].axvspan(-10,0,color='k',alpha=0.2,label='Forward',lw=0)
ax[0].axvspan(0,10,color='r',alpha=0.2,label='Reversal',lw=0)
for i, time in enumerate(time_bw_fw_ls):
    ax[1].plot(time,RMDD_bw_fw_ls[i],'k',alpha=0.2)
    ax[1].plot(time,RMDV_bw_fw_ls[i],'g',alpha=0.2)
ax[1].axvspan(0,10,color='k',alpha=0.2,label='Forward',lw=0)
ax[1].axvspan(-10,0,color='r',alpha=0.2,label='Reversal',lw=0)
ax[1].set_xlabel('Time (s)')
ax[1].set_ylabel('RMDD/V Ca2+')
ax[0].set_yticks([0,0.5,1,1.5])
# ax[1].set_yticks([0,0.5,1,1.5])
ax[0].set_xlim(-10,10)
ax[1].set_xlim(-10,10)
ax[0].legend()
plt.savefig('RMDDV_fw_bw_align.pdf')

#%% plot mean and std
def calculate_stats_with_truncation(time_series_list, time_list, truncate_length=100,dt=0.1):
    """
    Calculate mean and standard deviation across multiple time series with truncation.
    
    Parameters:
    time_series_list (list): List of 1D numpy arrays or lists, each representing a time series
    truncate_length (int): Maximum length to consider for calculations
    
    Returns:
    tuple: (mean_array, std_array) - numpy arrays of length truncate_length
    """
    # Initialize arrays to store sums and counts
    sum_array = np.zeros(2*truncate_length+1)
    sum_squares = np.zeros(2*truncate_length+1)
    count_array = np.zeros(2*truncate_length+1)
    
    # Process each time series
    for ts,time in zip(time_series_list,time_list):
        # Convert to numpy array if it's not already
        ts = np.array(ts)
        
        # Determine the effective length for this time series
        effective_index_start = max(0, int(round(time[0]/dt))+truncate_length)
        effective_index_end = min(2*truncate_length+1, int(round(time[-1]/dt))+1+truncate_length)
        assert effective_index_end - effective_index_start  == len(ts), f'{effective_index_end} - {effective_index_start}+1 != {len(ts)}'
        # Add values to the sum and count arrays
        sum_array[effective_index_start:effective_index_end] += ts
        sum_squares[effective_index_start:effective_index_end] += ts**2
        count_array[effective_index_start:effective_index_end] += 1
    
    # Avoid division by zero by setting counts of 0 to 1
    count_array[count_array == 0] = 1
    
    # Calculate mean
    mean_array = sum_array / count_array
    
    # Calculate variance: E[X^2] - (E[X])^2
    variance_array = (sum_squares / count_array) - (mean_array**2)
    
    # Calculate standard deviation
    std_array = np.sqrt(variance_array)
    
    return mean_array, std_array

RMDD_fw_bw_mean, RMDD_fw_bw_std = calculate_stats_with_truncation(RMDD_fw_bw_ls, time_fw_bw_ls, truncate_length=100)
RMDV_fw_bw_mean, RMDV_fw_bw_std = calculate_stats_with_truncation(RMDV_fw_bw_ls, time_fw_bw_ls, truncate_length=100)
RMDD_bw_fw_mean, RMDD_bw_fw_std = calculate_stats_with_truncation(RMDD_bw_fw_ls, time_bw_fw_ls, truncate_length=100)
RMDV_bw_fw_mean, RMDV_bw_fw_std = calculate_stats_with_truncation(RMDV_bw_fw_ls, time_bw_fw_ls, truncate_length=100)
time = np.arange(-100,100)*0.1
start = 2
end = len(time) - 2
fig,ax = plt.subplots(2,1,figsize=(2,4),dpi=300)
# for i, time in enumerate(time_fw_bw_ls):
#     ax[0].plot(time,RMDD_fw_bw_ls[i],'k',alpha=0.2)
#     ax[0].plot(time,RMDV_fw_bw_ls[i],'g',alpha=0.2)
color = ['tab:blue','tab:orange']
ax[0].axvspan(-10,0,color='k',alpha=0.2,label='Forward',lw=0)
ax[0].axvspan(0,10,color='r',alpha=0.2,label='Reversal',lw=0)
ax[0].plot(time[start:end],RMDD_fw_bw_mean[start:end],color[0],label='RMDD')
ax[0].fill_between(time[start:end],RMDD_fw_bw_mean[start:end]-RMDD_fw_bw_std[start:end],RMDD_fw_bw_mean[start:end]+RMDD_fw_bw_std[start:end],color=color[0],alpha=0.2)
ax[0].plot(time[start:end],RMDV_fw_bw_mean[start:end],color[1],label='RMDV')
ax[0].fill_between(time[start:end],RMDV_fw_bw_mean[start:end]-RMDV_fw_bw_std[start:end],RMDV_fw_bw_mean[start:end]+RMDV_fw_bw_std[start:end],color=color[1],alpha=0.2)
# for i, time in enumerate(time_bw_fw_ls):
#     ax[1].plot(time,RMDD_bw_fw_ls[i],'k',alpha=0.2)
#     ax[1].plot(time,RMDV_bw_fw_ls[i],'g',alpha=0.2)
ax[1].axvspan(0,10,color='k',alpha=0.2,label='Forward',lw=0)
ax[1].axvspan(-10,0,color='r',alpha=0.2,label='Reversal',lw=0)
ax[1].plot(time[start:end],RMDD_bw_fw_mean[start:end],color[0],label='RMDD')
ax[1].fill_between(time[start:end],RMDD_bw_fw_mean[start:end]-RMDD_bw_fw_std[start:end],RMDD_bw_fw_mean[start:end]+RMDD_bw_fw_std[start:end],color=color[0],alpha=0.2)
ax[1].plot(time[start:end],RMDV_bw_fw_mean[start:end],color[1],label='RMDV')
ax[1].fill_between(time[start:end],RMDV_bw_fw_mean[start:end]-RMDV_bw_fw_std[start:end],RMDV_bw_fw_mean[start:end]+RMDV_bw_fw_std[start:end],color=color[1],alpha=0.2)
ax[1].set_xlabel('Time (s)')
ax[1].set_ylabel('RMDD/V Ca2+')
ax[0].set_yticks([0,0.5,1,1.5])
# ax[1].set_yticks([0,0.5,1,1.5])
ax[0].set_xlim(-10,10)
ax[1].set_xlim(-10,10)
# ax[0].legend()
plt.savefig('RMDDV_fw_bw_align.pdf')

# %%
correlation_fw_bw_ls = []
lag_fw_bw_ls = []
for i in range(len(RMDD_fw_bw_ls)):
    s1 = RMDD_fw_bw_ls[i] - RMDD_fw_bw_ls[i].mean()
    s2 = RMDV_fw_bw_ls[i] - RMDV_fw_bw_ls[i].mean()
    correlation = scipy.signal.correlate(s1,s2,mode='same')
    lags = scipy.signal.correlation_lags(len(s1), len(s2), mode='same').astype(np.int32)
    correlation_fw_bw_ls.append(correlation/np.linalg.norm(s1)/np.linalg.norm(s2))
    dt = time_fw_bw_ls[i][1] - time_fw_bw_ls[i][0]
    lag_fw_bw_ls.append(lags*dt)

correlation_bw_fw_ls = []
lag_bw_fw_ls = []
for i in range(len(RMDD_bw_fw_ls)):
    s1 = RMDD_bw_fw_ls[i] - RMDD_bw_fw_ls[i].mean()
    s2 = RMDV_bw_fw_ls[i] - RMDV_bw_fw_ls[i].mean()
    correlation = scipy.signal.correlate(s1,s2,mode='same')
    lags = scipy.signal.correlation_lags(len(s1), len(s2), mode='same').astype(np.int32)
    correlation_bw_fw_ls.append(correlation/np.linalg.norm(s1)/np.linalg.norm(s2))
    dt = time_bw_fw_ls[i][1] - time_bw_fw_ls[i][0]
    lag_bw_fw_ls.append(lags*dt)

fig,ax = plt.subplots(1,1,figsize=(2,2),dpi=300)
for i, time in enumerate(time_fw_bw_ls):
    ax.plot(lag_fw_bw_ls[i],correlation_fw_bw_ls[i],'k',alpha=0.5)
for i, time in enumerate(time_bw_fw_ls):
    ax.plot(lag_bw_fw_ls[i],correlation_bw_fw_ls[i],'b',alpha=0.5)
ax.set_xlabel('Lags (s)')
ax.set_ylabel('RMDD * RMDV')
ax.set_xlim(-3,3)
plt.savefig('RMDDV_fw_bw_cross_corr_align.pdf')
plt.show()
# %%
corr_max_ls = [corr.max() for corr in correlation_fw_bw_ls]
print(np.argmax(corr_max_ls))

# %%
