#%%
import numpy as np
import matplotlib.pyplot as plt
import os
import glob
from pathlib import Path
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['font.size'] = 12

# Function to extract RMDV reversal segments from forward-to-reversal transitions
def extract_reversal_RMDV(folder, dt, before_s=10):
    """
    Extract RMDV segments during forward-to-reversal transitions.
    Returns RMDV segments aligned to reversal start (time=0).
    """
    forwards = np.loadtxt(folder + '/forward.csv', delimiter=',')
    if len(forwards.shape) == 1:
        forwards = forwards.reshape(1,2)
    forwards = forwards.astype(np.int32)
    reversals = np.loadtxt(folder + '/reversal.csv', delimiter=',')
    if len(reversals.shape) == 1:
        reversals = reversals.reshape(1,2)
    reversals = reversals.astype(np.int32)
    RMDV = np.loadtxt(folder + '/RMDV.csv', delimiter=',')
    
    RMDV_seg_fw_bw_ls = []
    time_fw_bw_ls = []
    
    for reversal in reversals:
        # Find forward interval before the reversal
        try:
            forward = forwards[np.where(forwards[:,1] <= reversal[0])][-1]
        except:
            continue
        start = max(forward[0], reversal[0] - int(before_s//dt))
        zero = reversal[0]  # Reversal start
        end = zero + min(int(before_s//dt), reversal[1] - reversal[0])
        time = np.arange(start, end) * dt
        time = time - time[zero - start]  # Align to reversal start (time=0)
        # RMDV_seg = (RMDV[start:end] - RMDV[zero])/(1+RMDV[zero])  # Normalize to reversal start
        RMDV_seg = (RMDV[start:end] - RMDV[zero])  # Normalize to reversal start
        
        RMDV_seg_fw_bw_ls.append(RMDV_seg)
        time_fw_bw_ls.append(time)
    
    return RMDV_seg_fw_bw_ls, time_fw_bw_ls

# Function to calculate mean and std with truncation (from RMDDV_fw_bw_align.py)
def calculate_stats_with_truncation(time_series_list, time_list, truncate_length=100, dt=0.1):
    """
    Calculate mean and standard deviation across multiple time series with truncation.
    Handles variable-length time series aligned to a common time axis.
    """
    # Initialize arrays to store sums and counts
    sum_array = np.zeros(2*truncate_length+1)
    sum_squares = np.zeros(2*truncate_length+1)
    count_array = np.zeros(2*truncate_length+1)
    
    # Process each time series
    for ts, time in zip(time_series_list, time_list):
        ts = np.array(ts)
        
        # Determine the effective index range for this time series
        effective_index_start = max(0, int(round(time[0]/dt)) + truncate_length)
        effective_index_end = min(2*truncate_length+1, int(round(time[-1]/dt)) + 1 + truncate_length)
        
        # Add values to the sum and count arrays
        if effective_index_end > effective_index_start:
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

# Load all RMDV.csv files from RMDV_gcamp8f folder
RMDV_trials = []
folder_ls = []

root = './RMDV_gcamp8f'

# Find all RMDV.csv files recursively
for root_dir, dirs, files in os.walk(root):
    if 'RMDV.csv' in files:
        csv_path = os.path.join(root_dir, 'RMDV.csv')
        try:
            data = np.loadtxt(csv_path, delimiter=',')
            RMDV_trials.append(data)
            folder_ls.append(csv_path)
            print(f"Loaded: {csv_path} (length: {len(data)})")
        except Exception as e:
            print(f"Error loading {csv_path}: {e}")

print(f"\nTotal trials loaded: {len(RMDV_trials)}")

# Normalize trials (subtract first value)
RMDV_normalized = [(trial - trial[0])/(trial[0]+1) for trial in RMDV_trials]

# Set manual length for calculating mean/std
target_length = 250  # Can be changed manually
dt = 0.04  # Time step in seconds
print(f"Target length: {target_length} frames ({target_length * dt:.2f} seconds)")

# Pad shorter trials with NaN to match target length, truncate longer ones
RMDV_aligned = []
for trial in RMDV_normalized:
    padded_trial = np.full(target_length, np.nan)
    actual_length = min(len(trial), target_length)
    padded_trial[:actual_length] = trial[:actual_length]
    RMDV_aligned.append(padded_trial)
RMDV_aligned = np.array(RMDV_aligned)

# Calculate mean and std ignoring NaN values
RMDV_mean = np.nanmean(RMDV_aligned, axis=0)
RMDV_std = np.nanstd(RMDV_aligned, axis=0)

# Create time axis in seconds
time_axis = np.arange(target_length) * dt

# Extract RMDV reversal segments from RMDDV folders
RMDV_reversal_segments = []
RMDV_reversal_times = []

# Find all RMDDV folders with forward.csv, reversal.csv, and RMDV.csv
folder_ls_reversal = ['RMDDV/20240424_50fps_5frames/stk2',
                       'RMDDV/20240424_50fps_5frames/stk3']
folder_ls_reversal.extend(glob.glob('RMDDV/RMD_DV_new/stk*'))

dt_reversal = 1/50 * 5  # Same as in RMDDV_fw_bw_align.py

for folder in folder_ls_reversal:
    if os.path.exists(folder + '/forward.csv') and os.path.exists(folder + '/reversal.csv') and os.path.exists(folder + '/RMDV.csv'):
        try:
            segs, times = extract_reversal_RMDV(folder, dt_reversal, before_s=10)
            RMDV_reversal_segments.extend(segs)
            RMDV_reversal_times.extend(times)
            print(f"Extracted {len(segs)} reversal segments from {folder}")
        except Exception as e:
            print(f"Error processing {folder}: {e}")

print(f"\nTotal reversal segments extracted: {len(RMDV_reversal_segments)}")

# Filter to keep only reversal portion (time >= 0)
RMDV_reversal_only_segments = []
RMDV_reversal_only_times = []
for seg, time in zip(RMDV_reversal_segments, RMDV_reversal_times):
    mask = time >= 0
    if np.any(mask):
        RMDV_reversal_only_segments.append(seg[mask])
        RMDV_reversal_only_times.append(time[mask])

print(f"Filtered to reversal-only segments: {len(RMDV_reversal_only_segments)}")

# Calculate mean and std for reversal segments (only reversal portion)
if len(RMDV_reversal_only_segments) > 0:
    # Find max length for reversal portion
    max_reversal_length = max(len(seg) for seg in RMDV_reversal_only_segments)
    truncate_length = max_reversal_length  # Use actual max length
    
    # Pad shorter segments with NaN
    RMDV_reversal_aligned = []
    for seg in RMDV_reversal_only_segments:
        padded_seg = np.full(truncate_length, np.nan)
        padded_seg[:len(seg)] = seg
        RMDV_reversal_aligned.append(padded_seg)
    RMDV_reversal_aligned = np.array(RMDV_reversal_aligned)
    
    # Calculate mean and std ignoring NaN
    RMDV_reversal_mean = np.nanmean(RMDV_reversal_aligned, axis=0)
    RMDV_reversal_std = np.nanstd(RMDV_reversal_aligned, axis=0)
    
    # Create time axis starting from 0
    dt_reversal_actual = RMDV_reversal_only_times[0][1] - RMDV_reversal_only_times[0][0] if len(RMDV_reversal_only_times[0]) > 1 else 0.1
    time_reversal = np.arange(len(RMDV_reversal_mean)) * dt_reversal_actual

# Plot all trials together
plt.figure(figsize=(3, 3),dpi=300)
# for i, trial in enumerate(RMDV_normalized):
#     trial_time = np.arange(len(trial)) * dt
#     plt.plot(trial_time, trial, alpha=0.3, linewidth=0.8, color='k')

# for i, trial in enumerate(RMDV_reversal_only_segments):
#     trial_time = np.arange(len(trial)) * dt_reversal
#     plt.plot(trial_time, trial, alpha=0.3, linewidth=0.8, color='r')
# Plot mean and std for gcamp8f data
plt.plot(time_axis, RMDV_mean, 'k-', linewidth=2, label='Mean (GCaMP8f)')
plt.fill_between(time_axis, RMDV_mean - RMDV_std, RMDV_mean + RMDV_std, 
                 alpha=0.3, color='gray', label='Mean ± Std (GCaMP8f)')

# Plot mean and std for reversal segments (only reversal portion)
if len(RMDV_reversal_only_segments) > 0:
    plt.plot(time_reversal, RMDV_reversal_mean, 
             'r-', linewidth=2, label='Mean (wNEMOS)')
    plt.fill_between(time_reversal, 
                     RMDV_reversal_mean - RMDV_reversal_std,
                     RMDV_reversal_mean + RMDV_reversal_std,
                     alpha=0.3, color='red', label='Mean ± Std (wNEMOS)')

plt.xlabel('Time (s)')
plt.ylabel('RMDV Signal (normalized)')
plt.title('RMDV Trials: GCaMP8f and wNEMOS')
plt.ylim(bottom=-0.2)
plt.legend(bbox_to_anchor=(0.05, 1), loc='upper left')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('RMDV_wNEMOS_gCamp8f_mean_std.pdf')

# Calculate difference between last and first frame for each normalized trial
# For gcamp8f dataset
# Calculate only for trials with duration > 5s

# For gcamp8f dataset
gcamp8f_differences = [
    (trial[-1] - trial[0])/(len(trial)*dt)
    for trial in RMDV_normalized
    # if (len(trial) * dt) > 5.0
]

# For reversal dataset
reversal_differences = [
    (seg[-1] - seg[0])/(len(seg)*dt_reversal)
    for seg in RMDV_reversal_only_segments
    # if (len(seg) * dt_reversal) > 5.0
]


from scipy.stats import mannwhitneyu
import WormTool
p = mannwhitneyu(reversal_differences, gcamp8f_differences)[1]
print(f'len wNEMOs: {len(reversal_differences)}, len GCaMP8f: {len(gcamp8f_differences)}, p={p}')
fig,ax = plt.subplots(1,1,figsize=(3,3),dpi=300)
WormTool.statistic_plot.plot_whisker_box([reversal_differences, gcamp8f_differences], ['wNEMOs','GCaMP8f'], 'Ca2+ change', label_rot=45, ax=ax)
# remove x tick
# ax.set_xticks([])
plt.savefig('RMDV_gcamp8f_Ca2_change_compare.pdf')
plt.show()



# %%
len_wnemos = [len(seg) * dt_reversal for seg in RMDV_reversal_only_segments]
len_gcamp8f = [len(trial) * dt for trial in RMDV_normalized]
fig,ax = plt.subplots(1,1,figsize=(3,3),dpi=300)
WormTool.statistic_plot.plot_whisker_box([len_wnemos, len_gcamp8f], ['wNEMOs','GCaMP7f'], 'Reversal duration (s)', label_rot=45, ax=ax)
plt.show()
# %%