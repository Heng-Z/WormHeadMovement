#%%
import numpy as np
import matplotlib.pyplot as plt
import WormTool 
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['font.size'] = 12
# %%
folder = 'RMDDV/20240424_50fps_5frames/stk2'
smooth_window_s = 0.3
dt = 1/50*5
plot_start = int(0/dt)
plot_end = int(40/dt)
smooth_window = int(smooth_window_s/dt)
print('smooth_window: ',smooth_window)
curvature = np.loadtxt(folder + '/curvature.csv')
forwards = np.loadtxt(folder + '/forward.csv', delimiter=',') # (N,2) forward intervals
if len(forwards.shape) == 1:
    forwards = forwards.reshape(1,2)
reversals = np.loadtxt(folder + '/reversal.csv', delimiter=',') # (N,2) reversal intervals
if len(reversals.shape) == 1:
    reversals = reversals.reshape(1,2)
RMDD = np.loadtxt(folder + '/RMDD.csv', delimiter=',')
RMDV = np.loadtxt(folder + '/RMDV.csv', delimiter=',')
RMDLR = np.loadtxt(folder + '/RMD_LR.csv', delimiter=',')

RMDD_smooth = WormTool.timeseries.smooth_data(RMDD, window=smooth_window)
RMDV_smooth = WormTool.timeseries.smooth_data(RMDV, window=smooth_window)
RMDLR_smooth = WormTool.timeseries.smooth_data(RMDLR, window=smooth_window)

t = np.arange(len(curvature))*dt
fig,ax = plt.subplots(1,1,figsize=(3.5,1.6),dpi=300)
for start, end in forwards:
    ax.axvspan(start*dt, min(end*dt,plot_end*dt), color='k', alpha=0.2 , lw=0)
for start, end in reversals:
    ax.axvspan(start*dt, min(end*dt,plot_end*dt), color='red', alpha=0.2 , lw=0)
twin_ax = ax.twinx()
twin_ax.plot(t[plot_start:plot_end],curvature[plot_start:plot_end],'k',label='curvature')
twin_ax.set_ylim(-18,14)
twin_ax.set_ylabel('Curvature')
twin_ax.spines['right'].set_visible(True)
ax.plot(t[plot_start:plot_end],RMDD_smooth[plot_start:plot_end],'tab:blue',label='RMDD')
ax.plot(t[plot_start:plot_end],RMDV_smooth[plot_start:plot_end],'tab:orange',label='RMDV')
ax.plot(t[plot_start:plot_end],RMDLR_smooth[plot_start:plot_end],'tab:red',label='RMDLR')
ax.set_xlabel('Time (s)')
ax.set_ylabel('Ca2+')
# ax.legend()
ax.set_xlim(0,plot_end*dt)
# ax.set_ylim(-,1)
plt.savefig('RMDDV_backward_corr_example.pdf')


#%%
data = np.load('/Users/hengzhang/projects/WormSim/WormMimic2/data/mid_trajectory/20190811_1627_w4_chunk_002.npy',allow_pickle=True)
curvature = data[1]
head_curv = curvature[0:2,:].mean(axis=0)
t0 = 1000
t1 = 1600# len(head_curv)
head_curvature_smooth = WormTool.timeseries.smooth_data(head_curv, window=1)
t = np.arange(len(head_curv))*0.02
fig,ax = plt.subplots(1,1,figsize=(1.7,1.8),dpi=300)
ax.plot(t[t0:t1]-t[t0],head_curvature_smooth[t0:t1],'k',label='curvature')
ax.set_xlabel('Time (s)')
ax.set_ylabel('Curvature')
# ax.set_xlim(t0*dt,t1*dt)
ax.set_ylim(-18,14)
ax.axvspan(300*0.02,450*0.02,color='red', alpha=0.2 , lw=0)
plt.savefig('curv_cast_backward_corr_example.pdf')

# %%
