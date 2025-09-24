#%% Dependencies
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42
plt.rcParams['font.family'] = 'Arial'
import WormTool.timeseries
import os
import scipy.io
import importlib
importlib.reload(WormTool.timeseries)
import pickle

#%% PSi and PSi ctrl excursion statistics polar plot
name_ls = ['PSi','PSi_ctrl']
with open('excursion_trial_info_dict_all.pkl','rb') as f:
    group_info = pickle.load(f)
fig, ax = plt.subplots(1, 2, figsize=(15,7),dpi=300,subplot_kw=dict(projection='polar'))
for ii,name in enumerate(name_ls):
    excursion_trial_ls = group_info[name]
    excursion_amps = []
    excursion_phase = []
    excursion_rel_amps = []
    data_len_all = 0 
    for trial in excursion_trial_ls:
        if trial is None:
            continue
        excursion_amps.extend(trial['excursion_amps'])
        excursion_phase.extend(trial['excursion_phase'])
        excursion_rel_amps.extend(trial['excursion_rel_amps'])
        data_len_all += trial['length']
    phase_amp_density,phase_edge,amp_edge = np.histogram2d(excursion_phase,excursion_rel_amps,bins=[12,12],range=[[-np.pi,np.pi],[0,2]])
    phase_mesh, amp_mesh = np.meshgrid((phase_edge[1:]+phase_edge[:-1])/2,(amp_edge[1:]+amp_edge[:-1])/2)
    phase_amp_density_log = np.log((phase_amp_density+1)/data_len_all)
    pcm = ax[ii].pcolormesh(phase_mesh,amp_mesh, phase_amp_density_log.T, cmap='Reds',vmax=-6.5,vmin=-8.5)
    # add grid on 
    ax[ii].grid(True)
    # set the font size of degree labels
    ax[ii].tick_params(axis='x', labelsize=17)
    # set radius ticks
    # ax[ii].set_yticks([0.5,1,1.5,2])
    # ax[ii].set_yticklabels(['0.5','1','1.5','2'],fontsize=17)
    ax[ii].set_title(name,fontsize=20)
    cbar = fig.colorbar(pcm, ax=ax[ii])
plt.savefig('Fig4_psi_excursion_polar.pdf')
# %%
