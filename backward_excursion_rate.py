'''
pickle Data generation script: behaviors/backward_analysis.py
'''
#%%
import pickle
import numpy as np
import matplotlib.pyplot as plt

# Load data
name = 'RMDk'
with open(f'/Users/hengzhang/projects/WormSim/behaviors/backward_results/excursion_trial_info_dict_{name}_backward.pkl', 'rb') as f:
    bw_excursion_info_ls = pickle.load(f)

with open('/Users/hengzhang/projects/WormSim/figures/excursion_trial_info_dict_MN_ablation.pkl','rb') as f:
    forward_excursion_MN_info_dict = pickle.load(f)
fw_excursion_inf_ls = forward_excursion_MN_info_dict[name]

fw_excursion_rel_amps = np.concatenate([info['excursion_rel_amps'] for info in fw_excursion_inf_ls])
fw_total_len = np.sum(np.array([info['length'] for info in fw_excursion_inf_ls]))
bw_excursion_rel_amps = np.concatenate([info['excursion_rel_amps'] for info in bw_excursion_info_ls])
bw_total_len = np.sum(np.array([info['length'] for info in bw_excursion_info_ls]))

#%% Plot
count,amp_edges = np.histogram(fw_excursion_rel_amps,bins=20,range=(0,2))
plt.figure(dpi=200)
plt.bar((amp_edges[:-1]+amp_edges[1:])/2,count/fw_total_len/0.02,align='center',width=amp_edges[1]-amp_edges[0],color='C0',label='Forward')
count,amp_edges = np.histogram(bw_excursion_rel_amps,bins=20,range=(0,2))
plt.bar((amp_edges[:-1]+amp_edges[1:])/2,count/bw_total_len/0.02,align='center',width=amp_edges[1]-amp_edges[0],color='C1',label='Backward')
plt.xlabel('Relative amplitude of head casts')
plt.ylabel('Rate (Hz)')
plt.legend()
plt.show()
# %%
