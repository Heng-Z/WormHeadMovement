#%%
import WormTool
import numpy as np
import scipy.io
from tqdm import tqdm
from ipywidgets import interact
import matplotlib.pyplot as plt
from csaps import csaps
import pickle
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42
plt.rcParams['font.family'] = 'Arial'
from scipy.stats import mannwhitneyu
#%%
# load excursion data of forward trials
name = 'N2'
# load backward excursion data
with open('excursion_trial_info_dict_{}_backward.pkl'.format(name),'rb') as f:
    excursion_trial_info_ls = pickle.load(f)

with open('/Users/hengzhang/projects/WormSim/figures/excursion_trial_info_dict_MN_ablation.pkl','rb') as f:
    forward_excursion_trial_info_dict = pickle.load(f)
fw_data = forward_excursion_trial_info_dict[name] # list of excursion information
fw_excursion_rel_amps = np.concatenate([info['excursion_rel_amps'] for info in fw_data])
fw_excursion_phase = np.concatenate([info['excursion_phase'] for info in fw_data])
fw_total_len = np.sum(np.array([info['length'] for info in fw_data]))
bw_excursion_rel_amps = np.concatenate([info['excursion_rel_amps'] for info in excursion_trial_info_ls])
bw_excursion_phase = np.concatenate([info['excursion_phase'] for info in excursion_trial_info_ls])
bw_total_len = np.sum(np.array([info['length'] for info in excursion_trial_info_ls]))

# plot relative amplitude distribution
count,amp_edges = np.histogram(fw_excursion_rel_amps,bins=20,range=(0,2))
plt.figure(dpi=200)
plt.bar((amp_edges[:-1]+amp_edges[1:])/2,count/fw_total_len/0.02,align='center',width=amp_edges[1]-amp_edges[0],color='C0',label='Forward')
count,amp_edges = np.histogram(bw_excursion_rel_amps,bins=20,range=(0,2))
plt.bar((amp_edges[:-1]+amp_edges[1:])/2,count/bw_total_len/0.02,align='center',width=amp_edges[1]-amp_edges[0],color='C1',label='Backward')
plt.xlabel('Relative amplitude of head casts')
plt.ylabel('Rate (Hz)')
plt.legend()
plt.title(name)
plt.show()

# Plot phase-amplitude joint distribution
WormTool.statistic_plot.plot_polor_joint(fw_excursion_phase,fw_excursion_rel_amps,fw_total_len*0.02,name,-3,-5.2,1.2,save_path='results_plots/{}_fw_phase_amp_joint.pdf'.format(name))
WormTool.statistic_plot.plot_polor_joint(bw_excursion_phase,bw_excursion_rel_amps,bw_total_len*0.02,name,-3,-5.2,1.2,save_path='results_plots/{}_bw_phase_amp_joint.pdf'.format(name))


# compare excursion rate
threshold = 0.3
dt = 0.02
fw_rate = [np.sum(info['excursion_rel_amps']>threshold)/(info['length']*dt) for info in fw_data]
bw_rate = [np.sum(info['excursion_rel_amps']>threshold)/(info['length']*dt) for info in excursion_trial_info_ls]
# Mann-Whitney U test
statistic,pvalue = mannwhitneyu(fw_rate,bw_rate)
print('Mann-Whitney U test: p-value =',pvalue)
fig,ax = plt.subplots(1,1,figsize=(3,5),dpi=300)
WormTool.statistic_plot.plot_whisker_box([fw_rate,bw_rate],['Forward','Backward'],'Head cast rate (Hz)',stat_label=['N={}'.format(len(fw_rate)),'N={}'.format(len(bw_rate))],ax=ax)
ax.set_title(name + f' p={pvalue:.2e}')
plt.savefig('results_plots/{}_head_cast_rate_fw_bw.pdf'.format(name))
plt.show()

# %%
# Compare the gap junction kill and control: 'Ai','unc-7_unc-9_mutant_Ai','excursion_trial_info_dict_GJk.pkl'
name1 = 'Ai'
name2 = 'unc-7_unc-9_mutant_Ai'
threshold = 0.3
dt = 0.02

with open('../excursion_trial_info_dict_GJk.pkl','rb') as f:
    forward_excursion_trial_info_dict = pickle.load(f)
fw_data1 = forward_excursion_trial_info_dict[name1] # list of excursion information
fw_data2 = forward_excursion_trial_info_dict[name2] # list of excursion information
fw_excursion_rel_amps1 = np.concatenate([info['excursion_rel_amps'] for info in fw_data1])
fw_excursion_phase1 = np.concatenate([info['excursion_phase'] for info in fw_data1])
fw_total_len1 = np.sum(np.array([info['length'] for info in fw_data1]))
fw_rate1 = [np.sum(info['excursion_rel_amps']>threshold)/(info['length']*dt) for info in fw_data1]

fw_excursion_rel_amps2 = np.concatenate([info['excursion_rel_amps'] for info in fw_data2])
fw_excursion_phase2 = np.concatenate([info['excursion_phase'] for info in fw_data2])
fw_total_len2 = np.sum(np.array([info['length'] for info in fw_data2]))
fw_rate2 = [np.sum(info['excursion_rel_amps']>threshold)/(info['length']*dt) for info in fw_data2]

# Compare rate1 and rate2
statistic,pvalue = mannwhitneyu(fw_rate1,fw_rate2)
print('Mann-Whitney U test: p-value =',pvalue)
fig,ax = plt.subplots(1,1,figsize=(3,5),dpi=300)
WormTool.statistic_plot.plot_whisker_box([fw_rate1,fw_rate2],[name1,name2],'Head cast rate (Hz)',stat_label=['N={}'.format(len(fw_rate1)),'N={}'.format(len(fw_rate2))],ax=ax)
ax.set_title(f'p={pvalue:.2e}')
plt.savefig('results_plots/{}_{}_head_cast_rate_fw.pdf'.format(name1,name2))
plt.show()

# Plot phase-amplitude joint distribution
save1 = 'results_plots/{}_fw_phase_amp_joint.pdf'.format(name1)
save2 = 'results_plots/{}_fw_phase_amp_joint.pdf'.format(name2)
WormTool.statistic_plot.plot_polor_joint(fw_excursion_phase1,fw_excursion_rel_amps1,fw_total_len1*dt,name1,-3,-5.2,1.2,save_path=save1)
WormTool.statistic_plot.plot_polor_joint(fw_excursion_phase2,fw_excursion_rel_amps2,fw_total_len2*dt,name2,-3,-5.2,1.2,save_path=save2)


# %%
