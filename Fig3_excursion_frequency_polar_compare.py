#%%
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
import scipy.io
import WormTool.timeseries
import pickle
from find_excursion import ExcurInfo
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42
plt.rcParams['font.family'] = 'Arial'
def frequency_ls(trials,min_excursion_time=10,dt = 0.02):
    '''
    Parameters
    ----------
    trials : list of 1D arrays
        Each array is an episode of the head movement
    
    min_excursion_time : int
        Minimum time of excursion to be counted as an excursion

    Returns
    -------
    frequency : 1D array
        Frequency of excursion each episode
    '''
    # frequency = [np.zeros(len(trials))]
    frequency_ls =[]
    for i,kt in enumerate(trials):
        excursion_ls = WormTool.timeseries.find_excursions_sign_match(kt,min_excursion_time=min_excursion_time)
        # frequency[i] = len(excursion_ls)/kt.shape[0]/dt
        if len(excursion_ls) >0:
            frequency_ls.append(len(excursion_ls)/kt.shape[0]/dt)
    return frequency_ls

def get_mutant_excursion_frequency(name,min_excursion_time=10):
    data = scipy.io.loadmat('../neuron_ablations/'+name+'.mat')[name+'_hb_dynamics']
    trial_ls = [data[i,0][:,1] for i in range(data.shape[0])]
    return frequency_ls(trial_ls,min_excursion_time=min_excursion_time)

def get_mutant_excursion_freq_from_file(name,threshold,file_name = 'excursion_trial_info_dict_MN_ablation.pkl'):
    info = ExcurInfo(file_name)
    trial_ls = info.group_info[name]
    freq_ls = []
    for trial in trial_ls:
        if trial is None:
            continue
        rel_amp = trial['excursion_rel_amps']
        # number of excursions with relative amplitude larger than threshold
        num_excursion = np.sum(rel_amp>threshold)
        if num_excursion>0:
            freq_ls.append(num_excursion/trial['length']/0.02)
        else:
            freq_ls.append(0)
    return freq_ls
#%%
# N2_freq_ls = get_mutant_excursion_frequency('N2',min_excursion_time=6)
# RMD_freq_ls = get_mutant_excursion_frequency('RMDk',min_excursion_time=6)
N2_freq_ls = get_mutant_excursion_freq_from_file('N2',0.3)
RMDk_freq_ls = get_mutant_excursion_freq_from_file('RMDk',0.3)
SMDk_freq_ls = get_mutant_excursion_freq_from_file('SMDk',0.3)
SMBk_freq_ls = get_mutant_excursion_freq_from_file('SMBk',0.3)
RMDSMDk_freq_ls = get_mutant_excursion_freq_from_file('RMDSMDk',0.3)
RMDSMBk_freq_ls = get_mutant_excursion_freq_from_file('RMDSMBk',0.3)
#%%
# MW U-test RMDk vs N2, SMDk vs N2, SMBk vs N2
print('Mann-Whitney U test:')
print('RMDk vs N2, p value =',stats.mannwhitneyu(RMDk_freq_ls,N2_freq_ls)[1])
print('SMDk vs N2, p value =',stats.mannwhitneyu(SMDk_freq_ls,N2_freq_ls)[1])
print('SMBk vs N2, p value =',stats.mannwhitneyu(SMBk_freq_ls,N2_freq_ls)[1])
data = [N2_freq_ls,RMDk_freq_ls,SMDk_freq_ls,SMBk_freq_ls]
plt.figure(dpi=300)
plt.boxplot(data,showfliers=False)
plt.ylim([0,2])
plt.xticks([1,2,3,4],['N2','RMDk','SMDk','SMBk'],fontsize=15)
plt.yticks(np.arange(0,2.1,0.5),fontsize=15)
plt.ylabel('Frequency (Hz)',fontsize=15)
# plt.savefig('Fig3_excursion_frequency_RMDk_SMDk_SMBk_vs_N2.png')
#%%
# MW U-test RMDSMDk vs N2, RMDSMBk vs N2
print('Mann-Whitney U test:')
print('RMDSMDk vs N2, p value =',stats.mannwhitneyu(RMDSMDk_freq_ls,N2_freq_ls)[1])
print('RMDSMBk vs N2, p value =',stats.mannwhitneyu(SMBk_freq_ls,N2_freq_ls)[1])
data = [N2_freq_ls,RMDSMDk_freq_ls,RMDSMBk_freq_ls]
plt.figure(dpi=300)
plt.boxplot(data,showfliers=False)
plt.ylim([0,2])
plt.xticks([1,2,3],['N2','RMDSMDk','RMDSMBk'],fontsize=15)
plt.yticks(np.arange(0,2.1,0.5),fontsize=15)
plt.ylabel('Frequency (Hz)',fontsize=15)
#%% Do Violin plot
data = [N2_freq_ls,RMDk_freq_ls,SMDk_freq_ls,SMBk_freq_ls]
# plt.figure(dpi=300,figsize=(5,5))
fig,ax = plt.subplots(1,1,figsize=(6,6),dpi=300)
colors = ['#a7d99e','#f27f91','#f6d98a','#708ac1']
violin_parts = ax.violinplot(data,showmedians=True,showextrema=False,quantiles=[[0.25,0.75]]*4)
font = {'family': 'serif','color':  'black','weight': 500,'size': 22}
for i, pc in enumerate(violin_parts['bodies']):
    pc.set_alpha(1)
    pc.set_linewidth(3)
    pc.set_linestyle('-')
    pc.set_edgecolor([0,0,0,1])
    pc.set_facecolor(colors[i])
for partname in ('cmedians','cquantiles'):
    vp = violin_parts[partname]
    vp.set_edgecolor('black')
    vp.set_linewidth(1.5)
    vp.set_linestyle('--') if partname=='cmedians' else vp.set_linestyle(':')
ax.set_ylim([0,2])
ax.set_xticks([1,2,3,4])
ax.set_xticklabels(['N2','RMD ablated','SMD ablated','SMB ablated'],fontsize=22,rotation=45)
ax.set_yticks(np.arange(0,2.1,0.5))
ax.set_yticklabels(np.arange(0,2.1,0.5),fontsize=22)
ax.set_ylabel('Head-casting Frequency (Hz)',fontsize=24)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.tick_params(axis='both',which='both',direction='in',length=6,width=2,labelsize=24)
plt.setp(ax.spines.values(), lw=2)
#Plot Stars above the violin plot
plt.text(2,1.2,'****',fontsize=24,ha='center',weight=700)
plt.text(3,1.8,'****',fontsize=24,ha='center',weight=700)
plt.text(4,1.8,'****',fontsize=24,ha='center',weight=700)
plt.savefig('Fig3_excursion_frequency_RMDk_SMDk_SMBk_vs_N2_violin.png')
#%%
with open('excursion_frequency_MN_ablation.pkl','wb') as f:
    pickle.dump([N2_freq_ls,RMDk_freq_ls,SMDk_freq_ls,SMBk_freq_ls,RMDSMDk_freq_ls,RMDSMBk_freq_ls],f)
with open('excursion_frequency_MN_ablation.pkl','rb') as f:
    N2_freq_ls,RMDk_freq_ls,SMDk_freq_ls,SMBk_freq_ls,RMDSMDk_freq_ls,RMDSMBk_freq_ls = pickle.load(f)
#%%
# MW U-test RMDSMDk vs N2, RMDSMBk vs N2
print('Mann-Whitney U test:')
print('RMDSMDk vs N2, p value =',stats.mannwhitneyu(RMDSMDk_freq_ls,N2_freq_ls)[1])
print('RMDSMBk vs N2, p value =',stats.mannwhitneyu(SMBk_freq_ls,N2_freq_ls)[1])
#%% Do Violin plot
data = [N2_freq_ls,RMDSMDk_freq_ls,RMDSMBk_freq_ls]
# plt.figure(dpi=300,figsize=(5,5))
fig,ax = plt.subplots(1,1,figsize=(3,6),dpi=300)
colors = ['#a7d99e','#f27f91','#f6d98a','#708ac1']
violin_parts = ax.violinplot(data,showmedians=True,showextrema=False,quantiles=[[0.25,0.75]]*3)
font = {'family': 'serif','color':  'black','weight': 500,'size': 22}
for i, pc in enumerate(violin_parts['bodies']):
    pc.set_alpha(1)
    pc.set_linewidth(3)
    pc.set_linestyle('-')
    pc.set_edgecolor([0,0,0,1])
    pc.set_facecolor(colors[i])
for partname in ('cmedians','cquantiles'):
    vp = violin_parts[partname]
    vp.set_edgecolor('black')
    vp.set_linewidth(1.5)
    vp.set_linestyle('--') if partname=='cmedians' else vp.set_linestyle(':')
ax.set_xticks([1,2,3])
ax.set_xticklabels(['N2','RMDSMD ablated','RMDSMB ablated'],fontsize=22,rotation=45)
ax.set_yticks(np.arange(0,1.6,0.5))
ax.set_yticklabels(np.arange(0,1.6,0.5),fontsize=22)
ax.set_ylabel('Head-casting Frequency (Hz)',fontsize=24)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.tick_params(axis='both',which='both',direction='in',length=6,width=2,labelsize=24)
plt.setp(ax.spines.values(), lw=2)
#Plot Stars above the violin plot
plt.text(2,1.0,'**',fontsize=24,ha='center',weight=700)
plt.text(3,1.2,'****',fontsize=24,ha='center',weight=700)
plt.savefig('Fig3_excursion_frequency_RMDSMDk_RMDSMBk_vs_N2_violin.pdf')
plt.show()

#%%
# frequency list as excel file 4 colmns for N2_freq_ls,RMDk_freq_ls,SMDk_freq_ls,SMBk_freq_ls
# the number of elements in each column is the different




#%%
# MW U-test RMDSMDk vs N2, RMDSMBk vs N2
print('Mann-Whitney U test:')
print('RMDSMDk vs SMDk, p value =',stats.mannwhitneyu(RMDSMDk_freq_ls,SMDk_freq_ls)[1])
print('RMDSMBk vs SMBk, p value =',stats.mannwhitneyu(SMBk_freq_ls,RMDSMBk_freq_ls)[1])
#%% Do Violin plot

data = [SMDk_freq_ls,RMDSMDk_freq_ls,SMBk_freq_ls,RMDSMBk_freq_ls]
# plt.figure(dpi=300,figsize=(5,5))
fig,ax = plt.subplots(1,1,figsize=(6,6),dpi=300)
colors = ['#a7d99e','#f27f91','#708ac1','#f6d98a']
violin_parts = ax.violinplot(data,showmedians=True,showextrema=False,quantiles=[[0.25,0.75]]*len(data))
font = {'family': 'serif','color':  'black','weight': 500,'size': 22}
for i, pc in enumerate(violin_parts['bodies']):
    pc.set_alpha(1)
    pc.set_linewidth(3)
    pc.set_linestyle('-')
    pc.set_edgecolor([0,0,0,1])
    pc.set_facecolor(colors[i])
for partname in ('cmedians','cquantiles'):
    vp = violin_parts[partname]
    vp.set_edgecolor('black')
    vp.set_linewidth(1.5)
    vp.set_linestyle('--') if partname=='cmedians' else vp.set_linestyle(':')
ax.set_xticks(np.arange(1,len(data)+1))
ax.set_xticklabels(['SMD ablated','RMDSMD ablated','SMB ablated','RMDSMB ablated'],fontsize=22,rotation=45)
ax.set_yticks(np.arange(0,1.6,0.5))
ax.set_yticklabels(np.arange(0,1.6,0.5),fontsize=22)
ax.set_ylabel('Head-casting Frequency (Hz)',fontsize=24)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.tick_params(axis='both',which='both',direction='in',length=6,width=2,labelsize=24)
plt.setp(ax.spines.values(), lw=2)
#Plot Stars above the violin plot
# plt.text(2,1.0,'****',fontsize=24,ha='center',weight=700)
# plt.text(4,1.2,'****',fontsize=24,ha='center',weight=700)
plt.savefig('Fig3_excursion_frequency_SMDk_RMDSMDk_SMBk_RMDSMBk_violin.pdf')
plt.show()



# %%
# Plot the freq-amplitude statistics
def freq_vs_amplitude_compare(name1,name2,file1,file2=None):
    if file2 is None:
        info = ExcurInfo(file1)
        concat_info1 = info.concat_trial_info(name1) # (excursion_amps,excursion_phase,excursion_rel_amps,data_len_all)
        concat_info2 = info.concat_trial_info(name2)
    else:
        info1 = ExcurInfo(file1)
        info2 = ExcurInfo(file2)
        concat_info1 = info1.concat_trial_info(name1)
        concat_info2 = info2.concat_trial_info(name2)
    amp_count1,amp_edge1 = np.histogram(concat_info1[0],bins=20,range=[0,12])
    amp_count2,amp_edge2 = np.histogram(concat_info2[0],bins=20,range=[0,12])
    rel_amp_count1,rel_amp_edge1 = np.histogram(concat_info1[2],bins=20,range=[0,1.5])
    rel_amp_count2,rel_amp_edge2 = np.histogram(concat_info2[2],bins=20,range=[0,1.5])
    fig,ax = plt.subplots(1,2,figsize=(8,8),dpi=300)
    ax[0].bar((amp_edge1[1:]+amp_edge1[:-1])/2,amp_count1/concat_info1[3]/0.02,width=amp_edge1[1]-amp_edge1[0],label=name1,alpha=0.5)
    ax[0].bar((amp_edge2[1:]+amp_edge2[:-1])/2,amp_count2/concat_info2[3]/0.02,width=amp_edge2[1]-amp_edge2[0],label=name2[:-1]+' ablated',alpha=0.5)
    ax[0].set_xlabel('Head-casts Amplitude',fontsize=22)
    ax[0].set_ylabel('Frequency(Hz)',fontsize=24)
    # ax[0].legend(fontsize=22,loc='upper center', bbox_to_anchor=(0.5, 1.1))
    ax[0].tick_params(axis='both',which='both',direction='in',length=6,width=2,labelsize=24)
    ax[0].spines['top'].set_visible(False)
    ax[0].spines['right'].set_visible(False)
    ax[0].set_ylim([0,0.16])
    plt.setp(ax[0].spines.values(), lw=2)
    # ax[0].set_position([0.1,0.1,0.35,0.8]) # 
    ax[1].set_position([0.1,0.55,0.8,0.4]) # 
    ax[0].set_position([0.1,0.1,0.8,0.4]) #
    ax[1].bar((rel_amp_edge1[1:]+rel_amp_edge1[:-1])/2,rel_amp_count1/concat_info1[3]/0.02,width=rel_amp_edge1[1]-rel_amp_edge1[0],label=name1,alpha=0.5)
    ax[1].bar((rel_amp_edge2[1:]+rel_amp_edge2[:-1])/2,rel_amp_count2/concat_info2[3]/0.02,width=rel_amp_edge2[1]-rel_amp_edge2[0],label=name2[:-1]+' ablated',alpha=0.5)
    ax[1].set_xlabel('Relative Head-casts Amplitude',fontsize=22)
    ax[1].set_ylabel('Frequency(Hz)',fontsize=24)
    ax[1].legend(fontsize=22,loc='upper center', bbox_to_anchor=(0.5, 1.1))
    ax[1].tick_params(axis='both',which='both',direction='in',length=6,width=2,labelsize=24)
    ax[1].spines['top'].set_visible(False)
    ax[1].spines['right'].set_visible(False)
    ax[1].set_ylim([0,0.16])
    plt.setp(ax[1].spines.values(), lw=2)
    # ax[1].set_position([0.6,0.1,0.35,0.8]) # 
    plt.show()

# freq_vs_amplitude_compare('N2','RMDSMDk','excursion_trial_info_dict_MN_ablation.pkl')
# freq_vs_amplitude_compare('SMDi_ctrl_chb','SMDi_chb','excursion_trial_info_dict_MN_inhibition.pkl','excursion_trial_info_dict_MN_inhibition.pkl')
# freq_vs_amplitude_compare('SAAc_chb','SAAk_chb','excursion_trial_info_dict_SAA.pkl')     
# freq_vs_amplitude_compare('RIA_twk_18','N2','excursion_trial_info_dict_RIA_twk_18.pkl','excursion_trial_info_dict_MN_ablation.pkl')
# freq_vs_amplitude_compare('RMEi_ctrl','RMEi','excursion_trial_info_dict_RMEi.pkl')
# freq_vs_amplitude_compare('PSi_ctrl','PSi','excursion_trial_info_dict_PSi.pkl')
freq_vs_amplitude_compare('Ai','unc-7_unc-9_mutant_Ai','excursion_trial_info_dict_GJk.pkl')
# freq_vs_amplitude_compare('PSi_ctrl','PSi','excursion_trial_info_dict_PSi.pkl')
# freq_vs_amplitude_compare('SMDa_ctrl','SMDa','excursion_trial_info_dict_opto.pkl')
# freq_vs_amplitude_compare('IS_PSi_ctrl','IS_PSi','excursion_trial_info_dict_IS_PSi.pkl')


# %% Plot Phase and amplitude joint distribution
def phase_amp_polar(name,file_name,vmax_abs=-3.5,vmax_rel=-3.5,vmin_abs=-5.2,vmin_rel=-5.2,size=(7,2.5),path=None,dt=0.02):

    info = ExcurInfo(file_name)
    concat_info = info.concat_trial_info(name)
    excursion_amps,excursion_phase,excursion_rel_amps,data_len_all = concat_info
    
    fig,ax = plt.subplots(1,2,figsize=size,dpi=300,subplot_kw=dict(projection='polar'))
    
    # 1. Plot the phase-amplitude joint distribution
    phase_amp_density,phase_edge,amp_edge = np.histogram2d(excursion_phase,excursion_amps,bins=[12,12],range=[[-np.pi,np.pi],[0,12]])
    phase_mesh, amp_mesh = np.meshgrid((phase_edge[1:]+phase_edge[:-1])/2,(amp_edge[1:]+amp_edge[:-1])/2)
    phase_amp_density_log = np.log((phase_amp_density+1)/data_len_all/dt)
    pcm = ax[0].pcolormesh(phase_mesh,amp_mesh, phase_amp_density_log.T, cmap='Reds', vmax=vmax_abs,vmin=vmin_abs,shading='auto')
    ax[0].grid(True)
    # set the font size of degree labels
    ax[0].tick_params(axis='both', labelsize=12)
    ax[0].set_xticklabels([])
    ax[0].set_yticklabels([])
    ax[0].set_title(name,fontsize=12)
    cbar = fig.colorbar(pcm, ax=ax[0])
    cbar.ax.tick_params(labelsize=12)
    # Plot the phase-relative amplitude joint distribution
    phase_amp_density,phase_edge,amp_edge = np.histogram2d(excursion_phase,excursion_rel_amps,bins=[12,12],range=[[-np.pi,np.pi],[0,1.5]])
    phase_mesh, amp_mesh = np.meshgrid((phase_edge[1:]+phase_edge[:-1])/2,(amp_edge[1:]+amp_edge[:-1])/2)
    phase_amp_density_log = np.log((phase_amp_density+1)/data_len_all/dt)
    pcm = ax[1].pcolormesh(phase_mesh,amp_mesh, phase_amp_density_log.T, cmap='Reds', vmax=vmax_rel,vmin=vmin_rel,shading='auto')
    ax[1].grid(True)
    # set the font size of degree labels
    # ax[1].set_yticks(np.arange(0,1.6,0.5))
    ax[1].tick_params(axis='x', labelsize=12)
    ax[1].tick_params(axis='y', labelsize=12)
    ax[1].set_title(name,fontsize=12)
    cbar = fig.colorbar(pcm, ax=ax[1],pad=0.1)
    cbar.ax.tick_params(labelsize=12)
    if path:
        plt.savefig(path)
    plt.show()

phase_amp_polar('RMDSMDk','excursion_data/excursion_trial_info_dict_MN_ablation.pkl',path='Fig3/Fig3_excursion_phase_amp_RMDSMDk.pdf',vmax_abs=-3.5,vmax_rel=-3.5,vmin_abs=-5,vmin_rel=-5.5)
# phase_amp_polar('SMDi_ctrl_chb','excursion_trial_info_dict_MN_inhibition.pkl',vmax_abs=-3,vmax_rel=-3) 
# phase_amp_polar('SAAk_chb','excursion_data/excursion_trial_info_dict_SAA.pkl',path='Fig3/Fig3_excursion_phase_amp_SAAk_chb.pdf',dt=0.01) 
# phase_amp_polar('SAAc_chb','excursion_data/excursion_trial_info_dict_SAA.pkl',path='Fig3/Fig3_excursion_phase_amp_SAAc_chb.pdf',dt=0.01) 
# phase_amp_polar('RIA_twk_18','excursion_trial_info_dict_RIA_twk_18.pkl')
# phase_amp_polar('RMEi_ctrl','excursion_data/excursion_trial_info_dict_RMEi.pkl',path='Fig3/Fig3_excursion_phase_amp_RMEi_ctrl.pdf')
# phase_amp_polar('RMEi','excursion_data/excursion_trial_info_dict_RMEi.pkl',path='Fig3/Fig3_excursion_phase_amp_RMEi.pdf')
# phase_amp_polar('PSi_ctrl','excursion_trial_info_dict_PSi.pkl',vmax_abs=-3,vmax_rel=-3,vmin_abs=-5,vmin_rel=-5)
# phase_amp_polar('Ai','excursion_trial_info_dict_GJk.pkl')
# phase_amp_polar('IS_PSi_ctrl','excursion_trial_info_dict_IS_PSi.pkl',vmin_abs=-5,vmin_rel=-5,vmax_abs=-3.5,vmax_rel=-3.5)

# %%
# RMEi_ctrl vs RMEi
RMEi_ctrl_freq_ls = get_mutant_excursion_freq_from_file('RMEi_ctrl',0.3,file_name='excursion_data/excursion_trial_info_dict_RMEi.pkl')
RMEi_freq_ls = get_mutant_excursion_freq_from_file('RMEi',0.3,file_name='excursion_data/excursion_trial_info_dict_RMEi.pkl')
print('Mann-Whitney U test:')
print('RMEi_ctrl vs RMEi, p value =',stats.mannwhitneyu(RMEi_ctrl_freq_ls,RMEi_freq_ls)[1])
data = [RMEi_ctrl_freq_ls,RMEi_freq_ls]
labels = ['control','RME inhibition']
size = (2.5,1.8)
WormTool.statistic_plot.plot_violin(data,label_ls=labels,y_label='Head casts freq. (Hz)',size=size,save_path='Fig3/Fig3_excursion_frequency_RMEi_ctrl_vs_RMEi.pdf')
# %%
# SAAk_chb vs SAAk_ctrl
SAAk_chb_freq_ls = get_mutant_excursion_freq_from_file('SAAk_chb',0.3,file_name='excursion_data/excursion_trial_info_dict_SAA.pkl')
SAAc_chb_freq_ls = get_mutant_excursion_freq_from_file('SAAc_chb',0.3,file_name='excursion_data/excursion_trial_info_dict_SAA.pkl')

SAAk_chb_freq_ls = [i *2 for i in SAAk_chb_freq_ls] # correct for dt = 0.01; in batch data processing, dt = 0.02
SAAc_chb_freq_ls = [i *2 for i in SAAc_chb_freq_ls]
print('Mann-Whitney U test:')
print('SAAk_chb vs SAAc_chb, p value =',stats.mannwhitneyu(SAAk_chb_freq_ls,SAAc_chb_freq_ls)[1])
data = [SAAk_chb_freq_ls,SAAc_chb_freq_ls]
labels = ['control','SAA ablation']
y_ticks = [0,0.5,1.0]
size = (2.5,1.8)
WormTool.statistic_plot.plot_violin(data,label_ls=labels,y_label='Head casts freq. (Hz)',size=size,y_ticks=y_ticks,save_path='Fig3/Fig3_excursion_frequency_SAAk_chb_vs_SAAc_chb.pdf')
# %%