#%% Dependencies
import numpy as np
import matplotlib.pyplot as plt
import os
import scipy.io
import pickle
import statsmodels.api as sm
from find_excursion import ExcurInfo
import WormTool.timeseries
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42
plt.rcParams['font.family'] = 'Arial'
#%% Fig2a Excursion definition
# Load example head curvature time series
data = scipy.io.loadmat('../neuron_ablations/hb_dynamics/N2.mat')['N2_hb_dynamics']
# head_curv = data[7,0][1100:,1]
start_ind = 170
head_curv = data[90,0][:,1]
low_freq,denoised,phase = WormTool.timeseries.find_lf_denoised(head_curv)
head_curv = head_curv[start_ind:]
t = np.arange(0,len(head_curv))*0.02
low_freq = low_freq[start_ind:]
denoised = denoised[start_ind:]
dlf = np.gradient(low_freq,0.02) # time derivative of low frequency 
ddnoised = np.gradient(denoised,0.02) # time derivative of denoised
# Find excursion
excursion_start_end,upsamples = WormTool.timeseries.find_excursions_sign_match(head_curv,more_output=True,min_excursion_time=3)
fig,ax = plt.subplots(2,1,figsize=(15,10),dpi=300)
ax[0].plot(t,head_curv,linewidth=3,label='original')
ax[0].plot(t[:len(denoised)],denoised,'--',linewidth=3,label='denoised')
# ax[0].plot(t[:len(low_freq)],low_freq,'--',color='g',linewidth=2,label='low frequency')
ax[0].hlines([-10,0,10],xmin=0,xmax=t[-1],color='k',linestyle='--',linewidth=1)
ax[0].legend(loc='upper right',bbox_to_anchor=(1, 1.1),frameon=False,ncol=2,fontsize=24)
ax[0].spines['top'].set_visible(False)
ax[0].spines['right'].set_visible(False)
ax[0].spines['bottom'].set_linewidth(2)
ax[0].spines['left'].set_linewidth(2)
ax[0].set_xlim([0,t[-1]])
ax[0].tick_params(axis='both', which='major', labelsize=26)
ax[0].set_xlabel('Time (s)',fontsize=24)
ax[0].set_ylabel(r'Head curvature (1/L)',fontsize=24)

ax[1].plot(t[0:len(ddnoised)],ddnoised,'C1',linewidth=3,label='denoised')
ax[1].plot(t[0:len(dlf)],dlf,'g',linewidth=3,label='slow dynamic mode')
ax[1].legend(loc='upper right',bbox_to_anchor=(1,1.1),frameon=False,ncol=2,fontsize=24)
ax[1].spines['top'].set_visible(False)
ax[1].spines['right'].set_visible(False)
ax[1].spines['bottom'].set_linewidth(2)
ax[1].spines['left'].set_linewidth(2)
ax[1].set_xlim([0,t[-1]])
# ax[1].set_ylim([-18,18])
ax[1].tick_params(axis='both', which='major', labelsize=26)
ax[1].set_xlabel('Time (s)',fontsize=24)
ax[1].set_ylabel('Time derivative (1/(sL))',fontsize=24)
ax[1].hlines(0, xmin=0, xmax=t[-1], color='k', linestyle='--',linewidth=1)
for i in range(len(excursion_start_end)):
    color = 'C3' if excursion_start_end[i]['sign'] >0 else 'C4'
    ax[0].axvspan(t[excursion_start_end[i]['start']],t[excursion_start_end[i]['end']],color=color,alpha=0.3)
    ax[1].axvspan(t[excursion_start_end[i]['start']],t[excursion_start_end[i]['end']],color=color,alpha=0.3)
plt.tight_layout()
# plt.show()
fig.savefig('Fig2a_excursion_definition.png')

#%%
# Plot frequency of excursion vs. amplitude and relative amplitude
with open('excursion_info_dict.pkl','rb') as f:
    info_dict = pickle.load(f)
name_ls = info_dict.keys()
for name in name_ls:
    info = info_dict[name]
    excursion_amps = info['amplitude']
    excursion_rel_amps = info['rel_amp']
    data_len = info['data_len']
    data_len_all = np.array(data_len).sum()*0.02 # in seconds
    plt.figure(figsize=(5,5),dpi=300)
    count,amp_edges = np.histogram(excursion_rel_amps,bins=20,range=(0,2))
    plt.bar((amp_edges[:-1]+amp_edges[1:])/2,count/data_len_all,align='center',width=amp_edges[1]-amp_edges[0],color='C0')
    plt.xlabel('Rel. Amplitude',fontsize=14)
    plt.ylabel('Frequency (Hz)',fontsize=14)
    plt.ylim([0,0.15])

#%%
# Plot N2 and RMDk
with open('excursion_info_dict.pkl','rb') as f:
    info_dict = pickle.load(f)
name_ls = ['N2','RMDk']
plt.figure(figsize=(5,5),dpi=300)
for name in name_ls:
    info = info_dict[name]
    excursion_amps = info['amplitude']
    excursion_rel_amps = info['rel_amp']
    data_len = info['data_len']
    data_len_all = np.array(data_len).sum()*0.02 # in seconds
    count,amp_edges = np.histogram(excursion_rel_amps,bins=20,range=(0,2))
    plt.bar((amp_edges[:-1]+amp_edges[1:])/2,count/data_len_all,align='center',width=amp_edges[1]-amp_edges[0],label=name,alpha=0.5)
plt.legend()
plt.xlabel('Rel. Amplitude',fontsize=14)
plt.ylabel('Frequency (Hz)',fontsize=14)
plt.ylim([0,0.15])



# %%
# Plot the excursion statistics
with open('excursion_info_dict.pkl','rb') as f:
    info_dict = pickle.load(f)
name_ls = info_dict.keys()
for name in name_ls:
    info = info_dict[name]
    excursion_amps = info['amplitude']
    excursion_rel_amps = info['rel_amp']
    excursion_times = info['duration']
    excursion_phase = info['phase']
    data_len = info['data_len']
    data_len_all = np.array(data_len).sum()*0.02
    count,phase_edges = np.histogram(excursion_phase,bins=12,range=(-np.pi,np.pi))
    freq = count/data_len_all
    # digitized the phase
    phase_digitized = np.digitize(excursion_phase,phase_edges[1:]) # make sure the digitized phase is in the range of [0,11]
    amp_phase = [excursion_amps[phase_digitized==i].mean() for i in range(len(phase_edges)-1)]

    print('{} overall frequency: '.format(name),freq.sum())
    plt.figure(figsize=(10,5),dpi=300)
    ax1 = plt.subplot(121, projection='polar',frameon=False)
    ax1.bar((phase_edges[:-1] + phase_edges[1:])/2, freq, width=phase_edges[1]-phase_edges[0]-0.01, bottom=0.0)
    ax1.set_theta_zero_location("E")
    ax1.set_yticks([0.05,0.1,0.15])
    ax1.set_yticklabels(['0.05Hz','0.1Hz','0.15Hz'])
    ax1.set_title('Frequency',fontsize=14)
    # ax1.set_xticks(np.linspace(0,2*np.pi,4,endpoint=False))
    # ax1.set_xticklabels(['0','90^o','180^o','270^o'])
    ax2 = plt.subplot(122, projection='polar',frameon=False)
    ax2.bar((phase_edges[:-1] + phase_edges[1:])/2, amp_phase, width=phase_edges[1]-phase_edges[0]-0.01, bottom=0.0)
    ax2.set_theta_zero_location("E")
    ax2.set_yticks([1,2,3,4])
    ax2.set_yticklabels(['1','2','3','4'])
    ax2.set_title('Amplitude',fontsize=14)
    plt.tight_layout()
    # plt.show()
    plt.savefig('Fig2b_{}_excursion_phase_amp.png'.format(name))


# ax = plt.subplot(212, projection='polar')

# %%
group_info = ExcurInfo('excursion_trial_info_dict_MN_ablation.pkl')
name_ls = ['N2','RMDk','SMDk','SMBk']
fig, ax = plt.subplots(2, 2, figsize=(16, 16),dpi=300,subplot_kw=dict(projection='polar'))
for ii in range(2):
    for jj in range(2):
        name = name_ls[ii*2+jj]
        info = group_info.concat_trial_info(name) # return: excursion_amps,excursion_phase,excursion_rel_amps,data_len_all
        excursion_amps = info[0]
        excursion_phase = info[1]
        excursion_rel_amps = info[2]
        data_len_all = info[3]*0.02
        
        phase_amp_density,phase_edge,amp_edge = np.histogram2d(excursion_phase,excursion_amps,bins=[12,12],range=[[-np.pi,np.pi],[0,12]])
        phase_mesh, amp_mesh = np.meshgrid((phase_edge[1:]+phase_edge[:-1])/2,(amp_edge[1:]+amp_edge[:-1])/2)
        phase_amp_density_log = np.log((phase_amp_density+1)/data_len_all)
        pcm = ax[ii,jj].pcolormesh(phase_mesh,amp_mesh, phase_amp_density_log.T, cmap='Reds', vmax=-3.5,vmin=-5.2)
        # set the font size of degree labels
        ax[ii,jj].tick_params(axis='both', labelsize=20)

        # set radius ticks
        # ax[ii,jj].set_yticks([0.5,1,1.5,2])
        # ax[ii,jj].set_yticklabels(['0.5','1','1.5','2'],fontsize=17)
        ax[ii,jj].set_title(name,fontsize=20)
        cbar = fig.colorbar(pcm, ax=ax[ii,jj],pad=0.1)
        cbar.ax.tick_params(labelsize=20)
plt.savefig('Fig2_excursion_polar.png')
# %%
# Excursion examples in curvature heatmap
name = 'N2'
data = np.load('../neuron_ablations/processed/'+name+'.npy',allow_pickle=True)
ind = 4
curvature = data[ind,2]
kt = data[ind,1]
start = 0
end = len(kt)
t = np.arange(end-start)*0.02
fig,ax = plt.subplots(2,1,figsize=(10,10),dpi=300)
ax[0].plot(t,kt[start:end],linewidth=2)
ax[0].set_xlim([t[0],t[-1]])
ax[0].set_ylim([-15,15])
ax[0].set_xlabel('Time (s)',fontsize=14)
ax[0].set_ylabel('Head curvature (L^-1)',fontsize=14)
ax[0].set_xticklabels(ax[0].get_xticks(),fontsize=14)
ax[0].set_yticklabels(ax[0].get_yticks(),fontsize=14)
ax[0].spines['top'].set_visible(False)
ax[0].spines['right'].set_visible(False)
ax[1].imshow(-curvature[:,start:end],aspect='auto',cmap='coolwarm',vmin=-15,vmax=15)
ax[1].set_xlabel('Time (s)',fontsize=14)
ax[1].set_ylabel('Curvature (L^-1)',fontsize=14)
ax[1].set_xticks(np.arange(0,end-start,100))
ax[1].set_xticklabels((np.arange(0,end-start,100)*0.02).astype(np.int32),fontsize=14)
ax[1].set_yticks(np.arange(0,curvature.shape[0],20))
ax[1].set_yticklabels(np.arange(0,curvature.shape[0],20),fontsize=14)
plt.tight_layout()
# plt.savefig('Fig2_N2_excursion_example.png')
# plt.close()




