#%%
import numpy as np
import matplotlib.pyplot as plt
from vmdpy import VMD
import scipy.io
import scipy.interpolate
import WormTool.timeseries
import importlib
importlib.reload(WormTool.timeseries)
from matplotlib.backends.backend_pdf import PdfPages
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42
plt.rcParams['font.family'] = 'Arial'
#%%
data = np.load('../neuron_ablations/processed/N2.npy',allow_pickle=True)
i = 7
kt = data[i,1]
lf,dn,_ = WormTool.timeseries.find_lf_denoised(kt)
hf = dn-lf
fig,ax = plt.subplots(2,1,sharex=True,dpi=300)
ax[0].plot(hf,alpha=0.8)
ax[1].plot(lf,alpha=0.8)
#%%
data = np.load('../neuron_ablations/processed/N2.npy',allow_pickle=True)
i = 36
kt = data[i,1][201:]
t = np.arange(0,len(kt))*0.02
body_curv = data[i,2][45:70,201:].mean(axis=0)
lf,dn,_ = WormTool.timeseries.find_lf_denoised(kt)
noise = kt-dn
hf = dn-lf
fig,ax = plt.subplots(3,1,sharex=True,dpi=300,figsize=(7,5))
ax[0].plot(t,kt,label='original',linewidth=2)
ax[0].plot(t,dn,'r--',label='denoised',linewidth=1)
ax[0].legend(loc='upper right',bbox_to_anchor=(1,1.25))
# ax[0].set_ylim([-12,12])
ax[1].plot(t,lf,label='slow dynamic mode')
ax[1].plot(t,body_curv,label='body curvature')
ax[1].legend(loc='upper right',bbox_to_anchor=(1,1.25))
ax[2].plot(t,hf,label='fast dynamic mode')
ax[2].plot(t,noise,'r',label='noise')
ax[2].legend(loc='upper right',bbox_to_anchor=(1,1.25))
ax[0].tick_params(axis='both', which='major', labelsize=20)
ax[1].tick_params(axis='both', which='major', labelsize=20)
ax[2].tick_params(axis='both', which='major', labelsize=20)

# %%
#%%
data = np.load('../neuron_ablations/curvature/N2.npy',allow_pickle=True)
i = 7
kt = data[i,0][:15,:].mean(axis=0)
lf,dn,_ = WormTool.timeseries.find_lf_denoised(kt)
hf = dn-lf
fig,ax = plt.subplots(2,1,sharex=True,dpi=300)
ax[0].plot(hf,alpha=0.8)
ax[1].plot(lf,alpha=0.8)
#%%
data = np.load('../neuron_ablations/curvature/N2.npy',allow_pickle=True)
i = 22
kt = data[i,0][:15,:].mean(axis=0)
t = data[i,1][:,0] - data[i,1][0,0]
body_curv = WormTool.timeseries.smooth_data(data[i,0][45:60].mean(axis=0),window=5)
lf,dn,_ = WormTool.timeseries.find_lf_denoised(kt)
noise = kt-dn
hf = dn-lf
fig,ax = plt.subplots(3,1,sharex=True,dpi=300,figsize=(8,5))
ax[0].plot(t,kt,label='original',linewidth=2)
ax[0].plot(t,dn,'r--',label='denoised',linewidth=1)
# ax[0].set_ylim([-12,12])
ax[1].plot(t,lf,label='slow dynamic mode')
ax[1].plot(t,body_curv,label='body curvature')
ax[2].plot(t,hf,label='fast dynamic mode')
ax[2].plot(t,noise,'r',label='noise')
_ = [ax[i].legend(loc='upper right',bbox_to_anchor=(1,1.25)) for i in range(3)]
_ = [ax[i].tick_params(axis='both', which='major', labelsize=20) for i in range(3)]
_ = [ax[i].set_yticks([-10,0,10]) for i in range(3)]
# put ticks inside the box
_ = [ax[i].tick_params(direction='in') for i in range(3)]
# remove top and right spines
_ = [ax[i].spines['top'].set_visible(False) for i in range(3)]
_ = [ax[i].spines['right'].set_visible(False) for i in range(3)]
# %%
data = np.load('../neuron_ablations/curvature/N2.npy',allow_pickle=True)
i = 22
kt = data[i,0][:15,:].mean(axis=0)
t = data[i,1][:,0] - data[i,1][0,0]
body_curv = WormTool.timeseries.smooth_data(data[i,0][45:60].mean(axis=0),window=5)
lf,dn,_ = WormTool.timeseries.find_lf_denoised(kt)
noise = kt-dn
hf = dn-lf
fig,ax = plt.subplots(2,1,sharex=True,dpi=300,figsize=(8,5))
ax[0].plot(t,kt,'k',label='original',linewidth=2)
ax[0].plot(t,dn,'r',label='denoised',linewidth=1)
# ax[0].set_ylim([-12,12])
ax[1].plot(t,lf,'g',label='slow mode',linewidth=2)
ax[1].plot(t,hf,'m',label='fast mode',linewidth=2)
_ = [ax[i].legend(loc='upper right',bbox_to_anchor=(1,1.1),ncol=2) for i in range(ax.shape[0])]
_ = [ax[i].tick_params(axis='both', which='major', labelsize=20) for i in range(ax.shape[0])]
_ = [ax[i].set_yticks([-10,0,10]) for i in range(ax.shape[0])]
# put ticks inside the box
_ = [ax[i].tick_params(direction='in') for i in range(ax.shape[0])]
# remove top and right spines
_ = [ax[i].spines['top'].set_visible(False) for i in range(ax.shape[0])]
_ = [ax[i].spines['right'].set_visible(False) for i in range(ax.shape[0])]
ax[1].set_xlabel('Time (s)',fontsize=20)
ax[0].set_ylabel('Curvature (rad)',fontsize=20)
ax[1].set_ylabel('Curvature (rad)',fontsize=20)
plt.savefig('Fig1_VMD.pdf',bbox_inches='tight')
# with PdfPages('Fig1_VMD.pdf') as pdf:
#     pdf.savefig(fig)
# %%
