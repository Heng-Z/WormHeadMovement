#%% Dependencies
import numpy as np
import matplotlib.pyplot as plt
import WormTool.timeseries
import os
import scipy.io
import importlib
importlib.reload(WormTool.timeseries)
import pickle
def excursion_metadata(name_ls,file_name,key_ls=None,**kwargs):
    #Calculate excursion information and save as pickle file
    # name_ls = ['N2','SMDk','SMBk','RMDk']
    min_excursion_time = kwargs.get('min_excursion_time',5)
    max_excursion_time = kwargs.get('max_excursion_time',70)
    dt = 0.02
    large_excur = 0
    group_info ={}

    for i,name in enumerate(name_ls):
        print(name)
        data = scipy.io.loadmat('../neuron_ablations/hb_dynamics/'+name+'.mat')
        if key_ls is not None and i< len(key_ls):
            key = key_ls[i]
        else:
            key = list(data.keys())[-1]
        data = data[key]
        trial_ls = []
        for i in range(data.shape[0]):
            kt = data[i,0][:,1]
            excursion_times = []
            excursion_amps = []
            excursion_rel_amps = []
            excursion_phase = []
            if np.isnan(kt).any():
                trial_ls.append(None)
                continue
            kt = WormTool.timeseries.smooth_data(kt,window=4)
            excursion_start_end,arrays = WormTool.timeseries.find_excursions_sign_match(kt,min_excursion_time=min_excursion_time,max_excursion_time=max_excursion_time,more_output=True)
            lf,denoised,phase_t = arrays[:3]
            amp_i = WormTool.timeseries.calc_amplitude(kt)
            threshold = amp_i*0.2
            excursion_start_end_valid = []
            for excursion in excursion_start_end:
                start_ind = excursion['start']
                end_ind = excursion['end']
                if start_ind < 30 or start_ind > len(kt)-30:
                    continue
                excursion_start_end_valid.append(excursion)
                excursion_times.append((end_ind-start_ind)*dt)
                amp = abs(denoised[end_ind]-denoised[start_ind])
                excursion_amps.append(amp)
                excursion_rel_amps.append(amp/amp_i)
                excursion_phase.append(phase_t[start_ind])
                if amp>threshold:
                    large_excur += 1
            excursion_amps = np.array(excursion_amps)
            excursion_rel_amps = np.array(excursion_rel_amps)
            excursion_times = np.array(excursion_times)
            excursion_phase = np.array(excursion_phase)
            trial_info_dict = {'excursion_ls':excursion_start_end_valid,'excursion_amps':excursion_amps,'excursion_rel_amps':excursion_rel_amps,
                               'excursion_times':excursion_times,'excursion_phase':excursion_phase,
                               'amp':amp_i,'length':len(kt)}
            trial_ls.append(trial_info_dict)
        group_info[name] = trial_ls
    file_name = file_name+'.pkl' if file_name[-4:]!='.pkl' else file_name
    with open(file_name,'wb') as f:
        pickle.dump(group_info,f)

def excursion_metadata_append(name_ls,old_file_name,new_name):
    #Calculate excursion information and save as pickle file
    # name_ls = ['N2','SMDk','SMBk','RMDk']
    min_excursion_time = 5
    max_excursion_time = 70
    dt = 0.02
    large_excur = 0
    with open(old_file_name,'rb') as f:
        group_info = pickle.load(f)
    for name in name_ls:
        print(name)
        if name in ['PSi','PSi_ctrl']:
            data = scipy.io.loadmat('../neuron_ablations/pan-sensory_inhibition/{}.mat'.format(name))
        else:
            data = scipy.io.loadmat('../neuron_ablations/hb_dynamics/'+name+'.mat')
        data = data[list(data.keys())[-1]]
        trial_ls = []
        for i in range(data.shape[0]):
            kt = data[i,0][:,1]
            excursion_times = []
            excursion_amps = []
            excursion_rel_amps = []
            excursion_phase = []
            if np.isnan(kt).any():
                trial_ls.append(None)
                continue
            kt = WormTool.timeseries.smooth_data(kt,window=4)
            excursion_start_end,arrays = WormTool.timeseries.find_excursions_sign_match(kt,min_excursion_time=min_excursion_time,max_excursion_time=max_excursion_time,more_output=True)
            lf,denoised,phase_t = arrays[:3]
            amp_i = WormTool.timeseries.calc_amplitude(kt)
            threshold = amp_i*0.2
            for excursion in excursion_start_end:
                start_ind = excursion['start']
                end_ind = excursion['end']
                excursion_times.append((end_ind-start_ind)*dt)
                amp = abs(denoised[end_ind]-denoised[start_ind])
                excursion_amps.append(amp)
                excursion_rel_amps.append(amp/amp_i)
                excursion_phase.append(phase_t[start_ind])
                if amp>threshold:
                    large_excur += 1
            excursion_amps = np.array(excursion_amps)
            excursion_rel_amps = np.array(excursion_rel_amps)
            excursion_times = np.array(excursion_times)
            excursion_phase = np.array(excursion_phase)
            trial_info_dict = {'excursion_ls':excursion_start_end,'excursion_amps':excursion_amps,'excursion_rel_amps':excursion_rel_amps,
                               'excursion_times':excursion_times,'excursion_phase':excursion_phase,
                               'amp':amp_i,'length':len(kt)}
            trial_ls.append(trial_info_dict)
        group_info[name] = trial_ls
    new_name = new_name+'.pkl' if new_name[-4:]!='.pkl' else new_name
    with open(new_name,'wb') as f:
        pickle.dump(group_info,f)

class ExcurInfo:
    def __init__(self,file_name) -> None:
        with open(file_name,'rb') as f:
            self.group_info = pickle.load(f)
        self.file_name = file_name
        self.name_ls = list(self.group_info.keys())

    def concat_trial_info(self,name):
        excursion_trial_ls = self.group_info[name]
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
        return excursion_amps,excursion_phase,excursion_rel_amps,data_len_all
        
#%%
if __name__ == '__main__':
    # excursion_metadata(['PSi','PSi_ctrl','RMDk_PSi_ctrl','RMDk_PSi','RMDSMDk_PSi_ctrl','RMDSMDk_PSi'],'excursion_trial_info_dict_PSi.pkl',min_excursion_time=5)
    excursion_metadata(['N2','SMDk','SMBk','RMDk','RMDSMBk','RMDSMDk','SMDSMBk'],'excursion_trial_info_dict_MN_ablation_20240922.pkl')
    # excursion_metadata(['RMDi_chb','RMDi_ctrl_chb','SMDi_chb','SMDi_ctrl_chb'],'excursion_trial_info_dict_MN_inhibition.pkl')
    # excursion_metadata(['SAAc_chb','SAAk_chb'],'excursion_trial_info_dict_SAA.pkl')
    # excursion_metadata(['RIA_twk_18'],'excursion_trial_info_dict_RIA_twk_18.pkl')
    # excursion_metadata(['RMEi_ctrl','RMEi'],'excursion_trial_info_dict_RMEi.pkl')
    # excursion_metadata(['Ai','unc-7_unc-9_mutant_Ai'],'excursion_trial_info_dict_GJk.pkl')
    # excursion_metadata(['RMDa','RMDa_ctrl','RMDi','RMDi_ctrl','SMDa','SMDa_ctrl','SMDi','SMDi_ctrl'],'excursion_trial_info_dict_opto.pkl')
    # excursion_metadata(['IS_PSi_ctrl','IS_PSi'],'excursion_trial_info_dict_IS_PSi.pkl',['dyn_HB','dyn_HB'],min_excursion_time=5)
# %%
# RIA_curv_data = data = np.load('../excursion/RIA_twk_18_curvatures.npy',allow_pickle=True)
# wrapped = np.empty_like(RIA_curv_data,dtype=object)
# for i in range(len(RIA_curv_data)):
#     curv_i = RIA_curv_data[i,0]
#     t_orig = np.arange(len(curv_i))*0.01
#     t_targ = np.arange(len(curv_i)//2)*0.02
#     curv_interp = np.interp(t_targ,t_orig,curv_i)
#     wrapped[i,0] = np.array([t_targ,curv_interp]).T
# scipy.io.savemat('../neuron_ablations/hb_dynamics/RIA_twk_18.mat',{'RIA_twk_18':wrapped})    
# %%
