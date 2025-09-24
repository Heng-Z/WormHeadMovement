#%%
import scipy.io
import glob
import numpy as np
data_file_ls = glob.glob('/Volumes/Lenovo/Pinjie/lpj paper videos 231/wen1120 pre 20210715/ExtractData/*.mat')

head_traj_list = []
time_stamp = []
for file in data_file_ls[-3:]:
    data = scipy.io.loadmat(file)
    curv = data['curvedatafiltered']
    for j in range(len(curv)//2000):
        if j == 2:
            plt.plot(curv[j*2000:(j+1)*2000,:15].mean(axis=1))
            plt.show()
        head_traj_list.append(curv[j*2000:(j+1)*2000,:15].mean(axis=1))
head_array = np.empty((len(head_traj_list),1),dtype=object)
for i in range(len(head_traj_list)):
    head_array[i,0] = np.stack([np.arange(2000)*0.01,head_traj_list[i]],axis=1)

import scipy.io

# Save the head_traj_list as a .mat file
scipy.io.savemat('/Volumes/Lenovo/neuron_ablations/hb_dynamics/triple.mat', 
                 {'head_array': head_array})


# %%
