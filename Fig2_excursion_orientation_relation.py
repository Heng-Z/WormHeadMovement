#%% Dependencies
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['font.size'] = 12
import WormTool
import os
import scipy.io
import pickle
import statsmodels.api as sm
from find_excursion import ExcurInfo
import pandas as pd
#%% Correlation between Dorsal-Ventral bias and orientation change
eig_worms = np.load('data/N2_eigenworms.npy')
name = 'N2'
data = np.load('/Volumes/Lenovo/neuron_ablations/processed/{}.npy'.format(name),allow_pickle=True)
head_curvature = data[:,1]
full_curvature = data[:,2]
orientation_data = data[:,3]


#%%
d_orientation = []
d_DV_bias = []
DV_shaffle = True
m = 3
for i in range(len(head_curvature)):
    if (np.isnan(full_curvature[i])).any():
        continue
# i = 8
# if True:
    curvature_i = full_curvature[i]
    proj_i = curvature_i.T @ eig_worms[:,:2]
    phase = np.arctan2(proj_i[:,0], proj_i[:,1])
    orientation_i = orientation_data[i]
    phase_jump = np.where(np.abs(np.diff(phase))>np.pi*0.8)[0]
    head_curv_i = head_curvature[i]
    n_jump = len(phase_jump)
    n_record = (n_jump-2)//m
    for j in range(n_record):
        start = phase_jump[j*m]
        end = phase_jump[(j+1)*m]
        d_orientation.append(orientation_i[end]-orientation_i[start])
        d_DV_bias.append(np.sum(head_curv_i[start:end])*0.02)


    # ## Plot to test the correctness of the code
    # plt.figure(dpi=300)
    # plt.plot(orientation_i*15)
    # plt.plot(head_curv_i)
    # plt.plot(np.cumsum(head_curv_i)*0.02)
    # plt.vlines(phase_jump,-50,50,'r')
#%% Calculate the correlation between orientation change and dorsal-ventral bias
do = np.array(d_orientation)
db = np.array(d_DV_bias)
# Correlation between orientation change and dorsal-ventral bias
cov = (do.T @ db/len(do) - do.mean()*db.mean())/(do.std()*db.std())
# Do a linear regression with sklearn
from sklearn.linear_model import LinearRegression
reg = LinearRegression().fit(db.reshape(-1,1),do)
alpha = reg.intercept_
beta = reg.coef_[0]
R2 = reg.score(db.reshape(-1,1),do)
x_range = np.linspace(-15,22,1000)
fig,ax = plt.subplots(1,1,figsize=(5,5),dpi=300)
ax.plot(db,do,'k.')
ax.plot(x_range,alpha+beta*x_range,'r')
ax.set_ylabel('Orientation change (rad)',fontsize=22)
ax.set_xlabel('Dorsoventral bias (s/L)',fontsize=22)
# ax.set_xlim([-15,22])
ax.tick_params(axis='both', which='major', labelsize=20)
# set linewidth of the frame
plt.setp(ax.spines.values(), linewidth=2)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
# plt.savefig('Fig2c_DV_bias_orientation_change.pdf')

#%% Testing the linear relationships with statsmodels
do = np.array(d_orientation)
db = np.array(d_DV_bias)
X = sm.add_constant(db)
model = sm.OLS(do,X)
results = model.fit()
print(results.summary())
p_value = model.fit().pvalues[1]
print('p value: ',p_value)







# %%
# Linear regression of excursion amplitude vs. orientation
# Boostrapping to estimate the 95% confidence interval of regression coefficients



# 1. Get the excursion-orientation data pairs
with open('excursion_data/excursion_trial_info_dict_MN_ablation.pkl','rb') as f:
    excursion_trial_info_dict = pickle.load(f)
excursion_info = excursion_trial_info_dict[name]
reorientation_ls = []
excursion_vec_ls = []
bin_edge = [-6,-3,0,3,6]
m = 3 # number of eigenworm space cycles
shaffle = True
# Set the random seed
np.random.seed(3)
for i in range(head_curvature.shape[0]):
    if (np.isnan(full_curvature[i])).any():
        continue
    sf = np.sign(np.random.rand()-0.5) if shaffle else 1
    curvature_i = full_curvature[i]
    proj_i = curvature_i.T @ eig_worms[:,:2]
    phase = np.arctan2(proj_i[:,0], proj_i[:,1])
    orientation_i = orientation_data[i] * sf
    phase_jump = np.where(np.abs(np.diff(phase))>np.pi*0.8)[0]
    head_curv_i = head_curvature[i] * sf
    n_jump = len(phase_jump)
    n_record = (n_jump-2)//m
    excursion_ls = excursion_info[i]['excursion_ls']
    excursion_start_ls = [excursion['start'] for excursion in excursion_ls]
    for j in range(n_record):
        start = phase_jump[j*m]
        end = phase_jump[(j+1)*m]
        excursion_vec = np.zeros(len(bin_edge)+1)
        # index of excursions whose start time is in the cycle
        excursion_in_cycle = np.where((np.array(excursion_start_ls)>start) & (np.array(excursion_start_ls)<end))[0]
        if len(excursion_in_cycle) == 0:
            continue
        for index in excursion_in_cycle:
            # excursion_amp = (excursions[index]['sign'].astype(np.float32))*abs(head_curvature_sm[excursions[index]['start']]-head_curvature_sm[excursions[index]['end']])
            # excursion_amp = (np.sign(head_curvature_sm[excursions[index]['pre_peak']]))*abs(head_curvature_sm[excursions[index]['start']]-head_curvature_sm[excursions[index]['end']])
            excursion_amp = (np.sign((head_curv_i[excursion_ls[index]['start']] + head_curv_i[excursion_ls[index]['end']])/2))*abs(head_curv_i[excursion_ls[index]['start']]-head_curv_i[excursion_ls[index]['end']])
            vec_index = np.digitize(excursion_amp, bin_edge)
            excursion_vec[vec_index] += 1
        excursion_vec_ls.append(excursion_vec)
        ori_change = orientation_i[end]-orientation_i[start]
        reorientation_ls.append(ori_change)
# %%
excursion_vec_arr = np.array(excursion_vec_ls) # Nx(len(bin_edge)+1)
reorientation_arr = np.array(reorientation_ls) # N
# Train and test data set: train on randomly chosen 100 data points
N_train = len(excursion_vec_arr)-1
train_index = np.random.choice(np.arange(excursion_vec_arr.shape[0]),N_train,replace=False)
test_index = np.array(list(set(np.arange(excursion_vec_arr.shape[0]))-set(train_index)))
excursion_vec_arr_train = excursion_vec_arr[train_index]
reorientation_arr_train = reorientation_arr[train_index]
excursion_vec_arr_test = excursion_vec_arr[test_index]
reorientation_arr_test = reorientation_arr[test_index]
# least square regression and boostrapping
coef_ls = []
N_bootstrap = 10000
for i in range(N_bootstrap):
    resample_ind = np.random.choice(np.arange(excursion_vec_arr_train.shape[0]),excursion_vec_arr_train.shape[0],replace=True)
    excursion_vec_resample = excursion_vec_arr_train[resample_ind]
    reorientation_resample = reorientation_arr_train[resample_ind]
    coef = np.linalg.lstsq(excursion_vec_resample, reorientation_resample, rcond=None)[0]
    coef_ls.append(coef)
coef_ls = np.array(coef_ls)

coef_cat_ls = [coef_ls[:,i] for i in range(6)]
color_ls = ['#708ac1']*6
size = (3,2.5)
y_ticks = [-0.6,-0.3,0,0.3,0.6]
save_path = 'Fig2d_excursion_orientation_regression_bootstrap_dist.pdf'
WormTool.statistic_plot.plot_violin(coef_cat_ls,['<-6','-6~-3','-3~0','0~3','3~6','>6'],'Orientation change (rad)',color_ls=color_ls,size=size,y_ticks=y_ticks,save_path=save_path)

# # %% Plot the bootstrap distribution of regression coefficients
# plt.figure(dpi=300,figsize=(5,5))
# for i in range(len(bin_edge)+1):
#     plt.hist(coef_ls[:,i],bins=100,alpha=0.5,)
# plt.xlabel('Regression coefficients',fontsize=14)
# plt.ylabel('Frequency',fontsize=14)
# plt.legend(['<-6','-6~-3','-3~0','0~3','3~6','>6'],fontsize=14)
# plt.savefig('Fig2d_excursion_orientation_regression_bootstrap_dist.pdf')
# %% Plot the coefficients bar and 95% confidence interval
coef_mean = coef_ls.mean(axis=0)
coef_std = coef_ls.std(axis=0)
# plt.figure(dpi=300)
# plt.bar([-7.5,-4.5,-1.5,1.5,4.5,7.5], coef_mean, yerr=coef_std*1.96,)
# plt.xlabel('Excursion amplitude',fontsize=14)
# plt.ylabel('Orientation change (rad)',fontsize=14)
# plt.xticks([-7.5,-4.5,-1.5,1.5,4.5,7.5],['<-6','-6~-3','-3~0','0~3','3~6','>6'],fontsize=14)
# plt.savefig('Fig2d_excursion_orientation_regression.png')
fig,ax = plt.subplots(1,1,figsize=(5,5),dpi=300)
ax.bar([-7.5,-4.5,-1.5,1.5,4.5,7.5], coef_mean, yerr=coef_std*1.96,)
ax.set_xlabel('Head-casts amplitude',fontsize=22)
ax.set_ylabel('Orientation change (rad)',fontsize=22)
ax.set_xticks([-7.5,-4.5,-1.5,1.5,4.5,7.5])
ax.set_xticklabels(['<-6','-6~-3','-3~0','0~3','3~6','>6'],fontsize=20)
ax.tick_params(axis='y', which='major', labelsize=20)
plt.setp(ax.spines.values(), linewidth=2)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.savefig('Fig2d_excursion_orientation_regression.pdf')
# %% Test the performance of the regression model on the test data set
# test_reorientation_pred = excursion_vec_arr_test @ coef_mean
# test_reorientation_true = reorientation_arr_test
# # test_R2 = 1 - np.sum((test_reorientation_pred-test_reorientation_true)**2)/np.sum((test_reorientation_true)**2)
# print('Test R^2: ',test_R2)
# %%
# Leave one out cross validation
N = excursion_vec_arr.shape[0]
pred_reorientation = []
for i in range(N):
    excursion_vec_train = np.delete(excursion_vec_arr,i,axis=0)
    reorientation_train = np.delete(reorientation_arr,i,axis=0)
    coef = np.linalg.lstsq(excursion_vec_train, reorientation_train, rcond=None)[0]
    pred_reorientation.append(excursion_vec_arr[i] @ coef)
pred_reorientation = np.array(pred_reorientation)
R2 = 1 - np.sum((pred_reorientation-reorientation_arr)**2)/np.sum((reorientation_arr)**2)
print('Leave one out cross validation R^2: ',R2)
plt.figure(dpi=300)
plt.plot(reorientation_arr,pred_reorientation,'k.',markersize=3)
plt.xlabel('True orientation change (rad)',fontsize=20)
plt.ylabel('Predicted orientation change (rad)',fontsize=20)
plt.hlines(0,-2,2,'r',linestyles='--')
plt.vlines(0,-2,2,'r',linestyles='--')
plt.xlim([-2,2])
plt.ylim([-1,1])






# %% 
# Find the linear relationship between excursion amplitude and the dorsal-ventral bias



# 1. Get the excursion-orientation data pairs
name = 'N2'
with open('excursion_trial_info_dict_MN_ablation.pkl','rb') as f:
    excursion_trial_info_dict = pickle.load(f)
excursion_info = excursion_trial_info_dict[name]
dv_bias_ls = []
excursion_vec_ls = []
bin_edge = [-6,-3,0,3,6]
m = 3 # number of eigenworm space cycles
shaffle = False
# Set the random seed
np.random.seed(1)
for i in range(head_curvature.shape[0]):
    if (np.isnan(full_curvature[i])).any():
        continue
    sf = np.sign(np.random.rand()-0.5) if shaffle else 1
    curvature_i = full_curvature[i]
    proj_i = curvature_i.T @ eig_worms[:,:2]
    phase = np.arctan2(proj_i[:,0], proj_i[:,1])
    orientation_i = orientation_data[i] * sf
    phase_jump = np.where(np.abs(np.diff(phase))>np.pi*0.8)[0]
    head_curv_i = head_curvature[i] * sf
    n_jump = len(phase_jump)
    n_record = (n_jump-2)//m
    excursion_ls = excursion_info[i]['excursion_ls']
    excursion_start_ls = [excursion['start'] for excursion in excursion_ls]
    for j in range(n_record):
        start = phase_jump[j*m]
        end = phase_jump[(j+1)*m]
        excursion_vec = np.zeros(len(bin_edge)+1)
        # index of excursions whose start time is in the cycle
        excursion_in_cycle = np.where((np.array(excursion_start_ls)>start) & (np.array(excursion_start_ls)<end))[0]
        if len(excursion_in_cycle) == 0:
            continue
        for index in excursion_in_cycle:
            # excursion_amp = (excursions[index]['sign'].astype(np.float32))*abs(head_curvature_sm[excursions[index]['start']]-head_curvature_sm[excursions[index]['end']])
            # excursion_amp = (np.sign(head_curvature_sm[excursions[index]['pre_peak']]))*abs(head_curvature_sm[excursions[index]['start']]-head_curvature_sm[excursions[index]['end']])
            excursion_amp = (np.sign((head_curv_i[excursion_ls[index]['start']] + head_curv_i[excursion_ls[index]['end']])/2))*abs(head_curv_i[excursion_ls[index]['start']]-head_curv_i[excursion_ls[index]['end']])
            vec_index = np.digitize(excursion_amp, bin_edge)
            excursion_vec[vec_index] += 1
        excursion_vec_ls.append(excursion_vec)
        dv_bias_ls.append(np.sum(head_curv_i[start:end])*0.02)
        
# %%
excursion_vec_arr = np.array(excursion_vec_ls) # Nx(len(bin_edge)+1)
dv_bias_arr = np.array(dv_bias_ls) # N
coef = np.linalg.lstsq(excursion_vec_arr, dv_bias_ls, rcond=None)[0]
R2 = 1 - np.sum((excursion_vec_arr @ coef-dv_bias_arr)**2)/np.sum((dv_bias_arr)**2)
print('R^2: ',R2)
plt.bar([-7.5,-4.5,-1.5,1.5,4.5,7.5], coef)
plt.xlabel('Excursion amplitude',fontsize=14)
plt.ylabel('Dorsal-ventral bias (s*L^-1)',fontsize=14)
plt.xticks([-7.5,-4.5,-1.5,1.5,4.5,7.5],['<-6','-6~-3','-3~0','0~3','3~6','>6'],fontsize=14)
plt.show()
# %%












# %%
# Do the bias - reorientation analysis for concatenated trials
name = 'SMBk'
eig_worms = np.load(f'../excursion/Eigenworm_basis/eigenworms_{name}.npy')
data = np.load('../neuron_ablations/processed/{}.npy'.format(name),allow_pickle=True)
head_curvature = data[:,1]
full_curvature = data[:,2]
orientation_data = data[:,3]

# Get the indices of individual worms
id_file =  '../neuron_ablations/frame_calibration_checked/framecali_wen1119_SMBk.xlsx'
# id_file =  '../neuron_ablations/frame_calibration_checked/framecali_wen1101_RMDk.xlsx'
# id_file =  '../neuron_ablations/frame_calibration_checked/framecali_N2.xlsx'
id_data = pd.read_excel(id_file, sheet_name=None)
keys = list(id_data.keys())
assert len(keys) == 1, 'More than one sheet in the excel file'
id_data = id_data[keys[0]]
# worm id strings are non-empty elements in the first column
id_ind = np.where(~id_data.iloc[:,0].isnull())[0]
num_worm = len(id_ind)
end_ind = len(~id_data.iloc[:,1])
id_ind = np.append(id_ind,end_ind)


def bias_reorien(ind_ls):
    d_orientation = []
    d_DV_bias = []
    # DV_shaffle = True
    m = 2
    for i in ind_ls:
        if (np.isnan(full_curvature[i])).any():
            continue
    # i = 8
    # if True:
        curvature_i = full_curvature[i]
        proj_i = curvature_i.T @ eig_worms[:,:2]
        phase = np.arctan2(proj_i[:,0], proj_i[:,1])
        orientation_i = orientation_data[i]
        phase_jump = np.where(np.abs(np.diff(phase))>np.pi*0.8)[0]
        head_curv_i = head_curvature[i]
        n_jump = len(phase_jump)
        n_record = (n_jump-1)//m
        for j in range(n_record):
            start = phase_jump[j*m]
            end = phase_jump[(j+1)*m]
            d_orientation.append(orientation_i[end]-orientation_i[start])
            d_DV_bias.append(np.sum(head_curv_i[start:end])*0.02)
    return d_orientation, d_DV_bias


#%% Load curvature data
d_orientation_centered_ls = []
d_DV_bias_centered_ls = []
fig,ax = plt.subplots(1,1,figsize=(5,5),dpi=300)
for i in range(num_worm):
    d_orientation, d_DV_bias = bias_reorien(np.arange(id_ind[i],id_ind[i+1]))
    d_orientation_centered = d_orientation - np.mean(d_orientation)
    d_DV_bias_centered = d_DV_bias - np.mean(d_DV_bias)
    print(np.arange(id_ind[i],id_ind[i+1]))
    # plt.plot(d_DV_bias,d_orientation,'.',markersize=3)
    ax.plot(d_DV_bias_centered,d_orientation_centered,'.',markersize=5)
    d_orientation_centered_ls.append(d_orientation_centered)
    d_DV_bias_centered_ls.append(d_DV_bias_centered)
d_orientation_centered_arr = np.concatenate(d_orientation_centered_ls)
d_DV_bias_centered_arr = np.concatenate(d_DV_bias_centered_ls)
# least square regression and r2
import scipy.stats
slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(d_DV_bias_centered_arr,d_orientation_centered_arr)
x_range = np.linspace(-9,9,100)
y_hat = slope*x_range + intercept
ax.plot(x_range,y_hat,'r')
ax.set_xlim([-10,10])
ax.set_ylim([-2,2])
ax.text(-9,1.5,'r = {:.2f}'.format(r_value),fontsize=14)
ax.set_xticks([-9,-6,-3,0,3,6,9])
ax.set_yticks([-2,-1,0,1,2])
ax.set_xlabel('Dorsal-ventral bias (s/L)',fontsize=14)
ax.set_ylabel('Orientation change (rad)',fontsize=14)
# title
ax.set_title(name)
# plt.savefig('Fig2c_DV_bias_orientation_change_centered.pdf')

    
# %%

# %%
with open('excursion_trial_info_dict_MN_ablation.pkl','rb') as f:
    excursion_trial_info_dict = pickle.load(f)
    excursion_info = excursion_trial_info_dict[name]
def excursion_vec_orientation(ind_ls):
    d_orientation = []
    excursion_vec_ls = []
    bin_edge = [-0.6,-0.3,0,0.3,0.6]
    # bin_edge = [-0.5,0,0.5]
    shuffle = False
    # set random seed
    # np.random.seed(2) # for RMDk
    np.random.seed(4) # for SMBk

    m = 2 # number of eigenworm space cycles
    for i in ind_ls:
        if (np.isnan(full_curvature[i])).any():
            continue
    # i = 8
    # if True:
        curvature_i = full_curvature[i]
        proj_i = curvature_i.T @ eig_worms[:,:2]
        phase = np.arctan2(proj_i[:,0], proj_i[:,1])
        orientation_i = orientation_data[i]
        phase_jump = np.where(np.abs(np.diff(phase))>np.pi*0.8)[0]
        head_curv_i = head_curvature[i]
        n_jump = len(phase_jump)
        n_record = (n_jump-1)//m

        excursion_ls = excursion_info[i]['excursion_ls']
        excursion_start_ls = [excursion['start'] for excursion in excursion_ls]
        excursion_rel_amp_ls = excursion_info[i]['excursion_rel_amps']
        for j in range(n_record):
            start = phase_jump[j*m]
            end = phase_jump[(j+1)*m]
            excursion_vec = np.zeros(len(bin_edge)+1)
            # index of excursions whose start time is in the cycle
            excursion_in_cycle = np.where((np.array(excursion_start_ls)>start) & (np.array(excursion_start_ls)<end))[0]
            if len(excursion_in_cycle) == 0:
                continue
            shf = np.sign(np.random.rand()-0.5) if shuffle else 1
            for index in excursion_in_cycle:
                excursion_amp = shf*(np.sign((head_curv_i[excursion_ls[index]['start']] + head_curv_i[excursion_ls[index]['end']])/2))*abs(head_curv_i[excursion_ls[index]['start']]-head_curv_i[excursion_ls[index]['end']])/excursion_info[i]['amp']
                # excursion_amp = shf*(np.sign((head_curv_i[excursion_ls[index]['start']] + head_curv_i[excursion_ls[index]['end']])/2))*excursion_rel_amp_ls[index]
                vec_index = np.digitize(excursion_amp, bin_edge)
                excursion_vec[vec_index] += 1
            d_orientation.append((orientation_i[end]-orientation_i[start])*shf)
            excursion_vec_ls.append(excursion_vec)
    d_orientation_arr = np.array(d_orientation)
    excursion_vec_arr = np.array(excursion_vec_ls)
    if len(d_orientation) == 0:
        # print(d_orientation_arr.shape,excursion_vec_arr.shape)
        return d_orientation_arr, excursion_vec_arr.reshape(-1,len(bin_edge)+1)
    else:
        return np.array(d_orientation), np.array(excursion_vec_ls)

d_orientation_centered_ls = []
excursion_vec_ls = []
for i in range(num_worm):
    d_orientation, excursion_vec = excursion_vec_orientation(np.arange(id_ind[i],id_ind[i+1]))
    d_orientation_centered = d_orientation - np.mean(d_orientation)
    excursion_vec_ls.append(excursion_vec)
    d_orientation_centered_ls.append(d_orientation_centered)
d_orientation_centered_arr = np.concatenate(d_orientation_centered_ls)
excursion_vec_arr = np.concatenate(excursion_vec_ls,axis=0)
# least square regression and r2
coef = np.linalg.lstsq(excursion_vec_arr, d_orientation_centered_arr, rcond=None)[0]
fig,ax = plt.subplots(1,1,figsize=(5,5),dpi=300)
ax.bar([-0.75,-0.45,-0.15,0.15,0.45,0.75], coef,width=0.2)
# ax.bar([-0.75,-0.25,0.25,0.75], coef,width=0.2)
ax.set_xlabel('Excursion amplitude')
ax.set_ylabel('Dorsal-ventral bias (s*L^-1)')
ax.set_xticks([-0.75,-0.45,-0.15,0.15,0.45,0.75],['<-.6','-.6~-.3','-.3~0','0~.3','.3~.6','>.6'],fontsize=14)
# ax.set_xticks([-0.75,-0.25,0.25,0.75],['<-0.5','-0.5~0','0~0.5','>0.5'],fontsize=14)
ax.set_title(name)
ax.set_ylim([-0.3,0.3])
plt.show()

    
# %%
