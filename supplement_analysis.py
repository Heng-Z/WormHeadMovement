#%%
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
import scipy.io
import pickle

def cal_amplitude(kt):
    '''
    Calculate the amplitude of the head curvature.
    Find the peaks with significant prominence and calculate the average hight of the peaks.
    Parameters
    ----------
    kt : array of shape (T,)
        Curvature of the head.

    Returns
    -------
    amplitude : np array of shape (N,)
        Array of amplitude of the head curvature.
    '''
    # smooth the head curvature
    kt_sm = np.convolve(kt,np.ones(7)/7,mode='same')
    # find positive peaks
    k_min = np.min(kt_sm)
    k_max = np.max(kt_sm)
    reference = (k_max-k_min)/2
    pos_peaks, _ = find_peaks(kt_sm,prominence=reference*0.8,height=reference*0.6)
    # find negative peaks
    neg_peaks, _ = find_peaks(-kt_sm,prominence=reference*0.8,height=reference*0.6)
    # Calculate the average height of the peaks
    peak_ave = np.mean(np.concatenate((kt_sm[pos_peaks],-kt_sm[neg_peaks])))
    # Calculate the amplitude

    return np.concatenate((kt_sm[pos_peaks],-kt_sm[neg_peaks]))

def compare_amp(name1,name2,key1=None,key2=None):
    data1 = scipy.io.loadmat('../neuron_ablations/hb_dynamics/'+name1+'.mat')
    data1 = data1[list(data1.keys())[-1]] if key1 is None else data1[key1]
    data2 = scipy.io.loadmat('../neuron_ablations/hb_dynamics/'+name2+'.mat')
    data2 = data2[list(data2.keys())[-1]] if key2 is None else data2[key2]
    amp1 = []
    amp2 = []
    for i in range(data1.shape[0]):
        kt = data1[i,0][:,1]
        amp1.extend(cal_amplitude(kt))
    for i in range(data2.shape[0]):
        kt = data2[i,0][:,1]
        amp2.extend(cal_amplitude(kt))
    amp1 = np.array(amp1)
    amp2 = np.array(amp2)
    plt.figure(dpi=300)
    plt.hist(amp1,bins=45,range=(0,15),alpha=0.5,label=name1,density=True)
    plt.hist(amp2,bins=45,range=(0,15),alpha=0.5,label=name2,density=True)
    plt.legend()
    plt.xlabel('Head Bending Amplitude')
    plt.ylabel('Density')
    plt.show()

#%%
# compare_amp('Ai','unc-7_unc-9_mutant_Ai')
# compare_amp('IS_PSi_ctrl','IS_PSi','dyn_HB','dyn_HB')
compare_amp('PSi_ctrl','PSi')

#%%
# Plot large excursion frequency v.s. head bending amplitude for different strains
with open('excursion_trial_info_dict_MN_ablation.pkl','rb') as f:
    excursion_info_MNk = pickle.load(f)
with open('excursion_trial_info_dict_MN_inhibition.pkl','rb') as f:
    excursion_info_MNi = pickle.load(f)
with open('excursion_trial_info_dict_PSi.pkl','rb') as f:
    excursion_info_PSi = pickle.load(f)
with open('excursion_trial_info_dict_RIA_twk_18.pkl','rb') as f:
    excursion_info_RIA = pickle.load(f)
with open('excursion_trial_info_dict_opto.pkl','rb') as f:
    excursion_info_opto = pickle.load(f)
with open('excursion_trial_info_dict_GJk.pkl','rb') as f:
    excursion_info_GJk = pickle.load(f)
with open('excursion_trial_info_dict_IS_PSi.pkl','rb') as f:
    excursion_info_IS_PSi = pickle.load(f)
# Concate the these dicts
excursion_trial_info_dict = {}
dicts = [excursion_info_MNk,excursion_info_IS_PSi]
# dicts = [excursion_info_MNk,excursion_info_MNi]
# dicts = [excursion_info_MNk,excursion_info_PSi,excursion_info_RIA]
# dicts = [excursion_info_MNk,excursion_info_opto]
for d in dicts:
    for key in d.keys():
        excursion_trial_info_dict[key] = d[key]
name_ls = list(excursion_trial_info_dict.keys())
threshold = 0.3
dt = 0.02
# excursion_trial_info_dict = excursion_info_MNk
# name_ls = list(excursion_trial_info_dict.keys())
# %%
# Plot a confidence ellipse
# https://matplotlib.org/devdocs/gallery/statistics/confidence_ellipse.html
from matplotlib.patches import Ellipse
import matplotlib.transforms as transforms 
def confidence_ellipse(x, y, ax, n_std=3.0, facecolor='none', **kwargs):
    """
    Create a plot of the covariance confidence ellipse of *x* and *y*.

    Parameters
    ----------
    x, y : array-like, shape (n, )
        Input data.

    ax : matplotlib.axes.Axes
        The axes object to draw the ellipse into.

    n_std : float
        The number of standard deviations to determine the ellipse's radiuses.

    **kwargs
        Forwarded to `~matplotlib.patches.Ellipse`

    Returns
    -------
    matplotlib.patches.Ellipse
    """
    if x.size != y.size:
        raise ValueError("x and y must be the same size")

    cov = np.cov(x, y)
    pearson = cov[0, 1]/np.sqrt(cov[0, 0] * cov[1, 1])
    # Using a special case to obtain the eigenvalues of this
    # two-dimensional dataset.
    ell_radius_x = np.sqrt(1 + pearson)
    ell_radius_y = np.sqrt(1 - pearson)
    ellipse = Ellipse((0, 0), width=ell_radius_x * 2, height=ell_radius_y * 2,
                      facecolor=facecolor, **kwargs)

    # Calculating the standard deviation of x from
    # the squareroot of the variance and multiplying
    # with the given number of standard deviations.
    scale_x = np.sqrt(cov[0, 0]) * n_std
    mean_x = np.mean(x)

    # calculating the standard deviation of y ...
    scale_y = np.sqrt(cov[1, 1]) * n_std
    mean_y = np.mean(y)
    # print(mean_x,mean_y,ell_radius_x,ell_radius_y)
    transf = transforms.Affine2D() \
        .rotate_deg(45) \
        .scale(scale_x, scale_y) \
        .translate(mean_x, mean_y)

    ellipse.set_transform(transf + ax.transData)
    end_point = np.array([-1.3*ell_radius_y/np.sqrt(2),1.3*ell_radius_y/np.sqrt(2)])*np.array([scale_x,scale_y])+np.array([mean_x,mean_y])
    ax.text(end_point[0],end_point[1],kwargs['label'],fontsize=14,color=kwargs['edgecolor'])
    return ax.add_patch(ellipse)
fig, ax = plt.subplots(dpi=300)
i = 0 
for name in name_ls:
    if name in ['SMDSMBk','RMDSMBk','RMDSMDk','RMDk_PSi_ctrl','RMDk_PSi', 'RMDSMDk_PSi_ctrl', 'RMDSMDk_PSi',]:
    # if name in ['SMDSMBk']:
        continue
    amp = []
    freq = []
    for trial in excursion_trial_info_dict[name]:
        num_large_excur = np.sum(trial['excursion_rel_amps']>threshold)
        if num_large_excur<2:
            continue
        amp.append(trial['amp'])
        freq.append(num_large_excur/trial['length']/dt)
    amp,freq = np.array(amp),np.array(freq)
    print(name+' len amp = '+str(len(amp)))
    confidence_ellipse(amp,freq,ax,n_std=2,label=name,edgecolor='C'+str(i),linewidth=3)
    ax.plot(amp,freq,'.',color='C'+str(i),alpha=0.5)
    i+=1
ax.set_xlim([3,14])
ax.set_ylim([0,1.6])
plt.legend()
plt.xlabel('Head Bending Amplitude')
plt.ylabel('Large Excursion Occuring Frequency (Hz)')
plt.show()

# %%
import matplotlib.pyplot as plt
import matplotlib.patches as patches

fig, ax = plt.subplots()

# Create an Ellipse object
ellipse = patches.Ellipse((0.5, 0.5), 0.6, 0.2, color='r')

# Add the Ellipse object to the Axes object
ax.add_patch(ellipse)

# Set the x and y limits of the Axes object
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)

# Show the plot
plt.show()
# %%
