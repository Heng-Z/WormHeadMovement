#%%
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42
plt.rcParams['font.family'] = 'Arial'
import scipy.stats as stats
import scipy.io
import WormTool
import pickle
from find_excursion import ExcurInfo

#%%
def simple_beeswarm(y, nbins=None):
    """
    Returns x coordinates for the points in ``y``, so that plotting ``x`` and
    ``y`` results in a bee swarm plot.
    """
    y = np.asarray(y)
    if nbins is None:
        nbins = len(y) // 6

    # Get upper bounds of bins
    x = np.zeros(len(y))
    ylo = np.min(y)
    yhi = np.max(y)
    dy = (yhi - ylo) / nbins
    ybins = np.linspace(ylo + dy, yhi - dy, nbins - 1)

    # Divide indices into bins
    i = np.arange(len(y))
    ibs = [0] * nbins
    ybs = [0] * nbins
    nmax = 0
    for j, ybin in enumerate(ybins):
        f = y <= ybin
        ibs[j], ybs[j] = i[f], y[f]
        nmax = max(nmax, len(ibs[j]))
        f = ~f
        i, y = i[f], y[f]
    ibs[-1], ybs[-1] = i, y
    nmax = max(nmax, len(ibs[-1]))

    # Assign x indices
    dx = 1 / (nmax // 2)
    for i, y in zip(ibs, ybs):
        if len(i) > 1:
            j = len(i) % 2
            i = i[np.argsort(y)]
            a = i[j::2]
            b = i[j+1::2]
            x[a] = (0.5 + j / 3 + np.arange(len(b))) * dx
            x[b] = (0.5 + j / 3 + np.arange(len(b))) * -dx

    return x

def freq_t_test(name1,name2,file1,file2=None,rel_threshold=0.3):
    if file2 is None:
        info = ExcurInfo(file1)
        exp1 = info.group_info[name1]
        exp2 = info.group_info[name2]
    else:
        info1 = ExcurInfo(file1)
        info2 = ExcurInfo(file2)
        exp1 = info1.group_info[name1]
        exp2 = info2.group_info[name2]
    freq1 = []
    freq2 = []
    for trial in exp1:
        if trial is None:
            continue
        rel_amp = trial['excursion_rel_amps']
        # number of excursions with relative amplitude larger than threshold
        num_excursion = np.sum(rel_amp>rel_threshold)
        if num_excursion>0:
            freq1.append(num_excursion/trial['length']/0.02)
        else:
            freq1.append(0)
    for trial in exp2:
        if trial is None:
            continue
        rel_amp = trial['excursion_rel_amps']
        # number of excursions with relative amplitude larger than threshold
        num_excursion = np.sum(rel_amp>rel_threshold)
        if num_excursion>0:
            freq2.append(num_excursion/trial['length']/0.02)
        else:
            freq2.append(0)

    stat, p = stats.mannwhitneyu(freq1,freq2)
    print(name1,'vs',name2,'p value =',p)
    # determine number of stars according to p value
    if p<0.0001:
        star = '****'
    elif p<0.001:
        star = '***'
    elif p<0.01:
        star = '**'
    elif p<0.05:
        star = '*'
    else:
        star = 'n.s.'
    fig,ax = plt.subplots(1,1,figsize=(4,5),dpi=300)
    # colors = ['#33a1c9','#e2cf56']
    colors = ['#a7d99e','#f27f91']
    ax.bar([1,2],[np.mean(freq1),np.mean(freq2)],color=colors,alpha=0.7,width=0.5)
    # ax.errorbar([1,2],[np.mean(freq1),np.mean(freq2)],yerr=[np.std(freq1),np.std(freq2)],fmt='',color='black',capsize=5, capthick=2,ls='',linewidth=2)
    # scatter plot of individual data points
    jitter = 0.1
    ax.scatter(simple_beeswarm(freq1)*0.3+1,freq1,color=colors[0],alpha=1)
    ax.scatter(simple_beeswarm(freq2)*0.3+2,freq2,color=colors[1],alpha=1)
    ax.set_xticks([1,2])
    ax.set_xlim([0.3,2.5])
    ax.set_xticklabels([name1,name2],fontsize=15)
    # ax.set_ylabel('Large Head-casting Frequency (Hz)',fontsize=22)
    ax.set_ylabel('Head casts rate (Hz)',fontsize=22)
    # ax.set_ylim([0,0.7])
    ax.set_yticks(np.arange(0,0.9,0.2))
    ax.set_yticklabels([ '{:.1f}'.format(num) for num in np.arange(0,0.9,0.2)],fontsize=26)
    ax.tick_params(axis='both', which='major', length=5, width=2,direction='in')
    ax.text(2,0.8,star,fontsize=22,ha='center', va='center')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.setp(ax.spines.values(), lw=2)
    plt.savefig('PSi_large_head_casting_freq.pdf')
freq_t_test('PSi_ctrl','PSi','excursion_trial_info_dict_PSi.pkl',rel_threshold=0.3)
# freq_t_test('IS_åPSi_ctrl','IS_PSi','excursion_trial_info_dict_IS_PSi.pkl',rel_threshold=0.3)
# %%
