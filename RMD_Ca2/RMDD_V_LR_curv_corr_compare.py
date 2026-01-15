#%%
import numpy as np
import matplotlib.pyplot as plt
import WormTool
from scipy.stats import mannwhitneyu
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42
plt.rcParams['font.family'] = 'Arial'
# %%
RMDD_curv_corr_ls = np.abs(np.loadtxt('RMDD_curv_corr_ls.csv', delimiter=','))
RMDV_curv_corr_ls = np.abs(np.loadtxt('RMDV_curv_corr_ls.csv', delimiter=','))
RMDLR_curv_corr_ls = np.abs(np.loadtxt('RMDLR_curv_corr_ls.csv', delimiter=','))
# Mann-Whitney U test
p1 = mannwhitneyu(RMDLR_curv_corr_ls, RMDD_curv_corr_ls)[1]
p2 = mannwhitneyu(RMDLR_curv_corr_ls, RMDV_curv_corr_ls)[1]
p3 = mannwhitneyu(RMDD_curv_corr_ls, RMDV_curv_corr_ls)[1]
print(p1, p2, p3)
print(f'num. of corr: RMDLR: {len(RMDLR_curv_corr_ls)}, RMDD: {len(RMDD_curv_corr_ls)}, RMDV: {len(RMDV_curv_corr_ls)}')
fig,ax = plt.subplots(1,1,figsize=(2,1.8),dpi=300)
WormTool.statistic_plot.plot_whisker_box([RMDLR_curv_corr_ls, RMDD_curv_corr_ls, RMDV_curv_corr_ls], ['RMDL/R', 'RMDD', 'RMDV'], 'Correlation Coefficient', label_rot=45, ax=ax)
# remove x tick
ax.set_xticks([])

plt.tight_layout()
plt.savefig('RMDD_V_LR_curv_corr_compare.pdf')
# %%
