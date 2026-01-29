#%%
import numpy as np
import matplotlib.pyplot as plt
from ipywidgets import interact, interactive, fixed, interact_manual
import WormTool
# set rcParams for saving figures as pdf
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42
plt.rcParams['font.size'] = 12
plt.rcParams['font.family'] = 'Arial'


# %% with neck curvature and K_slow (sencondary low-pass filtering)
def run_two_timescale_feedback(th1=1,th2=0.6,a_s=1,Az=1.5,save_fig=None):
    core = WormTool.headModel.BistableNeuron(th1=th1,th2=-th1)
    f_rmd = lambda x: float(x>th2)
    dt = 0.005
    if type(a_s) in [int,float] and type(Az) in [int,float]:
        t = np.arange(0, 10, dt)
        a_s = np.ones_like(t)*a_s
        Az = np.ones_like(t)*Az
    elif type(a_s) in [int,float] and type(Az) not in [int,float]:
        t = np.arange(len(Az))*dt
        a_s = np.ones_like(t)*a_s
    elif type(a_s) not in [int,float] and type(Az) in [int,float]:
        t = np.arange(len(a_s))*dt
        Az = np.ones_like(t)*Az
        

    
    Ay =1;tau_s=0.8;tau_f=0.3;b_s=10;b_f = 3
    Kst = np.zeros_like(t)
    Kft = np.zeros_like(t)
    Knt = np.zeros_like(t)
    Mt = np.zeros_like(t)
    Kt = np.zeros_like(t)
    Mus_t = np.zeros_like(t)
    Ks =0;Kf=0;M=1;K=0;Kn=0
    
    for i in range(len(t)):
        core.update(-Ks)
        Vrmd = f_rmd(abs(Kf))
        
        # dKndt = (-Kn + a_s[i]*np.tanh(b*K))/tau_s
        dKndt = (-Kn + a_s[i]*K)/tau_s
        dKsdt = (-Ks + Kn)/tau_f
        dKfdt = (-Kf + K)/tau_f
        Mus =  np.clip(Ay*(2*core.x-1) - Az[i]*np.sign(Kf)*Vrmd,-1,1)
        dMdt = (-M +Mus)/0.1
        dKdt = (-K + M)/0.4
        Ks += dKsdt*dt
        # Ks = a_s[i]*Kn
        Kf += dKfdt*dt
        Kn += dKndt*dt
        M += dMdt*dt
        K += dKdt*dt
        Kst[i] = Ks
        Kft[i] = Kf
        Mt[i] = M
        Kt[i] = K
        Knt[i] = Kn
        Mus_t[i] =  Mus
    fig,ax = plt.subplots(3,1,figsize=(3.8,3.7),dpi=300)
    _ = [ax[i].spines[loc].set_visible(False) for i in range(3) for loc in ['top','right']]
    ax[0].plot(t,Kt,'k',label='K')
    # ax[0].plot(t,Mus_t,'blue',label='M')
    ax[0].set_yticks([-1,0,1])
    # ax[0].axvspan(1800*dt,len(t)*dt,alpha=0.5,color='grey')
    ax[1].hlines([th2],0,t[-1],'g',linestyles='dashed',alpha=0.4)
    ax[1].text(t[-1],th2,'$\\theta_f$',fontsize=12,color='g')
    ax[1].hlines([-th2],0,t[-1],'g',linestyles='dashed',alpha=0.4)
    ax[1].text(t[-1],-th2,'$-\\theta_f$',fontsize=12,color='g')
    ax[1].plot(t,Kft,'tab:green')
    # no x ticks and save the space
    ax[0].set_xticks([])
    ax[1].set_xticks([])

    ax[1].set_ylim(-0.85,0.85)
    ax[1].set_yticks([-0.8,0,0.8])
    ax[2].plot(t,Kst,'tab:blue')
    # ax[2].plot(t,Knt,'r')
    ax[2].hlines([-th1,th1],0,t[-1],'b',linestyles='dashed',alpha=0.4)
    ax[2].text(t[-1],th1,'$\\theta_s$',fontsize=12,color='blue')
    ax[2].text(t[-1],-th1,'$-\\theta_s$',fontsize=12,color='blue')
    ax[2].set_yticks([-2,0,2])
    ax[2].spines['bottom'].set_visible(True)
    ax[2].set_xlabel('Time (s)')
    ax[2].set_ylim(-2,2)

    if save_fig is not None:
        plt.savefig(save_fig)
    plt.show()
    return Kt, Kst, Kft, Knt

# %% Another parameter set
#%% Simulated Neck inhibition
t = np.arange(0,21,0.005)
a_s = np.ones_like(t)*5
a_s[1700:] = 2.7
th1 = 0.8
th2 = 0.3
Az = 1.4
run_two_timescale_feedback(th1=th1,th2=th2,a_s=a_s,Az=Az,save_fig='model_results/model_results_neck_inhibition.pdf')


t = np.arange(0,21,0.005)
Az = np.ones_like(t)*1.4
Az[2100:] = 0
th1 = 0.8
th2 = 0.3
a_s = 5
run_two_timescale_feedback(th1=th1,th2=th2,a_s=a_s,Az=Az,save_fig='model_results/model_results_RMDk.pdf')


# %%
