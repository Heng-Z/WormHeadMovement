import numpy as np
import scipy.signal
import matplotlib.pyplot as plt

def smooth_1st_derivative(data, window_size=10):
    kernel = np.ones(window_size)
    L = window_size//2
    kernel[:L] = 1
    kernel[L:] = -1
    kernel = kernel/window_size
    data_pad = np.pad(data, (window_size//2,window_size//2), mode='edge')
    conv_data = np.convolve(data_pad, kernel, mode='same')
    return conv_data[window_size//2:-(window_size//2)]

def smooth_2nd_derivative(data, window_size=9):
    kernel = np.ones(window_size)
    assert window_size % 3 == 0, "Window size must be a multiple of 3"
    L = window_size//3
    kernel[:L] = 1
    kernel[L:2*L] = -2
    kernel[2*L:] = 1
    kernel = kernel/window_size
    data_pad = np.pad(data, (window_size//2,window_size//2), mode='edge')
    conv_data = np.convolve(data_pad, kernel, mode='same')
    return conv_data[window_size//2:-(window_size//2)]


def corr_lag(s1,s2,zero_mean=False):
    '''
    Calculate how many step s2 delay s1 using cross-correlation and the maximum correlation coefficient
    '''
    if zero_mean:
        s1 = s1 - s1.mean()
        s2 = s2 - s2.mean()
    corr = scipy.signal.correlate(s2, s1, mode='full')
    lags = scipy.signal.correlation_lags(len(s1), len(s2), mode='full').astype(np.int32)
    valid_range = (lags >= -5) & (lags <= 5)
    corr = corr[valid_range]
    lags = lags[valid_range]
    lag = lags[np.argmax(np.abs(corr))]
    corr_max = np.max(corr)/np.linalg.norm(s1)/np.linalg.norm(s2)
    return corr_max,lag

def curv_Ca2_correlation(curv, Ca2, dt, corr_window_s=3,smooth_window_s=1,plot=False):
    corr_window = int(corr_window_s/dt)
    if len(curv)//corr_window == 0:
        return [], [], [], []
    smooth_window = round(int(smooth_window_s/dt)/6)*6
    t = np.arange(len(curv))*dt
    curv_1st_der = smooth_1st_derivative(curv, window_size=smooth_window)
    Ca2_1st_der = smooth_1st_derivative(Ca2, window_size=smooth_window)
    Ca2_2nd_der = smooth_2nd_derivative(Ca2, window_size=smooth_window)
    # cut into segments based on corr_window
    
    t_segments = np.array_split(t, len(t)//corr_window)
    curv_segments = np.array_split(curv_1st_der, len(curv_1st_der)//corr_window)
    Ca2_1st_segments = np.array_split(Ca2_1st_der, len(Ca2_1st_der)//corr_window)
    Ca2_2nd_segments = np.array_split(Ca2_2nd_der, len(Ca2_2nd_der)//corr_window)
    # calculate correlation for each segment
    curv_Ca2_1st_corr_ls = []
    curv_Ca2_2nd_corr_ls = []
    curv_Ca2_1st_lag_ls = []
    curv_Ca2_2nd_lag_ls = []
    for i in range(len(curv_segments)):
        corr,lag = corr_lag(curv_segments[i], Ca2_1st_segments[i])
        curv_Ca2_1st_corr_ls.append(corr)
        curv_Ca2_1st_lag_ls.append(lag)
        corr,lag = corr_lag(curv_segments[i], Ca2_2nd_segments[i])
        curv_Ca2_2nd_corr_ls.append(corr)
        curv_Ca2_2nd_lag_ls.append(lag)

    if plot:
        fig,ax = plt.subplots(figsize=(5,2.5),dpi=300)
        for i in range(len(t_segments)):
            ax.plot(t_segments[i],(curv_segments[i]-curv_segments[i].mean())/np.linalg.norm(curv_segments[i]),'k')
            ax.plot(t_segments[i],(Ca2_1st_segments[i])/np.linalg.norm(Ca2_1st_segments[i]),'r')
            ax.plot(t_segments[i],(Ca2_2nd_segments[i])/np.linalg.norm(Ca2_2nd_segments[i]),'m')
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Normalized Curvature and Ca2+ derivatives')
        ax.legend()
        plt.show()

    return curv_Ca2_1st_corr_ls, curv_Ca2_2nd_corr_ls, curv_Ca2_1st_lag_ls, curv_Ca2_2nd_lag_ls
