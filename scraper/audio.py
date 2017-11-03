import numpy as np
from scipy import signal
import matplotlib.pyplot as plt

M = 512
window_shift = 256

def get_spectrogram(audio, sampling_rate):
    return signal.spectrogram(audio, fs=sampling_rate, window='hanning',
                              nperseg=M, noverlap=M-window_shift,
                              detrend=False, scaling='spectrum')

def plot_spectrogram(freqs, times, Sx):
    print(Sx.shape)
    f, ax = plt.subplots()
    ax.pcolormesh(times, freqs / 1000, 10 * np.log10(Sx), cmap='viridis')
    ax.set_ylabel('Frequency [kHz]')
    ax.set_xlabel('Time [s]')
    plt.show()