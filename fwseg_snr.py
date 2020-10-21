import numpy as np
import librosa
import scipy.signal as signal

import os
import pdb

def stft(x, n_fft = 2048, hop_length = 512, win_length = 2048, window = 'hann'):
    if win_length is None:
        win_length = n_fft
    if hop_length is None:
        hop_length = win_length // 4

    window = signal.get_window(window, win_length, fftbins = True)
    window = window.reshape((-1, 1))

    x_frame = librosa.util.frame(x, frame_length=win_length, hop_length=hop_length)

    stft_matrix = np.empty((int(1 + n_fft // 2), x_frame.shape[1]), dtype=np.complex, order='F')

    n_columns = int(librosa.util.MAX_MEM_BLOCK / (stft_matrix.shape[0] *
                                          stft_matrix.itemsize))

    for bl_s in range(0, stft_matrix.shape[1], n_columns):
        bl_t = min(bl_s + n_columns, stft_matrix.shape[1])

        stft_matrix[:, bl_s:bl_t] = np.fft.rfft(window * x_frame[:, bl_s:bl_t],
                                                n = n_fft, axis=0)
    return stft_matrix


def fwSNRseg(wav_ref, wav_deg, sr = 16000, num_crit_band = 25):

    length = np.minimum(len(wav_ref), len(wav_deg))
    wav_deg = wav_deg[0: length]
    wav_ref = wav_ref[0: length]
    win_length   = np.round(30 * sr / 1000).astype(np.int)    # window length in samples, 480 if sr=16000
    hop_length    = np.floor(win_length / 4).astype(np.int)	   # window skip in samples
    max_freq    = sr // 2	   # maximum bandwidth
    n_fft       = 2 ** np.ceil(np.log2(2 * win_length)).astype(np.int)
    n_fftby2    = n_fft // 2		   # FFT size/2
    gamma=0.2
    eps = 1e-8

    centre_freq = np.array([50, 120, 190, 260, 330, 400, 470, 540, 617.372, 703.378, 798.717,
                            904.128, 1020.38, 1148.30, 1288.72, 1442.54, 1610.70, 1794.16, 1993.93,
                            2211.08, 2446.71, 2701.97, 2978.04, 3276.17, 3597.63])
    bandwidth = np.array([70, 70, 70, 70, 70, 70, 70, 77.3742, 86.0056, 95.3398, 105.411, 116.256,
                            127.914, 140.423, 153.823, 168.154, 183.457, 199.776, 217.153, 235.631,
                            255.255, 276.072, 298.126, 321.465, 346.136])
    bw_min = bandwidth[0]

    min_factor = np.exp(-30 / (2 * 2.303))

    crit_filter = np.zeros([num_crit_band, n_fftby2])
    for ii in range(0, num_crit_band):
        f0 = (centre_freq[ii] / max_freq) * n_fftby2
        bw = (bandwidth[ii] / max_freq) * n_fftby2
        norm_factor = np.log(bw_min) - np.log(bandwidth[ii])
        jj = np.arange(0, n_fftby2)
        crit_filter[ii, :] = np.exp(-11 * ((jj - np.floor(f0)) / bw) ** 2 + norm_factor)
        crit_filter[ii, :] = crit_filter[ii, :] * (crit_filter[ii, :] > min_factor)

    spec_ref = stft(wav_ref, n_fft = n_fft, hop_length = hop_length,
                                win_length = win_length, window = 'hann')
    spec_ref = np.abs(spec_ref)  #shape: f * t
    spec_ref = spec_ref / np.sum(spec_ref[0: n_fftby2, :], axis = 0)
    spec_deg = stft(wav_deg, n_fft = n_fft, hop_length = hop_length,
                                win_length = win_length, window = 'hann')
    spec_deg = np.abs(spec_deg)
    spec_deg = spec_deg / np.sum(spec_deg[0: n_fftby2, :], axis = 0)
    n_frames = spec_ref.shape[1]

    energy_ref = np.zeros([num_crit_band, n_frames])
    energy_deg = np.zeros([num_crit_band, n_frames])
    energy_error = np.zeros([num_crit_band, n_frames])
    W_freq = np.zeros([num_crit_band, n_frames])

    for ii in range(0, num_crit_band):
        energy_ref[ii, :] = np.sum(spec_ref[0: n_fftby2, :].T * crit_filter[ii, :], axis = 1) #shape: t * 1
        energy_deg[ii, :] = np.sum(spec_deg[0: n_fftby2, :].T * crit_filter[ii, :], axis = 1)
        energy_error[ii, :] = np.maximum((energy_ref[ii, :] - energy_deg[ii, :])**2, eps)
        W_freq[ii, :] = energy_ref[ii, :] ** gamma

    SNRlog = 10 * np.log10(energy_ref**2 / energy_error)
    fwSNR = np.sum(W_freq * SNRlog, axis = 0) / np.sum(W_freq, axis = 0)
    fwSNR = np.minimum(np.maximum(fwSNR, -10), 35)
    fwSNR = np.mean(fwSNR)

    return fwSNR

if __name__ == '__main__':
    wav_ref = librosa.core.load("sp04.wav", sr = 8000)[0]
    wav_deg = librosa.core.load("enhanced.wav", sr = 8000)[0]

    score = fwSNRseg(wav_ref, wav_deg, sr = 8000)
    print(score)

    #import scipy.io

    #data = scipy.io.loadmat('wav_sp04_enhanced.mat')
    #wav_ref = data['wav_ref'].astype(np.float).squeeze()
    #wav_deg = data['wav_deg'].astype(np.float).squeeze()
    #score = fwSNRseg(wav_ref, wav_deg, sr = 8000)
    #print(score)
