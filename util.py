import numpy as np
import torch
import pdb

def add_noise(speech, noise, snr):
    alpha = np.sqrt(np.sum(speech ** 2) / (np.sum(noise ** 2) * 10 ** (snr / 10))    )
    mix = speech + alpha * noise
    return mix

def wav_to_spectrogram(wav, config):
    win = torch.hann_window(config['frame_length'])
    spec = torch.stft(wav, n_fft=config['n_fft'], hop_length=config['hop_length'], win_length = config['frame_length'],
                window=win, center=True, pad_mode='reflect')

    if spec.ndim != 4:
        spec = torch.unsqueeze(spec, 0)
    return spec.permute([0, 3, 2, 1])  #size: (batch_size, 2, T, F)


def process_speech_pair(wav_clean, wav_noisy, config, return_spec = False):
    spec_clean = wav_to_spectrogram(wav_clean, config['data'])
    spec_noisy = wav_to_spectrogram(wav_noisy, config['data'])   #torch tensors
    length = spec_clean.size(2)
    batch_size = spec_clean.size(0)


    if config["training"]["target"] == "cIRM":
        spec_clean = spec_clean.data.numpy()
        spec_noisy = spec_noisy.data.numpy()
        mask = (spec_clean[:,0,:,:] + 1j*spec_clean[:,1,:,:]) / (spec_noisy[:,0,:,:]+1e-8 + 1j*spec_noisy[:,1,:,:])
        mask_real = np.real(mask).reshape(batch_size, 1, length, -1)
        mask_imag = np.imag(mask).reshape(batch_size, 1, length, -1)
        mask = np.concatenate((mask_real, mask_imag), axis = 1)
        mask = torch.FloatTensor(mask)
        spec_clean = torch.FloatTensor(spec_clean)
        spec_noisy = torch.FloatTensor(spec_noisy)
    elif config["training"]["target"] == "IRM":
        mag_clean = torch.sqrt(spec_clean[:,0,:,:]**2 + spec_clean[:,1,:,:]**2)
        mag_reverb = torch.sqrt(spec_noisy[:,0,:,:]**2 + spec_noisy[:,1,:,:]**2)
        mask =torch.reshape(mag_clean / (mag_reverb+1e-8), (batch_size, 1, length, -1))
        #mask = util.target_compression(mask, self.config)
    elif config["training"]["target"] == "IRM_log":
        mag_clean = torch.sqrt(spec_clean[:,0,:,:]**2 + spec_clean[:,1,:,:]**2)
        mag_reverb = torch.sqrt(spec_noisy[:,0,:,:]**2 + spec_noisy[:,1,:,:]**2)
        mask = torch.reshape(mag_clean / (mag_reverb+1e-8), (batch_size, 1, length, -1))
        mask = torch.log10(mask + 1e-8)
        #mask = util.target_compression(mask, self.config)
    target = mask

    if config["training"]["input"] == "spec":
        x = spec_noisy
    elif config["training"]["input"] == "mag":
        x = torch.sqrt(spec_noisy[:,0,:,:]**2 + spec_noisy[:,1,:,:]**2)
        x = torch.reshape(x, (batch_size, 1, length, -1))
    elif config["training"]["input"] == "mag_log":
        x = torch.sqrt(spec_noisy[:,0,:,:]**2 + spec_noisy[:,1,:,:]**2)
        x = torch.reshape(torch.log10(x + 1e-8), (batch_size, 1, length, -1))

    input = x

    if return_spec:
        return input, target, spec_noisy, spec_clean
    else:
        return input, target


def audio_reconstruct(spectrogram, config, iter = 0):
    #shape: (f, t)
    #spectrogram = 10 ** (spectrogram)
    #spec = magnitude * phase
    #wav_ref = librosa.core.istft(spec, hop_length = config["data"]["hop_length"],
    #                        win_length = config["data"]["frame_length"],
    #                        window = "hann", center = True)
    if iter == 0:
        wav = librosa.core.istft(spectrogram, hop_length = config["data"]["hop_length"],
                                win_length = config["data"]["frame_length"],
                                window = "hann", center = True)
    else:
        mag, phase_new = librosa.magphase(spectrogram)
        for ii in range(0, iter):
            spec = mag * phase_new
            wav = librosa.core.istft(spec, hop_length = config["data"]["hop_length"],
                                    win_length = config["data"]["frame_length"],
                                    window = "hann", center = True)

            spec_new = librosa.core.stft(wav, n_fft = config["data"]["n_fft"],
                            hop_length = config["data"]["hop_length"], window = "hann",
                            center = True, dtype=np.complex64, pad_mode='reflect')
            mag_new, phase_new = librosa.magphase(spec_new)

    return wav


def target_compression(input, config, device = 'cuda'):
    Q = config["normalization"]["output"]["Q"]
    C = config["normalization"]["output"]["C"]
    if not isinstance(input, torch.Tensor):
        input = torch.FloatTensor(input)
    input = torch.where(input >= 10, torch.ones(input.size())*10, input)
    input = torch.where(input <= -10, torch.ones(input.size())*(-10), input)
    output = Q * (1 - torch.exp(-C * input)) / (1 + torch.exp(-C * input))
    return output

def target_decompression(input, config, device = 'cuda'):
    Q = config["normalization"]["output"]["Q"]
    C = config["normalization"]["output"]["C"]
    if not isinstance(input, torch.Tensor):
        input = torch.FloatTensor(input)
    input = torch.where(input >= 1, torch.ones(input.size())*(1 - 1e-5), input)
    input = torch.where(input <= -1, torch.ones(input.size())*(-1 + 1e-5), input)
    output = -1 / C * torch.log((Q - input) / (Q + input))
    return output

def normalization_0m_1v(input, axis = None, config = None, device = 'cpu'):
    # normalize input along dimensions except the given one
    # axis is the dimenson wanted to keep
    # e.g. input: [3 x 4 x 5] matrix, axis = 1, mean: [4,] vector

    epsilon = 1e-8
    dims = list(range(0, input.ndim))
    shape_desired = [1] * input.ndim
    for item in axis:
        dims.remove(item)
        shape_desired[item] = input.shape[item]
    if device == 'cpu':
        if config is None:
            input = input - np.reshape(np.mean(input, axis = tuple(dims)), shape_desired)
            input = input / np.reshape(np.std(input, axis = tuple(dims))+epsilon, shape_desired)
        else:
            input = input - np.reshape(config["mean"], shape_desired)
            input = input / np.reshape(config["std"]+epsilon, shape_desired)
    if device == 'cuda':
        input = input.cuda()
        if config is None:
            input = input - torch.mean(input, dim = tuple(dims), keepdim = True)
            input = input / (torch.std(input, dim = tuple(dims), keepdim = True, unbiased = True) + epsilon)
        else:
            mean = config["mean"].reshape(shape_desired)
            std = config["std"].reshape(shape_desired) + epsilon
            input = input - torch.FloatTensor(mean).cuda()
            input = input / torch.FloatTensor(std).cuda()
        input = input.cpu()
    return input

def denormalization_0m_1v(input, axis = None, config = None):
    epsilon = 1e-8
    dims = list(range(0, input.ndim))
    del dims[axis]
    shape_desired = [1] * input.ndim
    shape_desired[axis] = input.shape[axis]

    input = input * np.reshape(config["std"]+epsilon, shape_desired)
    input = input + np.reshape(config["mean"], shape_desired)

    return input

def ARMA_filter(input, order):
    rst = np.empty(input.shape)
    temp = input[order: -order, :]
    n_frame = temp.shape[0]
    temp = np.expand_dims(temp, 0)
    for ii in range(1, order+1):
        temp = np.concatenate((temp, np.expand_dims(input[order+ii: order+ii+n_frame, :], 0)), axis = 0)
        temp = np.concatenate((np.expand_dims(input[order-ii: order-ii+n_frame, :], 0), temp), axis = 0)

    for ii in range(0, order):
        rst[ii, :] = np.mean(input[0:ii+order, :], axis = 0)
        rst[-ii-1, :] = np.mean(input[-ii-order-1: ,], axis = 0)

    rst[order: -order, :] = np.mean(temp, axis = 0)
    return rst
