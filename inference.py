"""
usage: inference.py [options]

options:
--config = <path>         path of the config file
--output-dir = <path>     path to save the output file
--checkpoint = <path>     path of the checkpoint from which the model is restored
--device = <device>       "cpu" or "gpu", choose to run the model on cpu or GPU
--rir = <dir>             path of RIR file
--stride-mode = <int>     0, 1 or 2, 0: stride is 2; 1: stride is 1 & 2, changes in each layer; 2: stride is 1
--help                    show this help message
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
import scipy.io

from docopt import docopt
import logging
import json
import os
import sys
import pdb

import Model
import dataset
import preprocess
import util

lib_path = os.path.abspath(os.path.join(__file__, '..', 'feature_extraction'))
sys.path.append(lib_path)
import extract_feature

def get_rir_from_mat(rir_file):
    mat = scipy.io.loadmat(rir_file)
    rir_list = mat['rir_list']
    return rir_list

def load_checkpoint(path, model):

    print("Load checkpoint from: {}".format(path))
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint["state_dict"])
    #if optimizer_state is not None:
    #    print("Load optimizer state from {}".format(path))

    #logging.info("checkpoint is loaded from: " + checkpoint_load_path)
    print("finish loading checkpoint!")
    return model

def load_config(config_filepath):
    config_file = open(config_filepath, 'r')
    with config_file:
        return json.load(config_file)

def get_audio_filelist(index_file, audio_path):
    with open(index_file, 'r') as f:
        filelist = f.readlines()
    filelist = [os.path.join(audio_path, filename.replace('\n', '')) for filename in filelist]
    return filelist

def setHandler(filename = "/vol/vssp/msos/jz/complex_nn/checkpoint/train.log"):
    handler = logging.FileHandler(filename = filename, mode = 'a')
    format = '%(asctime)s-[%(levelname)s]: %(message)s'
    datefmt='%m/%d/%Y %I:%M:%S %p'
    formatter = logging.Formatter(fmt = format, datefmt = datefmt)
    handler.setFormatter(formatter)
    return handler

def write_wav(out_path, audio_name, postfix, extension, y_hat=None, y_target=None, y_reverb=None, config=None):
    os.makedirs(out_path, exist_ok = True)
    if y_hat is not None:
        filename_0 = audio_name + postfix + "_predict" + extension
        path = os.path.join(out_path, filename_0)
        librosa.output.write_wav(path, y_hat, sr = config["data"]["sample_rate"])
    if y_target is not None:
        filename_1 = audio_name + extension
        path = os.path.join(out_path, filename_1)
        librosa.output.write_wav(path, y_target, sr = config["data"]["sample_rate"])
    if y_reverb is not None:
        filename_2 = audio_name + postfix + extension
        path = os.path.join(out_path, filename_2)
        librosa.output.write_wav(path, y_reverb, sr = config["data"]["sample_rate"])
    print(filename_0 + " successfully saved!")
    return filename_0 + '\t' + filename_1 + '\t' + filename_2

def save_figure(out_path, audio_name, postfix, mag_hat, mag_target, mag_reverb):
    filename = os.path.join(out_path, audio_name+postfix+".png")
    fig = plt.figure(figsize = (18, 6))
    ax1, ax2, ax3 = fig.subplots(1,3)
    ax1 = librosa.display.specshow(20*np.log10(mag_reverb+1e-8), ax = ax1)
    plt.colorbar(ax1.get_children()[0], format='%+2.0f dB', ax = ax1)
    ax1.set_title("magnitude_reverb")
    ax2 = librosa.display.specshow(20*np.log10(mag_target+1e-8), ax = ax2)
    plt.colorbar(ax2.get_children()[0], format='%+2.0f dB', ax = ax2)
    ax2.set_title("magnitude_target")
    ax3 = librosa.display.specshow(20*np.log10(mag_hat+1e-8), ax = ax3)
    plt.colorbar(ax3.get_children()[0], format='%+2.0f dB', ax = ax3)
    ax3.set_title("magnitude_predicted")

    fig.savefig(os.path.join(out_path, filename))
    plt.close(fig)
    #save_spectrogram_plot(path, y_hat, y_target, config["dataset"]["sample_rate"])

def get_rir(rir_file):
    #input is rir file path
    with open(rir_file, 'r') as f:
        content = f.readlines()
        del content[0]
    #content contains n strings, each of which is an rir sequence
    #the first line of csv file is the indices of columns
    rir_list = []
    for input in content:
        input = input.split(',')
        input = list(filter(None, input))
        try:
            input.remove('\n')
        except ValueError:
            pass
        rir = np.array(input).astype(float)
        rir_list.append(rir)
    #find the actual start of rir, points before head represents the time when sound haven't reached the mic
    return rir_list


def do_eval_no_delay(model, device, audio_file, rir, postfix, output_path, config):

    sample_rate = config["data"]["sample_rate"]
    batch_size = 1

    audio_file_name = os.path.basename(audio_file)
    audio_name, extension = os.path.splitext(audio_file_name)

    feature_config_path = os.path.abspath(os.path.join(os.path.abspath(__file__), "../feature_extraction/config.json"))
    config_feature = load_config(feature_config_path)

    wav_clean = librosa.core.load(audio_file, sr = sample_rate)[0]
    rir = librosa.effects.trim(rir, top_db = 20, frame_length = config['data']['frame_length'], hop_length = config['data']['hop_length'])[0]
    length_wav = len(wav_clean)
    wav_reverb = np.convolve(wav_clean, rir, mode = 'full')[0: length_wav]
    wav_clean = torch.FloatTensor(wav_clean)
    wav_reverb = torch.FloatTensor(wav_reverb)

    spec_clean = util.wav_to_spectrogram(wav_clean, config['data'])
    spec_reverb = util.wav_to_spectrogram(wav_reverb, config['data'])   #torch tensors
    length = spec_clean.size(2)
    #shape: (n_frame, n_fftbins)

    if config["training"]["input"] == "spec":
        x = spec_reverb
    elif config["training"]["input"] == "mag":
        x = torch.sqrt(spec_reverb[:,0,:,:]**2 + spec_reverb[:,1,:,:]**2)
        x = torch.reshape(x, (batch_size, 1, length, -1))
    elif config["training"]["input"] == "mag_log":
        x = torch.sqrt(spec_reverb[:,0,:,:]**2 + spec_reverb[:,1,:,:]**2)
        x = torch.reshape(torch.log10(x + 1e-8), (batch_size, 1, length, -1))

    input = torch.FloatTensor(x).to(device)
    input = util.normalization_0m_1v(input, axis = [1,3], device = 'cuda')

    #if config["normalization"]["input"] == "feature-wise":
    #    features = util.normalization_0m_1v(features, axis = 1)

    model.eval()

    if input.size(2) % 2 == 0:
        input = input[:,:,0:-1,:]
        spec_clean = spec_clean[:,:, 0:-1, :]
        spec_reverb = spec_reverb[:,:, 0:-1, :]

    if config["training"]["target"] == "cIRM":
        mask_temp = torch.nn.parallel.data_parallel(model, input)  #shape: (n_frame * n_fftbins)

        mask_real = util.target_decompression(mask_temp.data[:,0,:,:].cpu(), config, device = 'cuda')
        mask_imag = util.target_decompression(mask_temp.data[:,1,:,:].cpu(), config, device = 'cuda')
        mask_hat = mask_real.squeeze().cpu().numpy() + 1j * mask_imag.squeeze().cpu().numpy()
    elif config["training"]["target"] == "IRM" or config["training"]["target"] == "PSM":
        mask_hat = torch.nn.parallel.data_parallel(model, input)
        mask_hat = util.target_decompression(mask_hat.data.cpu(), config, device = 'cuda')
        mask_hat = mask_hat.squeeze().cpu().numpy()
    elif config["training"]["target"] == "spec":
        spec_temp = torch.nn.parallel.data_parallel(model, input)
        spec_hat = spec_temp[:,0,:,:].data.squeeze().cpu().numpy() + 1j * spec_temp[:,1,:,:].data.squeeze().cpu().numpy()
        #indices = np.nonzero(mask_hat < 0)
        #mask_hat[indices] = 0


    #spec_target = spec_target[:, half_frame_window: -half_frame_window] / config["statistic"]["clean"]["energy"]
    #spec_hat = spec_reverb[half_frame_window:-half_frame_window, :].T / config["statistic"]["reverb"]["energy"] * mask_hat
    #spec_target = spec_target[:, half_frame_window: -half_frame_window]

    spec_clean = spec_clean.data[:,0,:,:].numpy() + 1j * spec_clean.data[:,1,:,:].numpy()
    spec_reverb = spec_reverb.data[:,0,:,:].numpy() + 1j * spec_reverb.data[:,1,:,:].numpy()
    spec_clean = spec_clean.squeeze()
    spec_reverb = spec_reverb.squeeze()

    if config["training"]["target"] == "cIRM":
        spec_hat = spec_reverb * mask_hat
    elif config["training"]["target"] == "IRM" or config["training"]["target"] == "PSM":
        spec_hat = spec_reverb * mask_hat

    spec_reverb = spec_reverb.T
    spec_clean = spec_clean.T
    spec_hat = spec_hat.T

    if config["training"]["target"] == "IRM":
        wav_predicted = util.audio_reconstruct(spec_hat, config, iter = 10)
    else:
        wav_predicted = util.audio_reconstruct(spec_hat, config)
    wav_target = util.audio_reconstruct(spec_clean, config)
    wav_reverb = util.audio_reconstruct(spec_reverb, config)

    mag_target, phase_target = librosa.core.magphase(spec_clean)
    mag_hat, phase_hat = librosa.core.magphase(spec_hat)
    angle_target = np.angle(phase_target)
    angle_hat = np.angle(phase_hat)

    mag_dif = np.mean(np.abs(mag_target - mag_hat))
    angle_dif = np.mean(np.abs(angle_target - angle_hat))

    index_result = write_wav(output_path, audio_name, postfix, extension, wav_predicted, wav_target, wav_reverb, config)
    save_figure(output_path, audio_name, postfix, np.abs(spec_hat), np.abs(spec_clean), np.abs(spec_reverb))

    return index_result, [mag_dif, angle_dif]


def inference_step_T60(device, output_path, model, rir_file = None, config = None, logger = None):
    model.eval()
    audio_path = config["data"]["audio_path"]
    T60_list = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 1.1, 1.2, 1.3, 1.4, 1.5]
    print("start inference")
    logger.info("start inference")

    if rir_file is None:
        rir_file = config["inference"]["rir"]

    if config["inference"]["rir"][-4:] == '.csv':
        rir_list = get_rir(rir_file)
    if config["inference"]["rir"][-4:] == '.mat':
        rir_list = get_rir_from_mat(rir_file)
    audio_filelist = get_audio_filelist(config["inference"]["audio_index"], audio_path)

    if logger is not None:
        logger.info("loading rir from: " + rir_file)
        logging.info("loading audio files from: " + config["inference"]["audio_index"])

    n_rir = len(rir_list)
    n_audio = 10
    rir_per_T60 = 20

    postfix_list = []
    for ii in T60_list:
        for jj in range(0, rir_per_T60):
            postfix_list.append("_{}_{}".format(ii, jj))

    assert len(rir_list) == len(postfix_list)
    index_list = []
    mag_angle_dif = []
    for ii in range(0, n_rir):
        for audio_file in audio_filelist[0: n_audio]:
            rir = rir_list[ii]
            postfix = postfix_list[ii]
            #index_0 = do_eval_corr_0(model, device, audio_file, rir, postfix, output_path+"_corr_0", config)
            #index_1 = do_eval_corr_1(model, device, audio_file, rir, postfix, output_path+"_corr_1", config)
            index, mag_angle_dif_item = do_eval_no_delay(model, device, audio_file, rir, postfix, output_path, config)
            mag_angle_dif.append(mag_angle_dif_item)
            index_list.append(index)

    with open(os.path.join(output_path, "index.txt"), 'w') as f:
        for item in index_list:
            f.write(item + '\n')

    with open(os.path.join(output_path, "raw_mag_angle_dif.txt"), 'w') as f:
        for item in mag_angle_dif:
            f.write("mag_dif:{}, angle_dif:{}\n".format(item[0], item[1]))

    print("INFERENCE FINISHES!")
    logger.info("INFERENCE FINISHES!")



def inference_3T60(device, output_path, model = None, rir_file = None, config = None, logger = None):

    model.eval()
    audio_path = config["data"]["audio_path"]
    if rir_file is None:
        rir_file = config["inference"]["rir"]

    print("start inference")
    logger.info("start inference")

    if config["inference"]["rir"][-4:] == '.csv':
        rir_list = get_rir(rir_file)
    if config["inference"]["rir"][-4:] == '.mat':
        rir_list = get_rir_from_mat(rir_file)

    logger.info("loading rir from: " + config["inference"]["rir"])

    audio_filelist = get_audio_filelist(config["inference"]["audio_index"], audio_path)
    logging.info("loading audio files from: " + config["inference"]["audio_index"])

    n_audio = len(audio_filelist)
    n_rir = len(rir_list)

    if len(rir_list) == 6:
        postfix_list = ["_0.3_0", "_0.3_1", "_0.6_0", "_0.6_1", "_0.9_0", "_0.9_1"]
    if len(rir_list) == 3:
        postfix_list = ["_0.3_0", "_0.6_0", "_0.9_0"]

    if len(rir_list) == 2:
        postfix_list = ["_0", "_1"]

    index_list = []
    mag_angle_dif = []
    for ii in range(0, n_rir):
        for audio_file in audio_filelist:
            rir = rir_list[ii]
            postfix = postfix_list[ii]
            #index_0 = do_eval_corr_0(model, device, audio_file, rir, postfix, output_path+"_corr_0", config)
            #index_1 = do_eval_corr_1(model, device, audio_file, rir, postfix, output_path+"_corr_1", config)
            index, mag_angle_dif_item = do_eval_no_delay(model, device, audio_file, rir, postfix, output_path, config)
            mag_angle_dif.append(mag_angle_dif_item)
            index_list.append(index)

    #with open(os.path.join(output_path+'_corr_0', "index.txt"), 'w') as f:
    #    for item in index_list:
    #        f.write(item + '\n')

    #with open(os.path.join(output_path+'_corr_1', "index.txt"), 'w') as f:
    #    for item in index_list:
    #        f.write(item + '\n')

    with open(os.path.join(output_path, "index.txt"), 'w') as f:
        for item in index_list:
            f.write(item + '\n')

    with open(os.path.join(output_path, "raw_mag_angle_dif.txt"), 'w') as f:
        for item in mag_angle_dif:
            f.write('mag_dif:{}, angle_dif:{}\n'.format(item[0], item[1]))

    print("INFERENCE FINISHES!")
    logger.info("INFERENCE FINISHES!")


if __name__ == "__main__":
    args = docopt(__doc__)
    config_path = args["--config"]
    output_path = args["--output-dir"]
    checkpoint_load_path = args["--checkpoint"]
    device = args["--device"]
    rir_file = args["--rir"]
    stride_mode = int(args["--stride-mode"])

    if config_path is None:
        config_path = "/user/HS228/jz00677/PYTHON_project/Unet/recurrent/src/config_5R.json"
    config = load_config(config_path)

    if output_path is None:
        if config["debug"]:
            output_path = "/vol/vssp/msos/jz/debug"
        else:
            output_path = config["inference"]["output_path"]
    config["training"]["input"] = "spec"
    config["training"]["target"] = "IRM"
    config["model"]["recurrent_type"] = "LSTM"
    config["model"]["bidirectional"] = True
    config["model"]["stride_mode"] = stride_mode

    if device is None:
        device = 'cuda'
    else:
        device = 'cuda' if device == 'gpu' else 'cpu'

    os.makedirs(output_path, exist_ok = True)

    logger = logging.getLogger("logger")
    handler = setHandler(filename = os.path.join(output_path, "inference.log"))
    logger.addHandler(handler)

    model = Model.UNet_recurrent(config)
    #print(model)
    #pdb.set_trace()

    if checkpoint_load_path is None:
        checkpoint_load_path = config["inference"]["checkpoint"]

    load_checkpoint(checkpoint_load_path, model)

    if rir_file is None:
        rir_file = "/user/HS228/jz00677/PYTHON_project/RIR_Generator/rir_5R_stepT60_test.csv"
    model = model.to(device)
    inference_step_T60(device, output_path = output_path, model = model, rir_file = rir_file, config = config, logger = logger)

    logger.removeHandler(handler)

    sys.exit(0)
