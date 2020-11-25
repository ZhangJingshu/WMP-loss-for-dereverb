
"""
usage: train.py [options]

options:
--config=<path>         path of the config file
--checkpoint-dir=<dir>  directory where to save checkpoint
--checkpoint=<path>     path of the checkpoint from which the model is restored
--weight-loss=<list>    list of weights in loss function
--loss=<string>         names of loss function
--stride-mode=<int>     0, 1 or 2. 0: stride is 2; 1: stride is 1 & 2, changes in each layer; 2: stride is 1
--target=<string>       cIRM, IRM, PSM, spec, the training target (output) of the model
--help                  show this help message
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt

from docopt import docopt
import logging
import json
import os
import sys
import pdb

import Model
import dataset_5R
import preprocess
import util
import inference
import evaluation

class WPMLoss(nn.Module):    #weighted phase magnitude loss
    def __init__(self, weight):
        super(WPMLoss, self).__init__()
        self.weight = weight

    def forward(self, y_real, y_imag, y_real_hat, y_imag_hat):
        pi = torch.FloatTensor([np.pi]).cuda()
        mag = torch.sqrt(y_real**2 + y_imag**2)
        mag_hat = torch.sqrt(y_real_hat**2 + y_imag_hat**2)

        theta = torch.atan2(y_imag, y_real)
        theta_hat = torch.atan2(y_imag_hat, y_real_hat)
        dif_theta = 2 * mag * torch.sin((theta_hat - theta)/2)  #0 <= dif_thera <= 2*mag

        #cos_theta = y_real / (torch.sqrt(y_real**2 + y_imag**2)+1e-8)
        #sin_theta = y_imag / (torch.sqrt(y_real**2 + y_imag**2)+1e-8)
        #cos_theta_hat = y_real_hat / (torch.sqrt(y_real_hat**2 + y_imag_hat**2)+1e-8)
        #sin_theta_hat = y_imag_hat / (torch.sqrt(y_real_hat**2 + y_imag_hat**2)+1e-8)
        #cos_dif_theta = cos_theta * cos_theta_hat + sin_theta * sin_theta_hat
        #sin_half_dif_theta_squared = (1 - cos_dif_theta) / 2
        dif_mag = mag_hat - mag
        loss = torch.mean(dif_mag**2 + self.weight * dif_theta**2)
        if torch.isnan(loss).any():
            np.save('y_real.npy', y_real.data.cpu().numpy())
            np.save('y_imag.npy', y_imag.data.cpu().numpy())
            np.save('y_real_hat.npy', y_real_hat.data.cpu().numpy())
            np.save('y_imag_hat.npy', y_imag_hat.data.cpu().numpy())
            raise ValueError("NAN encountered in loss")
        return loss


def plot_loss(loss_train, loss_test, loss_mag_train, loss_mag_test, loss_phase_train, loss_phase_test, loss_angle_train, loss_angle_test, path):
    fig = plt.figure(figsize = (12, 6))
    ax1, ax2 = fig.subplots(1,2)
    ax1.plot(loss_train)
    ax1.set_title("loss_train")
    ax1.set_xlabel("epoches")
    ax1.set_ylabel("loss")
    ax2.plot(loss_test)
    ax2.set_title("loss_test")
    ax2.set_xlabel("epoches")
    ax2.set_ylabel("loss")
    fig.savefig(os.path.join(path, "loss.png"))
    plt.close(fig)

    fig = plt.figure(figsize = (12, 6))
    ax1, ax2 = fig.subplots(1,2)
    ax1.plot(loss_mag_train)
    ax1.set_title("loss_train_magnitude")
    ax1.set_xlabel("epoches")
    ax1.set_ylabel("loss")
    ax2.plot(loss_mag_test)
    ax2.set_title("loss_test_magnitude")
    ax2.set_xlabel("epoches")
    ax2.set_ylabel("loss")
    fig.savefig(os.path.join(path, "loss_magnitude.png"))
    plt.close(fig)

    fig = plt.figure(figsize = (12, 6))
    ax1, ax2 = fig.subplots(1,2)
    ax1.plot(loss_phase_train)
    ax1.set_title("loss_train_phase")
    ax1.set_xlabel("epoches")
    ax1.set_ylabel("loss")
    ax2.plot(loss_phase_test)
    ax2.set_title("loss_test_phase")
    ax2.set_xlabel("epoches")
    ax2.set_ylabel("loss")
    fig.savefig(os.path.join(path, "loss_phase.png"))
    plt.close(fig)

    fig = plt.figure(figsize = (12, 6))
    ax1, ax2 = fig.subplots(1,2)
    ax1.plot(loss_angle_train / np.pi)
    ax1.set_title("loss_train_angle")
    ax1.set_xlabel("epoches")
    ax1.set_ylabel("loss")
    ax2.plot(loss_angle_test / np.pi)
    ax2.set_title("loss_test_angle")
    ax2.set_xlabel("epoches")
    ax2.set_ylabel("loss / pi")
    fig.savefig(os.path.join(path, "loss_angle.png"))
    plt.close(fig)


def save_checkpoint(checkpoint_save_path, model, optimizer = None, filename = None):
    global loss_train, loss_test
    global loss_mag_train, loss_mag_test, loss_phase_train, loss_phase_test, loss_angle_train, loss_angle_test
    global global_step, global_epoch
    if filename is None:
        checkpoint_path = os.path.join(checkpoint_save_path, "checkpoint_step{:09d}.pth".format(global_step))
    else:
        checkpoint_path = os.path.join(checkpoint_save_path, filename)
    optimizer_state = optimizer.state_dict()
    torch.save({
        "state_dict": model.state_dict(),
        "optimizer": optimizer_state,
        "global_step": global_step,
        "global_epoch": global_epoch,
        "loss_train": loss_train,
        "loss_test": loss_test,
        "loss_mag_train": loss_mag_train,
        "loss_mag_test": loss_mag_test,
        "loss_phase_train": loss_phase_train,
        "loss_phase_test": loss_phase_test,
        "loss_angle_train": loss_angle_train,
        "loss_angle_test": loss_angle_test
    }, checkpoint_path)

    print("Saved checkpoint:", checkpoint_path)


def load_checkpoint(path, model, optimizer):
    global loss_train, loss_test
    global loss_mag_train, loss_mag_test, loss_phase_train, loss_phase_test, loss_angle_train, loss_angle_test
    global global_step, global_epoch
    logger.info("Load checkpoint from: {}".format(path))
    checkpoint = torch.load(path)
    try:
        model.load_state_dict(checkpoint["state_dict"])
    except RuntimeError:
        logger.info("Failed to load checkpoint from: " + path)
        return False
    #if optimizer_state is not None:
    #    print("Load optimizer state from {}".format(path))
    optimizer.load_state_dict(checkpoint["optimizer"])

    global_step = checkpoint["global_step"]
    global_epoch = checkpoint["global_epoch"]
    loss_train = checkpoint["loss_train"]
    loss_test = checkpoint["loss_test"]
    loss_mag_train = checkpoint["loss_mag_train"]
    loss_mag_test = checkpoint["loss_mag_test"]
    loss_phase_train = checkpoint["loss_phase_train"]
    loss_phase_test = checkpoint["loss_phase_test"]
    loss_angle_train = checkpoint["loss_angle_train"]
    loss_angle_test = checkpoint["loss_angle_test"]

    logger.info("checkpoint loaded successfully from: " + path)

    return True


def save_result(out_path, y_hat=None, y_target=None, spec_hat=None, spec_target=None, mask_hat = None, mask = None, phase=None, config=None):
    global global_step, global_epoch

    os.makedirs(out_path, exist_ok = True)
    if y_hat is not None:
        path = os.path.join(out_path, "epoch{:04d}_step{:07d}_{}_predicted.wav".format(
                                global_epoch, global_step, phase))
        librosa.output.write_wav(path, y_hat, sr = config["data"]["sample_rate"])
    if y_target is not None:
        path = os.path.join(out_path, "epoch{:04d}_step{:07d}_{}_target.wav".format(
                            global_epoch, global_step, phase))
        librosa.output.write_wav(path, y_target, sr = config["data"]["sample_rate"])

    if spec_hat is not None and spec_target is not None:

        fig = plt.figure(figsize = (12, 18))
        axes = fig.subplots(3,2)
        axes[0,0] = librosa.display.specshow(np.log10(np.abs(spec_target)+1e-8), ax = axes[0,0])
        plt.colorbar(axes[0,0].get_children()[0], ax = axes[0,0])
        axes[0,0].set_title("magnitude_target")
        axes[0,1] = librosa.display.specshow(np.log10(np.abs(spec_hat)+1e-8), ax = axes[0,1])
        plt.colorbar(axes[0,1].get_children()[0], ax = axes[0,1])
        axes[0,1].set_title("magnitude_predicted")

        axes[1,0] = librosa.display.specshow(np.real(spec_target), ax = axes[1,0])
        plt.colorbar(axes[1,0].get_children()[0], ax = axes[1,0])
        axes[1,0].set_title("spec_real_target")
        axes[1,1] = librosa.display.specshow(np.real(spec_hat), ax = axes[1,1])
        plt.colorbar(axes[1,1].get_children()[0], ax = axes[1,1])
        axes[1,1].set_title("spec_real_predicted")
        axes[2,0] = librosa.display.specshow(np.imag(spec_target), ax = axes[2,0])
        plt.colorbar(axes[2,0].get_children()[0], ax = axes[2,0])
        axes[2,0].set_title("spec_imag_target")
        axes[2,1] = librosa.display.specshow(np.imag(spec_hat), ax = axes[2,1])
        plt.colorbar(axes[2,1].get_children()[0], ax = axes[2,1])
        axes[2,1].set_title("spec_imag_predicted")

        fig.savefig(os.path.join(out_path, "epoch{:04d}_step{:07d}_{}_spectrogram.png".format(
                            global_epoch, global_step, phase)))
        plt.close(fig)

    if mask_hat is not None and mask is not None:
        if mask.dtype == np.complex64 or mask.dtype == np.complex:
            fig = plt.figure(figsize = (12, 18))
            axes = fig.subplots(3,2)
            temp_r = util.target_compression(np.real(mask), config, device = 'cpu')
            temp_i = util.target_compression(np.imag(mask), config, device = 'cpu')
            mask_r = util.target_decompression(temp_r, config, device = 'cpu')
            mask_i = util.target_decompression(temp_i, config, device = 'cpu')
            mask_compressed = mask_r.cpu().numpy() + 1j * mask_i.cpu().numpy()

            axes[0,0] = librosa.display.specshow(np.log10(np.abs(mask_compressed)+1e-8), ax = axes[0,0])
            plt.colorbar(axes[0,0].get_children()[0], ax = axes[0,0])
            axes[0,0].set_title("mask_target")
            axes[0,1] = librosa.display.specshow(np.log10(np.abs(mask_hat)+1e-8), ax = axes[0,1])
            plt.colorbar(axes[0,1].get_children()[0], ax = axes[0,1])
            axes[0,1].set_title("mask_predicted")
            axes[1,0] = librosa.display.specshow(np.real(mask_compressed), ax = axes[1,0])
            plt.colorbar(axes[1,0].get_children()[0], ax = axes[1,0])
            axes[1,0].set_title("mask_real_target")
            axes[1,1] = librosa.display.specshow(np.real(mask_hat), ax = axes[1,1])
            plt.colorbar(axes[1,1].get_children()[0], ax = axes[1,1])
            axes[1,1].set_title("mask_real_predicted")
            axes[2,0] = librosa.display.specshow(np.imag(mask_compressed), ax = axes[2,0])
            plt.colorbar(axes[2,0].get_children()[0], ax = axes[2,0])
            axes[2,0].set_title("mask_imag_target")
            axes[2,1] = librosa.display.specshow(np.imag(mask_hat), ax = axes[2,1])
            plt.colorbar(axes[2,1].get_children()[0], ax = axes[2,1])
            axes[2,1].set_title("mask_imag_predicted")
        else:
            fig = plt.figure(figsize = (12, 12))
            axes = fig.subplots(2,2)
            mask = util.target_compression(mask, config, device = 'cpu')
            mask = util.target_decompression(mask, config, device = 'cpu')
            mask = mask.data.cpu().numpy()
            axes[0,0] = librosa.display.specshow(np.log10(np.abs(mask)+1e-8), ax = axes[0,0])
            plt.colorbar(axes[0,0].get_children()[0], ax = axes[0,0])
            axes[0,0].set_title("mask_target_log")
            axes[0,1] = librosa.display.specshow(np.log10(np.abs(mask_hat)+1e-8), ax = axes[0,1])
            plt.colorbar(axes[0,1].get_children()[0], ax = axes[0,1])
            axes[0,1].set_title("mask_predicted_log")
            axes[1,0] = librosa.display.specshow(mask, ax = axes[1,0])
            plt.colorbar(axes[1,0].get_children()[0], ax = axes[1,0])
            axes[1,0].set_title("mask_target_linear")
            axes[1,1] = librosa.display.specshow(mask_hat, ax = axes[1,1])
            plt.colorbar(axes[1,1].get_children()[0], ax = axes[1,1])
            axes[1,1].set_title("mask_predicted_linear")

        fig.savefig(os.path.join(out_path, "epoch{:04d}_step{:07d}_{}_mask.png".format(
                            global_epoch, global_step, phase)))
        plt.close(fig)
    #save_spectrogram_plot(path, y_hat, y_target, config["dataset"]["sample_rate"])


def load_config(config_filepath):
    config_file = open(config_filepath, 'r')
    with config_file:
        return json.load(config_file)


def setHandler(filename = "/vol/vssp/msos/jz/complex_nn/checkpoint/train.log"):
    handler = logging.FileHandler(filename = filename, mode = 'a')
    format = '%(asctime)s-[%(levelname)s]: %(message)s'
    datefmt='%m/%d/%Y %I:%M:%S %p'
    formatter = logging.Formatter(fmt = format, datefmt = datefmt)
    handler.setFormatter(formatter)
    return handler


def do_eval(model, wav_clean, wav_reverb, config):
    model.eval()
    sample_rate = config["data"]["sample_rate"]
    batch_size = 1

    #audio_file = audio_filelist[np.random.randint(0, len(audio_filelist))]
    #indice = np.random.randint(0, len(sample_filelist))

    spec_clean = util.wav_to_spectrogram(wav_clean, config['data'])
    spec_reverb = util.wav_to_spectrogram(wav_reverb, config['data'])   #torch tensors
    length = spec_clean.size(2)


    if config["training"]["target"] == "cIRM":
        spec_clean = spec_clean.data.numpy()
        spec_reverb = spec_reverb.data.numpy()
        mask = (spec_clean[:,0,:,:] + 1j*spec_clean[:,1,:,:]) / (spec_reverb[:,0,:,:]+1e-8 + 1j*spec_reverb[:,1,:,:])
        mask_real = np.real(mask).reshape(batch_size, 1, length, -1)
        mask_imag = np.imag(mask).reshape(batch_size, 1, length, -1)
        mask = np.concatenate((mask_real, mask_imag), axis = 1)
        mask = torch.FloatTensor(mask)
        spec_clean = torch.FloatTensor(spec_clean)
        spec_reverb = torch.FloatTensor(spec_reverb)
    elif config["training"]["target"] == "IRM":
        mag_clean = torch.sqrt(spec_clean[:,0,:,:]**2 + spec_clean[:,1,:,:]**2)
        mag_reverb = torch.sqrt(spec_reverb[:,0,:,:]**2 + spec_reverb[:,1,:,:]**2)
        mask =torch.reshape(mag_clean / (mag_reverb+1e-8), (batch_size, 1, length, -1))
        #mask = util.target_compression(mask, self.config)
    elif config["training"]["target"] == "PSM":
        spec_clean = spec_clean.data.numpy()
        spec_reverb = spec_reverb.data.numpy()
        mask = (spec_clean[:,0,:,:] + 1j*spec_clean[:,1,:,:]) / (spec_reverb[:,0,:,:]+1e-8 + 1j*spec_reverb[:,1,:,:])
        mask = np.real(mask).reshape(batch_size, 1, length, -1)
        mask = torch.FloatTensor(mask)
        spec_clean = torch.FloatTensor(spec_clean)
        spec_reverb = torch.FloatTensor(spec_reverb)

        #mask = util.target_compression(mask, self.config)

    if config["training"]["input"] == "spec":
        x = spec_reverb
    elif config["training"]["input"] == "mag":
        x = torch.sqrt(spec_reverb[:,0,:,:]**2 + spec_reverb[:,1,:,:]**2)
        x = torch.reshape(x, (batch_size, 1, length, -1))
    elif config["training"]["input"] == "mag_log":
        x = torch.sqrt(spec_reverb[:,0,:,:]**2 + spec_reverb[:,1,:,:]**2)
        x = torch.reshape(torch.log10(x + 1e-8), (batch_size, 1, length, -1))

    input = torch.FloatTensor(x).cuda()
    input = util.normalization_0m_1v(input, axis = [1,3], device = 'cuda')

    #if config["normalization"]["input"] == "feature-wise":
    #    features = util.normalization_0m_1v(features, axis = 1)
    model.eval()

    if input.size(2) % 2 == 0:
        input = input[:,:,0:-1,:]
        spec_clean = spec_clean[:, :, 0:-1, :]
        spec_reverb = spec_reverb[:, :, 0:-1, :]

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

    #spec_target = spec_target[:, half_frame_window: -half_frame_window] / config["statistic"]["clean"]["energy"]
    #spec_hat = spec_reverb[half_frame_window:-half_frame_window, :].T / config["statistic"]["reverb"]["energy"] * mask_hat
    #spec_target = spec_target[:, half_frame_window: -half_frame_window]

    spec_clean = spec_clean.data[:,0,:,:].numpy() + 1j * spec_clean.data[:,1,:,:].numpy()
    spec_reverb = spec_reverb.data[:,0,:,:].numpy() + 1j * spec_reverb.data[:,1,:,:].numpy()
    spec_clean = spec_clean.squeeze()
    spec_reverb = spec_reverb.squeeze()

    if config["training"]["target"] == "cIRM":
        mask = mask.data[:,0,:,:].squeeze().numpy() + 1j * mask.data[:,1,:,:].squeeze().numpy()
        spec_hat = spec_reverb * mask_hat
    elif config["training"]["target"] == "IRM" or config["training"]["target"] == "PSM":
        mask = mask.data.squeeze().numpy()
        spec_hat = spec_reverb * mask_hat
    elif config["training"]["target"] == "spec":
        mask_hat = spec_hat
        mask = spec_clean

    spec_reverb = spec_reverb.T
    spec_clean = spec_clean.T
    spec_hat = spec_hat.T
    wav_reconstructed = util.audio_reconstruct(spec_hat, config)
    wav_target = util.audio_reconstruct(spec_clean, config)
    return wav_target, wav_reconstructed, spec_clean, spec_hat, mask, mask_hat


def train(model, optimizer, criterion, dataloader, index_file, config, checkpoint_save_path):
    global loss_train, loss_test
    global loss_mag_train, loss_mag_test, loss_phase_train, loss_phase_test, loss_angle_train, loss_angle_test
    global global_step, global_epoch
    n_epoch = config["training"]["n_epoch"]
    loss_best = 100
    loss_mag_best = 100

    if config["debug"]:
        checkpoint_interval = 100
        save_result_interval = 10
    else:
        checkpoint_interval = 10000
        save_result_interval = 500

    states_path = os.path.join(checkpoint_save_path, "training_state")
    current_lr = config["training"]["learning_rate"]
    while(global_epoch < n_epoch):
        if global_epoch % 10 == 0 and global_epoch != 0:
            current_lr = current_lr / 2
            for param_group in optimizer.param_groups:
                param_group['lr'] = current_lr
        #if global_epoch % 50 == 0 and global_epoch > 50:
        #    current_lr = current_lr / 5
        #    for param_group in optimizer.param_groups:
        #        param_group['lr'] = current_lr

        for phase, loader in dataloader.items():
            train = (phase == "train")
            running_loss = np.array([])
            running_loss_mag = np.array([])
            running_loss_phase = np.array([])
            running_loss_angle = np.array([])
            if train:
                save_test_state = False

            #print("epoch {} {} phase start".format(global_epoch, phase))
            for step, (wav_reverb, wav_clean) in enumerate(loader):
                if train:
                    model.train()
                else:
                    model.eval()
                #print("step {} start".format(step))
                if config["debug"]:
                    print("epoch {} step {} start".format(global_epoch, step))

                batch_size = wav_clean.size(0)

                spec_clean = util.wav_to_spectrogram(wav_clean, config['data'])
                spec_reverb = util.wav_to_spectrogram(wav_reverb, config['data'])   #torch tensors
                length = spec_clean.size(2)

                if config["training"]["target"] == "cIRM":
                    spec_clean = spec_clean.data.numpy()
                    spec_reverb = spec_reverb.data.numpy()
                    mask = (spec_clean[:,0,:,:] + 1j*spec_clean[:,1,:,:]) / (spec_reverb[:,0,:,:]+1e-8 + 1j*spec_reverb[:,1,:,:])
                    mask_real = np.real(mask).reshape(batch_size, 1, length, -1)
                    mask_imag = np.imag(mask).reshape(batch_size, 1, length, -1)
                    mask = np.concatenate((mask_real, mask_imag), axis = 1)
                    mask = torch.FloatTensor(mask)
                    spec_clean = torch.FloatTensor(spec_clean)
                    spec_reverb = torch.FloatTensor(spec_reverb)
                    y = util.target_compression(mask, config, device = 'cpu')
                elif config["training"]["target"] == "IRM":
                    mag_clean = torch.sqrt(spec_clean[:,0,:,:]**2 + spec_clean[:,1,:,:]**2)
                    mag_reverb = torch.sqrt(spec_reverb[:,0,:,:]**2 + spec_reverb[:,1,:,:]**2)
                    mask = torch.reshape(mag_clean / (mag_reverb+1e-8), (batch_size, 1, length, -1))
                    y = util.target_compression(mask, config, device = 'cpu')
                    #mask = util.target_compression(mask, self.config)
                elif config["training"]["target"] == "PSM":
                    spec_clean = spec_clean.data.numpy()
                    spec_reverb = spec_reverb.data.numpy()
                    mask = (spec_clean[:,0,:,:] + 1j*spec_clean[:,1,:,:]) / (spec_reverb[:,0,:,:]+1e-8 + 1j*spec_reverb[:,1,:,:])
                    mask = np.real(mask).reshape(batch_size, 1, length, -1)
                    mask = torch.FloatTensor(mask)
                    spec_clean = torch.FloatTensor(spec_clean)
                    spec_reverb = torch.FloatTensor(spec_reverb)
                    y = util.target_compression(mask, config, device = 'cpu')
                    #mask = util.target_compression(mask, self.config)
                elif config["training"]["target"] == "spec":
                    y = spec_clean


                if config["training"]["input"] == "spec":
                    x = spec_reverb
                elif config["training"]["input"] == "mag":
                    x = torch.sqrt(spec_reverb[:,0,:,:]**2 + spec_reverb[:,1,:,:]**2)
                    x = torch.reshape(x, (batch_size, 1, length, -1))
                elif config["training"]["input"] == "mag_log":
                    x = torch.sqrt(spec_reverb[:,0,:,:]**2 + spec_reverb[:,1,:,:]**2)
                    x = torch.reshape(torch.log10(x + 1e-8), (batch_size, 1, length, -1))

                if x.size(2) % 2 == 0:
                    x = x[:,:,0:-1,:]
                    y = y[:,:,0:-1,:]

                x = x.cuda()
                y = y.cuda()

                x = util.normalization_0m_1v(x, axis = [1,3], device = 'cuda')

                optimizer.zero_grad()
                y_hat = torch.nn.parallel.data_parallel(model, x)
                if config["training"]["loss_function"] == "MSE":
                    loss = criterion(y_hat, y)
                else:
                    y_real_hat = y_hat[:,0,:,:]
                    y_imag_hat = y_hat[:,1,:,:]
                    y_real = y[:,0,:,:]
                    y_imag = y[:,1,:,:]
                    loss = criterion(y_real, y_imag, y_real_hat, y_imag_hat)

                if y.size(1) == 2:
                    mag = torch.sqrt(y.data[:,0,:,:]**2 + y.data[:,1,:,:]**2)
                    mag_hat = torch.sqrt(y_hat.data[:,0,:,:]**2 + y_hat.data[:,1,:,:]**2)
                    dif_mag = torch.mean((mag - mag_hat)**2)
                    theta = torch.atan2(y.data[:,1,:,:], y.data[:,0,:,:])     #atan2(b, a) == atan(b/a)
                    theta_hat = torch.atan2(y_hat.data[:,1,:,:], y_hat.data[:,0,:,:])
                    dif_phase = torch.mean((mag * torch.sin((theta_hat - theta)/2))**2).data.cpu().numpy()

                    dif_angle = (theta_hat - theta).data.cpu().numpy()
                    dif_angle = dif_angle + 2*np.pi * (dif_angle < -np.pi)
                    dif_angle = dif_angle - 2*np.pi * (dif_angle > np.pi)
                    dif_angle = np.mean(np.abs(dif_angle))
                else:
                    dif_mag = torch.mean((y - y_hat)**2)
                    dif_angle = 0
                    dif_phase = 0

                running_loss = np.append(running_loss, loss.data.cpu().numpy())
                running_loss_mag = np.append(running_loss_mag, dif_mag.data.cpu().numpy())
                running_loss_phase = np.append(running_loss_phase, dif_phase)
                running_loss_angle = np.append(running_loss_angle, dif_angle)

                if train and global_step % checkpoint_interval == 0 and global_step > 0:
                    save_checkpoint(checkpoint_save_path, model, optimizer)
                if train and global_step % save_result_interval == 0 and global_step > 0:
                    #save_mask_figure(states_path, mask_real_hat, mask_real, mask_imag_hat, mask_imag, phase = "train")
                    index = np.random.randint(0, batch_size)
                    wav, wav_hat, spec, spec_hat, mask, mask_hat = do_eval(model, wav_clean[index, :], wav_reverb[index, :], config)
                    save_result(states_path, wav_hat, wav, spec_hat, spec, mask_hat, mask, phase = "train_eval", config = config)
                    save_test_state = True
                if not train and save_test_state == True:
                    #save_mask_figure(states_path, mask_real_hat, mask_real, mask_imag_hat, mask_imag, phase = "test")
                    index = np.random.randint(0, batch_size)
                    wav, wav_hat, spec, spec_hat, mask, mask_hat = do_eval(model, wav_clean[index, :], wav_reverb[index, :], config)
                    save_result(states_path, wav_hat, wav, spec_hat, spec, mask_hat, mask, phase = "eval", config = config)
                    save_test_state = False

                if train:
                    loss.backward()
                    optimizer.step()
                    global_step += 1
                #print("step {} finish".format(step))

                ############ end of for step, (x, y, noise, g) in enumerate(loader) ###################################
            if train:
                new_item = np.expand_dims(np.array([np.mean(running_loss), np.var(running_loss)]), 0)
                loss_train = new_item if loss_train.size == 0 else np.append(loss_train, new_item, axis = 0)
                loss_mag_train = np.append(loss_mag_train, np.mean(running_loss_mag))
                loss_phase_train = np.append(loss_phase_train, np.mean(running_loss_phase))
                loss_angle_train = np.append(loss_angle_train, np.mean(running_loss_angle))
            else:
                new_item = np.expand_dims(np.array([np.mean(running_loss), np.var(running_loss)]), 0)
                loss_test = new_item if loss_test.size == 0 else np.append(loss_test, new_item, axis = 0)
                loss_mag_test = np.append(loss_mag_test, np.mean(running_loss_mag))
                loss_phase_test = np.append(loss_phase_test, np.mean(running_loss_phase))
                loss_angle_test = np.append(loss_angle_test, np.mean(running_loss_angle))

                if loss_best > np.mean(running_loss):
                    loss_best = np.mean(running_loss)
                    save_checkpoint(checkpoint_save_path, model, optimizer, filename = "best_loss_model.pth")

                if loss_mag_best > np.mean(running_loss_mag):
                    loss_mag_best = np.mean(running_loss_mag)
                    save_checkpoint(checkpoint_save_path, model, optimizer, filename = "best_mag_model.pth")

            logger.info("epoch {}: loss of {} phase is {} (Var: {})".format(global_epoch, phase, new_item[0][0], new_item[0][1]))
            print("epoch {}: loss of {} phase is {} (Var: {})".format(global_epoch, phase, new_item[0][0], new_item[0][1]))
            ########## end of for phase, loader in dataloader.items(): #############################################
        global_epoch += 1
        save_checkpoint(checkpoint_save_path, model, optimizer, filename = "checkpoint_current.pth")
    #wav, wav_hat, spec, spec_hat, mask, mask_hat = do_eval(model, wav_clean[index, :], wav_reverb[index, :], config)
    #save_result(states_path, wav_hat, wav, spec_hat, spec, phase = "finish", config = config)


loss_train = np.array([])
loss_test = np.array([])
loss_mag_train = np.array([])
loss_mag_test = np.array([])
loss_phase_train = np.array([])
loss_phase_test = np.array([])
loss_angle_train = np.array([])
loss_angle_test = np.array([])
global_step = 0
global_epoch = 0

if __name__ == "__main__":
    args = docopt(__doc__)
    config_path = args["--config"]
    checkpoint_save_path = args["--checkpoint-dir"]
    checkpoint_load_path = args["--checkpoint"]
    weight_loss_list = args["--weight-loss"]
    loss = args["--loss"]
    stride_mode = args["--stride-mode"]
    target_in = args["--target"]
    print(args)

    if config_path is None:
        config_path = "/user/HS228/jz00677/PYTHON_project/Unet/recurrent/src/config_5R.json"
    config = load_config(config_path)
    print("finish loading config file")

    if checkpoint_save_path is None:
        if config["debug"]:
            checkpoint_save_path = "/vol/research/Dereverb/debug"
        else:
            checkpoint_save_path = "/vol/vssp/msos/jz/cIRM_base/checkpoint/"

    original_path = checkpoint_save_path
    logger = logging.getLogger("logger")
    #dropout = config["training"]["dropout"]
    #weight_decay = config["training"]["weight_decay"]
    #hidden_layers = config["model"]["hidden_layers"]
    loss_function = config["training"]["loss_function"] if loss is None else [loss]
    weight_loss = weight_loss_list.split(',') if weight_loss_list is not None else config["training"]["weight_loss"]
    stride_mode = stride_mode.split(',') if stride_mode is not None else config["model"]["stride_mode"]
    target = target_in if target_in is not None else config["training"]["target"]
    weight_decay = config["training"]["weight_decay"]
    bidirectional = config["model"]["bidirectional"]
    recurrent_type = config["model"]["recurrent_type"]
    loss_feature_type = ["phase", "GD", "IF", "both"]
    for ii in range(0,1):
        for jj in range(0,1):
            #config["model"]["hidden_layers"] = hidden_layers[ii]
            if config["training"]["loss_function"] == "MSE" and jj > 0:
                break

            config["training"]["loss_function"] = loss_function[ii]
            config["training"]["target"] = target[0]  #cIRM
            config["training"]["weight_loss"] = float(weight_loss[jj])
            config["training"]["input"] = "spec"
            config["training"]["target"] = target
            config["model"]["recurrent_type"] = "LSTM"
            config["model"]["bidirectional"] = True
            config["model"]["stride_mode"] = int(stride_mode[0])

            #if ii == 0 and jj > 0:
            #    break

            loss_train = np.array([])
            loss_test = np.array([])
            loss_mag_train = np.array([])
            loss_mag_test = np.array([])
            loss_phase_train = np.array([])
            loss_phase_test = np.array([])
            loss_angle_train = np.array([])
            loss_angle_test = np.array([])
            global_step = 0
            global_epoch = 0


            checkpoint_save_path = original_path + "strideMode_{}".format(config["model"]["stride_mode"])

            if config["training"]["loss_function"] == "MSE":
                checkpoint_save_path = checkpoint_save_path + "_MSE"
            else:
                checkpoint_save_path = checkpoint_save_path + "_{}_{}".format(config["training"]["loss_function"], config["training"]["weight_loss"])
            os.makedirs(checkpoint_save_path, exist_ok = True)
            handler = setHandler(filename = os.path.join(checkpoint_save_path, "train.log"))
            logger.addHandler(handler)
            logger.setLevel(logging.INFO)

            model = Model.UNet_recurrent(config)

            optimizer = torch.optim.Adam(model.parameters(), lr=config["training"]["learning_rate"],
                            betas=(0.9, 0.999), eps=1e-08, weight_decay=config["training"]["weight_decay"])

            #rir_list = preprocess.get_rir(config["data"]["rir"])
            print("start loading dataset")
            logger.info("start loading dataset")
            dataloader = dataset_5R.get_dataloader(config)
            print("finish loading dataset")
            logger.info("finish loading dataset")

            if checkpoint_load_path is not None:
                if not load_checkpoint(checkpoint_load_path, model, optimizer):
                    logger.info("failed to load given checkpoint")
                    model = Model.UNet_recurrent(config)
            else:
                try:
                    if not load_checkpoint(os.path.join(checkpoint_save_path, "checkpoint_current.pth"), model, optimizer):
                        logger.info("failed to checkpoint_current.pth")
                        model = Model.UNet_recurrent(config)
                except FileNotFoundError:
                    pass

            if config["training"]["loss_function"] == "MSE":
                criterion = nn.MSELoss().cuda()
            elif config["training"]["loss_function"] == "WPM":
                criterion = WPMLoss(config["training"]["weight_loss"]).cuda()
            else:
                criterion = GDMLoss(config["training"]["weight_loss"], loss_feature_type[0]).cuda()

            logger.info("config file is: " + config_path)
            logger.info(config)
            logger.info("checkpoint is saved at: " + checkpoint_save_path)


            try:
                logger.info(model)
                print("start training!")
                logger.info("starting training!")
                model = model.cuda()
                train(model, optimizer, criterion, dataloader, config["data"]["index_file"][1], config, checkpoint_save_path)
            except KeyboardInterrupt:
                print("Interrupted by Keyboard Input")
                pass
            finally:
                save_checkpoint(checkpoint_save_path, model, optimizer)
                plot_loss(loss_train[:,0], loss_test[:,0], loss_mag_train, loss_mag_test, loss_phase_train, loss_phase_test, loss_angle_train, loss_angle_test, checkpoint_save_path)

            logger.info("FINISH TRAINING!")
            logger.removeHandler(handler)
            print("FINISH TRAINING!")

            output_path = os.path.join(checkpoint_save_path, "output_result")
            os.makedirs(output_path, exist_ok = True)
            handler = setHandler(filename = os.path.join(output_path, "inference.log"))
            logger.addHandler(handler)

            best_model = model
            if not load_checkpoint(os.path.join(checkpoint_save_path, "best_loss_model.pth"), best_model, optimizer):
                logger.info("failed to load best_model.pth")
                best_model = model
            rir_file = "/user/HS228/jz00677/PYTHON_project/RIR_Generator/rir_5R_stepT60_test.csv"
            output_path = os.path.join(checkpoint_save_path, "output_result_step_T60_bestloss")
            os.makedirs(output_path, exist_ok = True)
            handler = setHandler(filename = os.path.join(output_path, "inference.log"))
            logger.addHandler(handler)
            inference.inference_step_T60(device = 'cuda', output_path = output_path, rir_file = rir_file, model = best_model, config = config, logger = logger)
            T60_list = ['0.3s', '0.4s', '0.5s', '0.6s', '0.7s', '0.8s', '0.9s', '1s', '1.1s', '1.2s', '1.3s', '1.4s', '1.5s']
            rir_per_T60 = 20
            n_audio = 10
            evaluation.evaluation_step_T60(output_path, "index.txt", T60_list, rir_per_T60, n_audio)
            evaluation.evaluation_mag_angle_dif(output_path, 'index.txt', config['data'])
            logger.info("evaluation finish!")
            logger.removeHandler(handler)


    sys.exit(0)
