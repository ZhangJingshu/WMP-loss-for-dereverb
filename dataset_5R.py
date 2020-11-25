import numpy as np
import torch
import torch.utils.data
import librosa

import pdb
import os

import util

class reverb_dataset(object):

    def __init__(self, config):
        self.config = config
        self.path = config['data']['path']
        self.sample_rate = config['data']['sample_rate']
        self.audio_filelist = {'train':[], 'test': []}
        self.rir_list = {'train':[], 'test':[]}
        self.speech_list = {'train':[], 'test':[]}
        self.speech_num_list = {'train': [], 'test': []}
        self.rir_num_list = {'train': [], 'test': []}
        self.n_speech = {'train':[], 'test':[]}
        self.n_rir = {'train':[], 'test':[]}
        self.n_sample = {'train': [], 'test': []}
        self.rir_file = {'train': os.path.join(config["data"]["rir_path"], config['data']['rir'][0]),
                        'test': os.path.join(config["data"]["rir_path"], config['data']['rir'][1])}
        self.audio_index_file = {"train": os.path.join(self.path, config["data"]["audio_index_file"][0]),
                        "test": os.path.join(self.path, config["data"]["audio_index_file"][1])}
        self.statistic = {"mean": [], "variance": []}   #compute on all training set
        self.load_data()
        #self.read_audio_filelist()


    def load_audio_filelist(self):
        path = self.path
        audio_path = self.config["data"]["audio_path"]
        for phase in ["train","test"]:
            with open(os.path.join(path, self.audio_index_file[phase]), 'r') as f:
                filelist = f.readlines()
            if self.config['debug'] is True:
                filelist = filelist[0:20]
            filelist = [os.path.join(audio_path, filename.replace('\n', '')) for filename in filelist]
            self.audio_filelist[phase] = filelist
            self.n_speech[phase] = len(filelist)
        self.audio_filelist["test"] = self.audio_filelist["test"][0: 15]   #use 15 speech combined with dev RIRs for validation
        self.n_speech["test"] = 15


    def load_rir(self):
        #input is rir file path
        for phase in ["train", "test"]:
            with open(self.rir_file[phase], 'r') as f:
                content = f.readlines()
                del content[0]
        #content contains n strings, each of which is an rir sequence
        #the first line of csv file is the indices of columns
            for input in content:
                input = input.split(',')
                input = list(filter(None, input))
                try:
                    input.remove('\n')
                except ValueError:
                    pass
                rir = np.array(input).astype(float)
                #head = np.nonzero(abs(rir) > 1e-3)[0][0]
                #rir = rir[head:]
                rir = librosa.effects.trim(rir, top_db = 20, frame_length = self.config['data']['frame_length'], hop_length = self.config['data']['hop_length'])[0]
                self.rir_list[phase].append(rir)
            #find the actual start of rir, points before head represents the time when sound haven't reached the mic
            self.n_rir[phase] = len(self.rir_list[phase])


    def load_speech(self):
        sr = self.sample_rate
        for phase in ["train", "test"]:
            for ii in range(0, self.n_speech[phase]):
                wav = librosa.core.load(self.audio_filelist[phase][ii], sr)[0]
                wav = librosa.effects.trim(wav, top_db = 20, frame_length = self.config['data']['frame_length'], hop_length = self.config['data']['hop_length'])[0]
                self.speech_list[phase].append(wav)


    def load_data(self):
        path = self.path
        self.load_audio_filelist()
        self.load_rir()
        self.load_speech()
        self.n_sample["train"] = self.n_rir["train"] * 50
        self.n_sample["test"] = self.n_rir["test"] * 15
        #print("start loading data")

    def get_data(self, idx, phase):

        rir_idx = idx[0]
        speech_idx = idx[1]

        wav_clean = self.speech_list[phase][speech_idx]
        rir = self.rir_list[phase][rir_idx]
        length = len(wav_clean)

        return (wav_clean, rir, length)  #shape: t * f


class PyTorchDataset(object):
    def __init__(self, dataset, phase):
        self.dataset = dataset
        self.phase = phase

    def __getitem__(self, idx):
        return self.dataset.get_data(idx, self.phase)

    def __len__(self):
        return self.dataset.n_sample[self.phase]


class MySampler(torch.utils.data.Sampler):

    def __init__(self, n_rir, n_speech, n_sample):
        self.n_sample = n_sample
        self.n_rir = n_rir
        self.n_speech = n_speech

    def __len__(self):
        return self.n_sample

    def __iter__(self):
        indices_speech = []
        indices_rir = np.arange(0, self.n_rir)
        indices_rir = np.repeat(indices_rir, self.n_sample // self.n_rir)

        for ii in range(0, self.n_rir):
            indices_speech.append(np.random.choice(np.arange(0, self.n_speech), self.n_sample // self.n_rir))
        indices_speech = np.array(indices_speech)

        indices = np.concatenate((indices_rir.reshape(-1, 1), indices_speech.reshape(-1, 1)), 1)

        return iter(indices)


def collate_fn(batch):
    """
    input: batch -> (wav_clean, rir, length)
    output: wav_reverb, wav_clean
    size: batch_size, t
    """
    x_batch = None
    y_batch = None
    length = []
    #y_batch_real = None
    #y_batch_imag = None
    batch_size = len(batch)
    for item in batch:
        length.append(item[2])

    length = np.array(length)
    max_length = np.amax(length)

    for item in batch:
        if item[2] < max_length:
            pad = (max_length - item[2])
            y = np.pad(item[0], [0, pad], mode = 'constant', constant_values = 0)
            x = np.convolve(item[0], item[1], mode = 'full')[0:max_length]
        else:
            y = item[0]
            x = np.convolve(item[0], item[1], mode = 'full')[0:max_length]

        if len(x) < max_length:
            x = np.pad(x, [0, max_length-len(x)], mode = 'constant', constant_values = 0)
        x_batch = np.array([x]) if x_batch is None else np.append(x_batch, np.array([x]), axis = 0)
        y_batch = np.array([y]) if y_batch is None else np.append(y_batch, np.array([y]), axis = 0)

    x_batch = torch.FloatTensor(np.array(x_batch))
    y_batch = torch.FloatTensor(np.array(y_batch))
    assert x_batch.size(0) == batch_size
    assert y_batch.size(0) == batch_size

    return x_batch, y_batch  #(batch_size, t) waveform of reverb speech and clean speech
    #shape: (batch_size, n_features)

def get_dataloader(config):
    datasource = reverb_dataset(config)
    batch_size = config["training"]["batch_size"]
    dataloader = {}
    for phase in ["train","test"]:
        dataset = PyTorchDataset(datasource, phase)
        #shuffle = True if phase == "train" else False
        sampler = MySampler(n_rir = dataset.dataset.n_rir[phase], n_speech = dataset.dataset.n_speech[phase], n_sample = len(dataset))
        loader = torch.utils.data.DataLoader(dataset, batch_size, sampler = sampler,
                    collate_fn = collate_fn, num_workers = 2, pin_memory = True)
        dataloader[phase] = loader
    return dataloader
