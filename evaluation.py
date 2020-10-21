"""
usage: evaluation.py [options]

options:
--dir=<path>            path of the folder that contains score files
--num-T60=<int>         total number of RIR conditions
--rir-per-T60=<int>     number of RIRs for each T60 condition
--num-audio=<int>       number of speech used
--help                  show this help message
"""

import librosa
import numpy as np
import csv
import json

from pypesq import pypesq
import mypesq
import stoi
import fwseg_snr

from docopt import docopt
import pdb
import sys
import os


def create_table_noise(result, noise_type, snr_list):
    n_eval = len(result)
    n_noise = len(noise_type)
    n_snr = len(snr_list)
    table = []

    for item in result:
        table.append([item])
        headline = ["noise type"]
        for noise_name in noise_type:
            headline = headline + [noise_name] + [""] * (n_snr - 1)
        table.append(headline)
        table.append(["snr"] + snr_list * n_noise)
        for condition in result[item]:
            line = [condition]
            data = np.mean(result[item][condition], axis = 2)
            data_out = data.reshape(n_snr * n_noise)
            data_out = list(data_out)
            line = line + data_out
            table.append(line)
        table.append([""])

    return table


def create_table_reverb(result, T60_list, rir_per_T60):
    n_eval = len(result)
    n_T60 = len(T60_list)
    table = []

    for item in result:
        table.append([item])
        headline = ["T60"]
        for T60 in T60_list:
            #headline = headline + [T60] + [""] * (rir_per_T60)
            headline = headline + [T60]
        table.append(headline)
        #table.append(["RIR"] + (list(range(0, rir_per_T60)) + ['average']) * n_T60)
        for condition in result[item]:
            line = [condition]
            #data = np.mean(result[item][condition], axis = 2)
            #for T60 in range(0, n_T60):
            #    data_out = data[T60, :]
            #    data_out = list(data_out) + [np.mean(data[T60, :])]
            #    line = line + data_out
            data = np.mean(result[item][condition], axis = (1,2))
            assert data.ndim == 1 and len(data) == n_T60
            line = line + list(data)
            table.append(line)
        table.append([""])

    return table


def write_table(path, filename, table):
    with open(os.path.join(path, filename), 'w', newline = '') as f:
        writer = csv.writer(f, delimiter = ',', quotechar = '\'', quoting = csv.QUOTE_MINIMAL)
        for lines in table:
            writer.writerow(lines)



def write_raw_scores(path, result, audio_list):
    for item in result:
        table = []
        for condition in result[item].keys():
            data = result[item][condition].reshape(-1)
            assert len(data) == len(audio_list[condition])
            for ii in range(0, len(data)):
                line = [audio_list[condition][ii], data[ii]]
                table.append(line)
        filename = "{}_score.csv".format(item)
        write_table(path, filename, table)



def compute_scores(audiofile_ref, audiofile_deg, sr = 16000):
    wav_ref = librosa.core.load(audiofile_ref, sr)[0]
    wav_deg = librosa.core.load(audiofile_deg, sr)[0]

    pesq_score = pypesq(sr, wav_ref, wav_deg, 'nb')[0]
    stoi_score = stoi.stoi(wav_ref, wav_deg, sr)
    fwsnr_score = fwseg_snr.fwSNRseg(wav_ref, wav_deg, sr)

    return pesq_score, stoi_score, fwsnr_score


def compute_mag_angle_dif(audio_ref, audio_deg, config):
    sr = config["sample_rate"]
    wav_ref = librosa.core.load(audio_ref, sr)[0]
    wav_deg = librosa.core.load(audio_deg, sr)[0]

    spec_ref = librosa.core.stft(wav_ref, n_fft = config["n_fft"],
                    hop_length = config["hop_length"], window = "hann",
                    center = True, dtype=np.complex64, pad_mode='reflect')
    spec_deg = librosa.core.stft(wav_deg, n_fft = config["n_fft"],
                    hop_length = config["hop_length"], window = "hann",
                    center = True, dtype=np.complex64, pad_mode='reflect')

    mag_ref, phase_ref = librosa.core.magphase(spec_ref)
    mag_deg, phase_deg = librosa.core.magphase(spec_deg)
    angle_ref = np.angle(phase_ref)
    angle_deg = np.angle(phase_deg)

    mag_dif = np.mean(np.abs(mag_ref - mag_deg))
    angle_dif = np.mean(np.abs(angle_ref - angle_deg))

    return mag_dif, angle_dif


def evaluation_mag_angle_dif(path, index_file, config):
    with open(os.path.join(path, index_file), "r") as f_in:
        lines = f_in.readlines()
    list_in = [item.replace('\n', '').replace('|', '\t').split('\t') for item in lines]
    list_in = np.array(list_in)
    audio_list = {"estm": list_in[:, 0], "ref": list_in[:, 1], "deg": list_in[:, 2]}
    n_audio = len(audio_list["ref"])
    mag_dif = {"estm":np.zeros(n_audio), "deg":np.zeros(n_audio)}
    angle_dif = {"estm":np.zeros(n_audio), "deg":np.zeros(n_audio)}

    for ii in range(0, len(audio_list["ref"])):
        audio_estm = os.path.join(path, audio_list["estm"][ii])
        audio_ref = os.path.join(path, audio_list["ref"][ii])
        audio_deg = os.path.join(path, audio_list["deg"][ii])
        mag_dif["deg"][ii], angle_dif["deg"][ii] = compute_mag_angle_dif(audio_ref, audio_deg, config)
        mag_dif["estm"][ii], angle_dif["estm"][ii] = compute_mag_angle_dif(audio_ref, audio_estm, config)
        if ii % 100 == 0:
            print("finish evaluating 100 audio files")

    mag_dif_mean = {"estm": np.mean(mag_dif["estm"]), "deg": np.mean(mag_dif["deg"])}
    angle_dif_mean = {"estm": np.mean(angle_dif["estm"]), "deg": np.mean(angle_dif["deg"])}

    with open(os.path.join(path, 'raw_mag_angle_dif.txt'), 'r') as f:
        lines = f.readlines()
    raw_dif = [item.split(',') for item in lines]
    raw_mag_dif = [float(item[0].split(':')[1]) for item in raw_dif]
    raw_angle_dif = [float(item[1].split(':')[1]) for item in raw_dif]
    raw_mag_dif_mean = np.mean(np.array(raw_mag_dif))
    raw_angle_dif_mean = np.mean(np.array(raw_angle_dif))

    with open(os.path.join(path, "mag_angle_dif.txt"), 'w') as f:
        f.write("mean magnitude difference:\n")
        f.write("estm: {}\ndegraded:{}\n".format(mag_dif_mean["estm"], mag_dif_mean["deg"]))
        f.write("mean angle difference:\n")
        f.write("estm: {}\ndegraded:{}\n".format(angle_dif_mean["estm"], angle_dif_mean["deg"]))
        f.write("raw magnitude difference:\n")
        f.write("estm: {}\n".format(raw_mag_dif_mean))
        f.write("raw angle difference:\n")
        f.write("estm: {}\n".format(raw_angle_dif_mean))
    print("finish evaluating mag and angle")


def evaluation_all(path, index_file):
    with open(os.path.join(path, index_file), 'r') as f:
        filelist = f.readlines()
    filelist = [filename.replace('\n', '') for filename in filelist]
    pesq_estm = []
    pesq_noisy = []
    stoi_estm = []
    stoi_noisy = []
    fwsnr_estm = []
    fwsnr_noisy = []
    sr = 16000
    n_audio = len(filelist)
    for item in filelist:
        wav_clean = librosa.core.load(os.path.join(path, item[1]), sr = 16000)[0]
        wav_noisy = librosa.core.load(os.path.join(path, item[2]), sr = 16000)[0]
        wav_estm = librosa.core.load(os.path.join(path, item[0], sr = 16000))[0]

        pesq_estm.append("{}: {}\n".format(item[0], pypesq(sr, wav_clean, wav_estm, 'nb')[0]))
        pesq_noisy.append("{}: {}\n".format(item[2], pypesq(sr, wav_clean, wav_noisy, 'nb')[0]))
        stoi_estm.append("{}: {}\n".format(item[0], stoi.stoi(wav_clean, wav_estm, sr)))
        stoi_noisy.append("{}: {}\n".format(item[2], stoi.stoi(wav_clean, wav_noisy, sr)))
        fwsnr_estm.append("{}: {}\n".format(item[0], fwseg_snr.fwSNRseg(wav_clean, wav_estm, sr)))
        fwsnr_noisy.append("{}: {}\n".format(item[2], fwseg_snr.fwSNRseg(wav_clean, wav_noisy, sr)))

    with open(os.path.join(path, "pesq_score.txt"), 'w') as f:
        f.writelines(pesq_estm)
        f.writelines(pesq_noisy)
    with open(os.path.join(path, "stoi_score.txt"), 'w') as f:
        f.writelines(stoi_estm)
        f.writelines(stoi_noisy)
    with open(os.path.join(path, "fw_seg_snr.txt"), 'w') as f:
        f.writelines(fwsnr_estm)
        f.writelines(fwsnr_noisy)

    pesq, pesq_avg = read_score_1(os.path.join(path, "pesq_score.txt"), n_audio = n_audio)
    stoi, stoi_avg = read_score_1(os.path.join(path, "stoi_score.txt"), n_audio = n_audio)
    fwsnr, fwsnr_avg = read_score_1(os.path.join(path, "fw_seg_snr.txt"), n_audio = n_audio)

    write_final_result_1(os.path.join(path, "final_results.txt"), pesq_avg, stoi_avg, fwsnr_avg)


def evaluation_3T60(path, index_file, T60_list, rir_per_T60, n_audio):

    n_T60 = len(T60_list)

    result = {}
    init = np.zeros([n_T60, rir_per_T60, n_audio])
    result["pesq"] = {"estm": np.copy(init), "reverb": np.copy(init)}
    result["stoi"] = {"estm": np.copy(init), "reverb": np.copy(init)}
    result["fwsnr"] = {"estm": np.copy(init), "reverb": np.copy(init)}
    file_dir = os.path.join(path, index_file)

    with open(file_dir, "r") as f_in:
        lines = f_in.readlines()
    list_in = [item.replace('\n', '').replace('|', '\t').split('\t') for item in lines]
    list_in = np.array(list_in)
    audio_list = {"estm": list_in[:, 0], "ref": list_in[:, 1], "reverb": list_in[:, 2]}

    for ii in range(0, n_T60):
        for jj in range(0, rir_per_T60):
            for kk in range(0, n_audio):
                index = ii * rir_per_T60*n_audio + jj * n_audio + kk
                audio_estm = os.path.join(path, audio_list["estm"][index])
                audio_ref = os.path.join(path, audio_list["ref"][index])
                audio_noisy = os.path.join(path, audio_list["reverb"][index])
                result["pesq"]["estm"][ii][jj][kk], result["stoi"]["estm"][ii][jj][kk], result["fwsnr"]["estm"][ii][jj][kk] = compute_scores(audio_ref, audio_estm, sr = 16000)
                result["pesq"]["reverb"][ii][jj][kk], result["stoi"]["reverb"][ii][jj][kk], result["fwsnr"]["reverb"][ii][jj][kk] = compute_scores(audio_ref, audio_noisy, sr = 16000)
            print("finish evaluating {}, RIR {}".format(T60_list[ii], jj))

    write_raw_scores(path, result, audio_list)
    output_table = create_table_reverb(result, T60_list, rir_per_T60)
    write_table(path, 'final_result.csv', output_table)
    print("finish evaluation")


def evaluation_step_T60(path, index_file, T60_list, rir_per_T60, n_audio):
    n_T60 = len(T60_list)

    result = {}
    init = np.zeros([n_T60, rir_per_T60, n_audio])
    result["pesq"] = {"estm": np.copy(init), "reverb": np.copy(init)}
    result["stoi"] = {"estm": np.copy(init), "reverb": np.copy(init)}
    result["fwsnr"] = {"estm": np.copy(init), "reverb": np.copy(init)}
    file_dir = os.path.join(path, index_file)

    with open(file_dir, "r") as f_in:
        lines = f_in.readlines()
    list_in = [item.replace('\n', '').replace('|', '\t').split('\t') for item in lines]
    list_in = np.array(list_in)
    audio_list = {"estm": list_in[:, 0], "ref": list_in[:, 1], "reverb": list_in[:, 2]}

    for ii in range(0, n_T60):
        for jj in range(0, rir_per_T60):
            for kk in range(0, n_audio):
                index = ii * rir_per_T60*n_audio + jj * n_audio + kk
                audio_estm = os.path.join(path, audio_list["estm"][index])
                audio_ref = os.path.join(path, audio_list["ref"][index])
                audio_noisy = os.path.join(path, audio_list["reverb"][index])
                result["pesq"]["estm"][ii][jj][kk], result["stoi"]["estm"][ii][jj][kk], result["fwsnr"]["estm"][ii][jj][kk] = compute_scores(audio_ref, audio_estm, sr = 16000)
                result["pesq"]["reverb"][ii][jj][kk], result["stoi"]["reverb"][ii][jj][kk], result["fwsnr"]["reverb"][ii][jj][kk] = compute_scores(audio_ref, audio_noisy, sr = 16000)
            print("finish evaluating {}, RIR {}".format(T60_list[ii], jj))

    write_raw_scores(path, result, audio_list)
    output_table = create_table_reverb(result, T60_list, rir_per_T60)
    write_table(path, 'final_result.csv', output_table)
    print("finish evaluation")



def evaluation_noise_type(path, index_file, noise_type, snr_list, n_audio, sr = 16000):

    n_noise = len(noise_type)
    n_snr = len(snr_list)

    result = {}
    init = np.zeros([n_noise, n_snr, n_audio])
    result["pesq"] = {"estm": np.copy(init), "noisy": np.copy(init)}
    result["stoi"] = {"estm": np.copy(init), "noisy": np.copy(init)}
    result["fwsnr"]= {"estm": np.copy(init), "noisy": np.copy(init)}
    file_dir = os.path.join(path, index_file)

    with open(file_dir, "r") as f_in:
        lines = f_in.readlines()
    list_in = [item.replace('\n', '').split('\t') for item in lines]
    list_in = np.array(list_in)
    audio_list = {"estm": list_in[:, 0], "ref": list_in[:, 1], "noisy": list_in[:, 2]}

    for ii in range(0, n_noise):
        for jj in range(0, n_snr):
            for kk in range(0, n_audio):
                index = ii *n_snr*n_audio + jj * n_audio + kk
                audio_estm = os.path.join(path, audio_list["estm"][index])
                audio_ref = os.path.join(path, audio_list["ref"][index])
                audio_noisy = os.path.join(path, audio_list["noisy"][index])
                result["pesq"]["estm"][ii][jj][kk], result["stoi"]["estm"][ii][jj][kk], result["fwsnr"]["estm"][ii][jj][kk] = compute_scores(audio_ref, audio_estm, sr = 16000)
                result["pesq"]["noisy"][ii][jj][kk], result["stoi"]["noisy"][ii][jj][kk], result["fwsnr"]["noisy"][ii][jj][kk] = compute_scores(audio_ref, audio_noisy, sr = 16000)
            print("finish evaluating {} and {}".format(noise_type[ii], snr_list[jj]))

    write_raw_scores(path, result, audio_list)
    output_table = create_table_noise(result, noise_type, snr_list)
    write_table(path, 'final_result.csv', output_table)
    print("finish evaluation")

if __name__ == "__main__":
    args = docopt(__doc__)
    path_in = args["--dir"]
    path = path_in if path_in is not None else "/vol/research/Dereverb/Unet_recurrent/checkpoint_noisy_MSE/output_result"
    config_file = "/user/HS228/jz00677/PYTHON_project/Unet/recurrent/src/config_5R.json"
    with open(config_file, 'r') as f:
        config = json.load(f)

    noise_type = ["factory", "living room", "office", "bus", "cafe", "square", "park"]
    snr_list = [-5, 0, 5, 10]
    n_audio = 50

    #evaluation_noise_type(path = path, index_file = "index.txt", noise_type = noise_type, snr_list = snr_list, n_audio = n_audio)
    #T60_list = ['0.3s', '0.6s', '0.9s']
    T60_list = ['0.3s', '0.4s', '0.5s', '0.6s', '0.7s', '0.8s', '0.9s', '1s', '1.1s', '1.2s', '1.3s', '1.4s', '1.5s']
    rir_per_T60 = 20
    n_audio = 10
    evaluation_step_T60(path, "index.txt", T60_list, rir_per_T60, n_audio)
    evaluation_mag_angle_dif(path, 'index.txt', config['data'])
    sys.exit(0)
