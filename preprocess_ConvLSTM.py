import os
import re
import logging
import argparse
import numpy as np
import pandas as pd
from copy import deepcopy
import torch
import mne
from utils.transform_to_standard import transform_to_standard

# Config
edf_dir = './data/edf'
train_dir = './data/train/'
test_dir = './data/test/'
val_dir = './data/val/'
subfix = '.edf'
file_name_format = 'S%03dR%02d'
NUM_SUBJECTS = 1
RUNS = [4, 8, 12]  # runs of task 2: imagine opening and closing left or right fist
sample_freq = 160
power_line_freq = 60
highpass_cutoff = 1.0  # Hz
ICA_components = 15
t_min, t_max = -0.5, 3.5  # define epochs around events (in s)

train_percent = 0.7
val_percent = 0.1


def preprocess(file_path):
    assert os.path.exists(file_path), "The file {} does not exist, please check your input".format(file_path)
    raw = mne.io.read_raw_edf(file_path, preload=True)
    original_bad_channels = deepcopy(raw.info['bads'])

    # basic info of the data
    sample_freq = raw.info.get('sfreq')
    ch_names = raw.info.get('ch_names')

    # add locations info
    raw_ch_names = raw.info.get('ch_names')
    montage = mne.channels.make_standard_montage('standard_1020')
    mapping = transform_to_standard(raw_ch_names, montage.ch_names)
    raw.rename_channels(mapping)
    raw.set_montage(montage, on_missing='raise', verbose=None)

    # set the EEG reference
    raw.set_eeg_reference(ref_channels='average')

    # interpolating bad channels
    if len(raw.info['bads']) > 0:
        raw.interpolate_bads()

    # Filter
    # Power line noise
    raw.notch_filter(freqs=(power_line_freq,))
    # Slow drift
    raw.filter(l_freq=highpass_cutoff, h_freq=None)

    # Parsing Events
    events_from_annot, event_dict = mne.events_from_annotations(raw)
    epochs = mne.Epochs(raw,
                        events_from_annot,
                        event_dict,
                        t_min,
                        t_max - 1.0 / raw.info['sfreq'],  # make sure that each length of each epoch is a nice integer
                        preload=True)

    # Access to the data
    data = epochs.get_data()
    events = epochs.events[:, 2]
    channel = epochs.ch_names

    return data, events, channel


ch_pos_csv = './electrode_positions.csv'
with open(ch_pos_csv, 'r') as f:
    content = f.readlines()
ch_pos_dict = {}
cnt = 0
for row, line in enumerate(content):
    l = re.split('\t|\n', line)
    for col, name in enumerate(l):
        if name is not '' and name is not '0':
            r = row
            c = col - 1
            if r >= 9 or c < 0 or c >= 9:
                continue
            ch_pos_dict[name] = [r, c]

if __name__ == '__main__':
    LOG_FORMAT = "[%(levelname)s] %(message)s"
    logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)

    data_files = [''] * NUM_SUBJECTS * len(RUNS)
    for i in range(0, NUM_SUBJECTS):
        for j, run_idx in enumerate(RUNS):
            data_files[3 * i + j] = file_name_format % (i + 1, run_idx)

    data = []
    events = []
    n_points = int((t_max - t_min) * sample_freq)  # 640
    h = 9
    w = 9
    for _, name in enumerate(data_files):
        file_name = name + subfix
        file_path = os.path.join(edf_dir, file_name)
        d, e, channel = preprocess(file_path)
        # d: n_events, n_channels, n_points
        img = np.zeros([d.shape[0], h, w, n_points])
        # img: n_events, h, w, n_points
        for i, ep in enumerate(d):
            for j, ch in enumerate(channel):
                if ch not in ch_pos_dict.keys():
                    continue
                pos = ch_pos_dict[ch]
                img[i, pos[0], pos[1], :] = d[i, j, :]
        img = np.transpose(img, (0, 3, 1, 2))
        # img: n_events, n_points, h, w,

        data.append(img)
        events.append(e)
        logging.info('Preprocessed data {}'.format(name))

    data = torch.as_tensor(np.array(data), dtype=torch.float32)
    events = torch.as_tensor(np.array(events), dtype=torch.float32)
    print(data.shape)
    print(events.shape)
    n_subjects, n_events, n_points, h, w = data.shape

    all_data = data.reshape([n_subjects * n_events, n_points, h, w])
    all_events = events.flatten()

    rand_idx = np.random.permutation(n_subjects*n_events)
    n_train = int(train_percent * all_data.shape[0])
    n_val = int(val_percent * all_data.shape[0])
    train_data = all_data[rand_idx[0:n_train], :]
    train_label = all_events[rand_idx[0:n_train]]
    test_data = all_data[rand_idx[n_train:(n_train+n_val)], :]
    test_label = all_events[rand_idx[n_train:(n_train+n_val)]]
    val_data = all_data[rand_idx[(n_train+n_val):], :]
    val_label = all_events[rand_idx[(n_train+n_val):]]

    logging.info("Saving train and val data now ...")
    torch.save(train_data, train_dir + 'train_data_ConvLSTM.pt')
    torch.save(train_label, train_dir + 'train_label_ConvLSTM.pt')
    torch.save(test_data, test_dir+'test_data_ConvLSTM.pt')
    torch.save(test_label, test_dir+'test_label_ConvLSTM.pt')
    torch.save(val_data, val_dir + 'val_data_ConvLSTM.pt')
    torch.save(val_label, val_dir + 'val_label_ConvLSTM.pt')
    logging.info("Data saved!")
