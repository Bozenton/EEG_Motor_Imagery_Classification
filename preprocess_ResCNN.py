import os
import logging
import argparse
import numpy as np
from copy import deepcopy
import torch
import mne
from utils.transform_to_standard import transform_to_standard

# Config
edf_dir = '/home/featurize/data'
train_dir = './data/train/'
test_dir = './data/test/'
val_dir = './data/val/'
subfix = '.edf'
file_name_format = 'S%03dR%02d'
NUM_SUBJECTS = 80
RUNS = [4, 8, 12]  # runs of task 2: imagine opening and closing left or right fist
power_line_freq = 60
highpass_cutoff = 1.0  # Hz
ICA_components = 10
t_min, t_max = -0.5, 3.5  # define epochs around events (in s)

train_percent = 0.7
val_percent = 0.1

def preprocess(file_path, viz=False):
    assert os.path.exists(file_path), \
        "The file {} does not exist, please check your input".format(file_path)
    raw = mne.io.read_raw_edf(file_path, preload=True)
    original_bad_channels = deepcopy(raw.info['bads'])
    if viz:
        raw.plot(duration=30, n_channels=len(raw.ch_names), scalings={'eeg': 200e-6}, remove_dc=False)
        plt.show()

    # basic info of the data
    sample_freq = raw.info.get('sfreq')
    ch_names = raw.info.get('ch_names')

    # add locations info
    raw_ch_names = raw.info.get('ch_names')
    montage = mne.channels.make_standard_montage('standard_1020')
    mapping = transform_to_standard(raw_ch_names, montage.ch_names)
    raw.rename_channels(mapping)
    raw.set_montage(montage, on_missing='raise', verbose=None)
    if viz:
        fig = montage.plot(kind='3d', show_names=False)
        fig.gca().view_init(azim=20, elev=15)  # set view angle
        plt.show()

    # set the EEG reference
    raw.set_eeg_reference(ref_channels='average')

    # interpolating bad channels
    if len(raw.info['bads']) > 0:
        raw.interpolate_bads()

    # Filter
    # Power line noise
    raw.notch_filter(freqs=(power_line_freq,))
    if viz:
        raw.plot_psd(tmax=np.inf, fmax=sample_freq / 2)
    # Slow drift
    raw.filter(l_freq=highpass_cutoff, h_freq=None)

    if viz:  # that is, only one sample, thus we can do ICA mannually
        ica = mne.preprocessing.ICA(n_components=ICA_components, max_iter='auto', random_state=97)
        ica.fit(raw)
        raw.load_data()
        ica.plot_sources(raw)
        ica.plot_components()
        ica.apply(raw)
        raw.plot(duration=20, n_channels=10, scalings={'eeg': 200e-6}, remove_dc=False)

    # Parsing Events
    events_from_annot, event_dict = mne.events_from_annotations(raw)
    if viz:
        mne.viz.plot_events(events_from_annot,
                            sfreq=raw.info['sfreq'],
                            first_samp=raw.first_samp,
                            event_id=event_dict)
    epochs = mne.Epochs(raw,
                        events_from_annot,
                        event_dict,
                        t_min,
                        t_max - 1.0 / raw.info['sfreq'],  # make sure that each length of each epoch is a nice integer
                        preload=True)

    # Access to the data
    data = epochs.get_data()
    events = epochs.events[:, 2]

    return data, events


def str2bool(s):
    return True if s.lower() == 'true' else False


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--all", default=True, help="Preprocess all the data", type=str2bool)
    args = parser.parse_args()

    LOG_FORMAT = "[%(levelname)s] %(message)s"
    logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)

    if not args.all:
        data_files = [file_name_format % (1, 4), ]
        viz = True
    else:
        data_files = [''] * NUM_SUBJECTS * len(RUNS)
        for i in range(0, NUM_SUBJECTS):
            for j, run_idx in enumerate(RUNS):
                data_files[3 * i + j] = file_name_format % (i + 1, run_idx)
        viz = False

    data = []
    events = []
    for _, name in enumerate(data_files):
        file_name = name + subfix
        file_path = os.path.join(edf_dir, file_name)
        d, e = preprocess(file_path, viz=viz)
        data.append(d)
        events.append(e)
        logging.info('Preprocessed data {}'.format(name))
    data = torch.as_tensor(np.array(data), dtype=torch.float32)
    events = torch.as_tensor(np.array(events), dtype=torch.float32)

    if not args.all:
        logging.info("Exit without saving data")
        exit()

    n_subjects, n_events, n_channels, n_points = data.shape

    all_data = data.reshape([n_subjects*n_events, n_channels, n_points])
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
    torch.save(train_data, train_dir+'train_data.pt')
    torch.save(train_label, train_dir+'train_label.pt')
    torch.save(test_data, test_dir+'test_data.pt')
    torch.save(test_label, test_dir+'test_label.pt')
    torch.save(val_data, val_dir+'val_data.pt')
    torch.save(val_label, val_dir+'val_label.pt')
    logging.info("Data saved!")
