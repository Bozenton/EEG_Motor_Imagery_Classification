import os
import numpy as np


def load_preprocessed(filename, ch_num: int = 64):
    assert os.path.exists(filename), \
        "Path of the preprocessed file: {} does not exist".format(filename)
    raw_data = np.genfromtxt(filename, delimiter=',')
    data = raw_data.reshape([-1, ch_num, raw_data.shape[-1]])
    return data


if __name__ == '__main__':
    # test load_pre
    d = load_preprocessed('../data/preprocessed/S001R04.csv')
    print(d.shape)
