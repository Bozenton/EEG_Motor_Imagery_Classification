#!/usr/bin/env python3

import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
# from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from Models.ResCNN.ResCNN import ResCNN
from Models.ResCNN.EegDataset import EegDataset

# writer = SummaryWriter('ResCNN_tensorboard')

dropout_p = 0.5

test_dir = './data/test'
test_data_file = 'test_data.pt'
test_label_file = 'test_label.pt'
weights_path = './Models/ResCNN/weights/ResNet.pth'



if __name__ == '__main__':
    assert os.path.exists(weights_path), "The weights of ResCNN does not exist in {}".format(weights_path)
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Using {} device".format(device))

    test_dataset = EegDataset(os.path.join(test_dir, test_data_file),
                               os.path.join(test_dir, test_label_file))

    example_data, example_label = test_dataset[0]
    n_channel, h, w = example_data.shape

    model = ResCNN(in_channel=n_channel,
                   out_channel=3,
                   h=h,
                   w=w,
                   dropout_p=dropout_p)
    model.to(device=device)
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.eval()

    labels = []
    predicts = []
    for i, data in enumerate(test_dataset):
    # if True:
    #     data = test_dataset[0]
        sample, label = data
        sample = torch.unsqueeze(sample, dim=0)  # expand batch dimension
        labels.append(label)
        with torch.no_grad():
            predict = model(sample.to(device))
            predict = int(torch.max(predict, 1).indices[0])
            predicts.append(predict)
    labels = np.array(labels)
    predicts = np.array(predicts)
    
    pred_dir = './data/pred/'
    if not os.path.exists(pred_dir):
        os.system('mkdir -p ./data/pred/')
    
    label_path = os.path.join(pred_dir, 'label_ResConv.npy')
    pred_path = os.path.join(pred_dir, 'pred_ResConv.npy')
    np.save(label_path, labels)
    np.save(pred_path, predicts)
            
