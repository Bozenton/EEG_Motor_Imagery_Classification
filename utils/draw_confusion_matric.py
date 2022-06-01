import os
import argparse
import numpy as np
import seaborn as sns

from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt 
import matplotlib as mpl

network = ['ResConv', 'ConvLSTM']

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--which", default=0, help="0 for ResConv and 1 for ConvLSTM", type=int)
    args = parser.parse_args()
    which_one = int(args.which)
    if which_one not in [0, 1]:
        raise ValueError("The parameter should be 0 or 1, please check")
    print("Draw the confusion matrix of "+network[which_one])
    label_file = '../data/pred/label_' + network[which_one] + '.npy'
    pred_file = '../data/pred/pred_' + network[which_one] + '.npy'

    assert os.path.exists(label_file), "Label file does not exist in {}".format(label_file)
    assert os.path.exists(pred_file), "Predict file does not exist in {}".format(pred_file)

    labels = np.load(label_file).astype(np.float32)
    predicts = np.load(pred_file).astype(np.float32)

    print("The precision is", np.sum(predicts==labels)/len(labels))

    # Set photo parameters
    mpl.rcParams['font.sans-serif'] = 'Times New Roman'
    mpl.rcParams['axes.unicode_minus'] = False

    # Draw confusion matrix 
    f, ax = plt.subplots()
    cm = confusion_matrix(labels, predicts, labels=[0,1,2])
    cm = np.around(cm/sum(cm), 3)

    cmap = sns.color_palette("Spectral", as_cmap=True)
    sns.heatmap(cm, annot=True, ax=ax, fmt='.4f', cmap=cmap)
    ax.set(xticklabels=['T0', 'T1', 'T2'], yticklabels=['T0', 'T1', 'T2'])
    ax.set_title(network[which_one]+' Confusion Matrix', fontsize=16, fontweight='bold')
    ax.set_xlabel('Predicted Class', fontsize=14, fontweight='bold')
    ax.set_ylabel('True Class', fontsize=14, fontweight='bold')
    plt.savefig(network[which_one]+'_Confusion_Matrix.pdf', format='pdf', bbox_inches='tight', transparent=True, dpi=600)
    plt.show()
    


