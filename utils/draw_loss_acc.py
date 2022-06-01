import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np

if __name__ == '__main__':
    loss_data = pd.read_csv('./loss.csv')
    step = np.array(loss_data['Step'])+1
    loss = np.array(loss_data['Value'])
    
    acc_data = pd.read_csv('./acc.csv')
    acc = np.array(acc_data['Value'])

    mpl.rcParams['font.sans-serif'] = 'Times New Roman'
    fig, axs = plt.subplots(1, 2, figsize=(10, 4))
    ax1 = axs[0]
    ax1.set_xlabel('Epochs')  # Add an x-label to the axes.
    ax1.set_ylabel('Loss')  # Add a y-label to the axes.
    ax1.plot(step, loss)
    ax2 = axs[1]
    ax2.set_xlabel('Epochs')  # Add an x-label to the axes.
    ax2.set_ylabel('Accuracy')  # Add a y-label to the axes.
    ax2.plot(step, acc)
    plt.savefig('loss_acc.pdf', format='pdf', bbox_inches='tight', transparent=True, dpi=600)
    plt.show()

