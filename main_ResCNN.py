#!/usr/bin/env python3

import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from Models.ResCNN.ResCNN import ResCNN
from Models.ResCNN.EegDataset import EegDataset

writer = SummaryWriter('ResCNN_tensorboard')

batch_size = 32
lr = 1e-4
epochs = 50
dropout_p = 0.4

train_dir = './data/train'
val_dir = './data/val'
train_data_file = 'train_data.pt'
train_label_file = 'train_label.pt'
val_data_file = 'val_data.pt'
val_label_file = 'val_label.pt'

save_path = './Models/ResCNN/weights/ResNet.pth'
if not os.path.exists('./Models/ResCNN/weights/'):
    os.system('mkdir -p ./Models/ResCNN/weights/')



if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Using {} device".format(device))
    num_workers = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])
    # num_workers = 0
    print("Using {} dataloader workers every process".format(num_workers))

    train_dataset = EegDataset(os.path.join(train_dir, train_data_file),
                               os.path.join(train_dir, train_label_file))
    val_dataset = EegDataset(os.path.join(val_dir, val_data_file),
                              os.path.join(val_dir, val_label_file))
    train_num = len(train_dataset)
    val_num = len(val_dataset)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    print("Load dataset: \n"
          "Train Dataset: %d samples\n"
          "val Dataset: %d samples" % (len(train_dataset), len(val_dataset)))

    example_data, example_label = train_dataset[0]
    n_channel, h, w = example_data.shape

    model = ResCNN(in_channel=n_channel,
                   out_channel=3,
                   h=h,
                   w=w,
                   dropout_p=dropout_p)
    model.to(device=device)

    # define loss function
    loss_function = torch.nn.CrossEntropyLoss()

    # construct an optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(params, lr=lr)

    best_acc = 0.0

    train_steps = len(train_dataloader)

    for epoch in range(epochs):
        # train
        model.train()
        running_loss = 0.0
        train_bar = tqdm(train_dataloader)
        for step, data in enumerate(train_bar):
            signals, labels = data
            optimizer.zero_grad()
            logits = model(signals.to(device))
            loss = loss_function(logits, labels.to(device).to(torch.long))
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            train_bar.desc = "train epoch [{}/{}] loss: {:.3f}".format(epoch+1, epochs, loss)

        model.eval()
        acc = 0.0
        with torch.no_grad():
            val_bar = tqdm(val_dataloader)
            for val_data in val_bar:
                val_signals, val_labels = val_data
                outputs = model(val_signals.to(device))
                predict = torch.max(outputs, dim=1)[1]
                acc += torch.eq(predict, val_labels.to(device)).sum().item()

                val_bar.desc = "val epoch[{}/{}]".format(epoch+1, epochs)
        val_accuracy = acc/val_num
        print("[epoch %d] train_loss: %.3f val_accuracy: %.3f" % (epoch+1, running_loss/train_steps, val_accuracy))

        writer.add_scalar(tag='val_acc:', scalar_value=val_accuracy, global_step=epoch)
        writer.add_scalar(tag='loss', scalar_value=running_loss / train_steps, global_step=epoch)

        if val_accuracy > best_acc:
            best_acc = val_accuracy
            torch.save(model.state_dict(), save_path)


