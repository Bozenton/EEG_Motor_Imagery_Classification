import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter


class ResCNN(torch.nn.Module):
    def __init__(self, in_channel=64, out_channel=3, h=32, w=20, dropout_p=0.2):
        super(ResCNN, self).__init__()
        # input size: [32, 20]
        ch1 = 256
        self.conv1 = nn.Conv2d(in_channels=in_channel, out_channels=ch1, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(ch1)
        self.relu1 = nn.LeakyReLU()
        self.drop1 = nn.Dropout2d(p=dropout_p)
        # --------------------------------------------------------
        self.conv2 = nn.Conv2d(in_channels=ch1, out_channels=ch1, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(ch1)
        self.relu2 = nn.LeakyReLU()
        # --------------------------------------------------------
        ch3 = 512
        self.conv3 = nn.Conv2d(in_channels=ch1, out_channels=ch3, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(ch3)
        self.relu3 = nn.LeakyReLU()
        self.drop3 = nn.Dropout2d(p=dropout_p)
        # --------------------------------------------------------
        self.mp3 = nn.MaxPool2d(2, stride=2) # h, w -> 1/2*h, 1/2*w
        # --------------------------------------------------------
        self.conv4 = nn.Conv2d(in_channels=ch3, out_channels=ch3, kernel_size=3, stride=1, padding=0)
        self.bn4 = nn.BatchNorm2d(ch3)
        self.relu4 = nn.LeakyReLU()
        self.drop4 = nn.Dropout2d(p=dropout_p)
        # --------------------------------------------------------
        self.conv5 = nn.Conv2d(in_channels=ch3, out_channels=ch3, kernel_size=3, stride=1, padding=1)
        self.bn5 = nn.BatchNorm2d(ch3)
        self.relu5 = nn.LeakyReLU()
        # --------------------------------------------------------
        ch6 = 1024
        self.conv6 = nn.Conv2d(in_channels=ch3, out_channels=ch6, kernel_size=3, stride=1, padding=1)
        self.bn6 = nn.BatchNorm2d(ch6)
        self.relu6 = nn.LeakyReLU()
        self.drop6 = nn.Dropout2d(p=dropout_p)
        # --------------------------------------------------------
        self.mp6 = nn.MaxPool2d(2, stride=2)
        # --------------------------------------------------------
        h1 = int((h/2-2)/2)
        w1 = int((w/2-2)/2)
        self.fc1 = nn.Linear(h1*w1*ch6, 512)
        self.fc1_bn = nn.BatchNorm1d(512)
        self.fc1_relu = nn.LeakyReLU()
        self.fc1_drop = nn.Dropout(p=dropout_p)
        # --------------------------------------------------------
        self.fc2 = nn.Linear(512, out_channel)
        self.sm = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.drop1(x)
        identity1 = x

        y = self.conv2(x)
        y = self.bn2(y)
        y = self.relu2(y)

        y = y + identity1
        y = self.conv3(y)
        y = self.bn3(y)
        y = self.relu3(y)
        y = self.drop3(y)
        y = self.mp3(y)

        y = self.conv4(y)
        y = self.bn4(y)
        y = self.relu4(y)
        y = self.drop4(y)
        identity2 = y

        z = self.conv5(y)
        z = self.bn5(z)
        z = self.relu5(z)

        z = z + identity2
        z = self.conv6(z)
        z = self.bn6(z)
        z = self.relu6(z)
        z = self.drop6(z)
        z = self.mp6(z)

        # Flatten Layer
        z = torch.flatten(z, 1)

        # First Fully Connected Layer
        z = self.fc1(z)
        z = self.fc1_bn(z)
        z = self.fc1_relu(z)
        z = self.fc1_drop(z)

        # Second Fully Connected Layer
        z = self.fc2(z)
        # z = self.sm(z)

        return z


if __name__ == "__main__":
    dummy_input = torch.randn(8, 64, 32, 20)
    model = ResCNN()
    with SummaryWriter(comment='ResCNN') as w:
        w.add_graph(model, (dummy_input,))
