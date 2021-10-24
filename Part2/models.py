from numpy.__config__ import show
import torch
import torch.nn as nn
import torch.nn.functional as F
from datetime import datetime
import numpy as np
import cv2

CHROMREG_MINIBATCH_SIZE = 10
COLORIZE_MINIBATCH_SIZE = 8

# REF: https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html
class Chrominance_Regressor(nn.Module):
    def __init__(self):
        super().__init__()
        # implementation of convolution downsampling and FC layers for regression output
        self.conv1 = nn.Conv2d(1, 16, 3, stride = 1, padding = 1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 32, 3, stride = 1, padding = 1)
        self.conv3 = nn.Conv2d(32, 64, 3, stride = 1, padding = 1)
        self.conv4 = nn.Conv2d(64, 64, 3, stride = 1, padding = 1)
        self.conv5 = nn.Conv2d(64, 128, 3, stride = 1, padding = 1)
        self.fc1 = nn.Linear(4 * 4 * 128, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 128)
        self.fc4 = nn.Linear(128, 64)
        self.fc5 = nn.Linear(64, 32)
        self.fc6 = nn.Linear(32, 16)
        self.fc7 = nn.Linear(16, 8)
        self.fc8 = nn.Linear(8, 2)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = self.pool(F.relu(self.conv4(x)))
        x = self.pool(F.relu(self.conv5(x)))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = F.relu(self.fc5(x))
        x = F.relu(self.fc6(x))
        x = F.relu(self.fc7(x))
        x = self.fc8(x)
        return x

#Regressor
def train_chrominance_reg(trainloader, device):
    EPOCH_COUNT = 16
    LEARNING_RATE = 0.001

    net = Chrominance_Regressor()

    if torch.cuda.device_count() > 1:
        net = nn.DataParallel(net)

    net.to(device)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=LEARNING_RATE)

    statistics_count = 6750 / (CHROMREG_MINIBATCH_SIZE * 5)
    for epoch in range(EPOCH_COUNT):  # loop over the dataset multiple times
        net.train()
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, expected = data

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, expected)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % statistics_count == (statistics_count - 1):    # print 5 times per epoch
                print('[%d, %5d] loss: %.5f' %
                    (epoch + 1, i + 1, running_loss / statistics_count))
                running_loss = 0.0
    
    return net


class ColorizationNet(nn.Module):
    def __init__(self):
        super(ColorizationNet, self).__init__()
        
        self.upsample = nn.Sequential(    
            #convtranpse2d inverse of conv2d, upsample inverse of pooling
            nn.Conv2d(1, 32, 3, stride = 1, padding = 1, bias = True),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, 3, stride = 1, padding = 1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, 3, stride = 1, padding = 1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(128, 256, 3, stride = 1, padding = 1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(256, 512, 3, stride = 1, padding = 1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.ConvTranspose2d(512, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Upsample(scale_factor=2),
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Upsample(scale_factor=2),
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Upsample(scale_factor=2),
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Upsample(scale_factor=2),
            nn.ConvTranspose2d(32, 2, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Upsample(scale_factor=2)
        )
    def forward(self, input):
        # Upsample to get colors
        output = self.upsample(input)
        return output


def train_colorizer(trainloader, device):
    EPOCH_COUNT = 32
    LEARNING_RATE = 0.003

    net = ColorizationNet()

    if torch.cuda.device_count() > 1:
        net = nn.DataParallel(net)

    net.to(device)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=LEARNING_RATE)

    statistics_count = int(6750 / (COLORIZE_MINIBATCH_SIZE * 6))
    for epoch in range(EPOCH_COUNT):  # loop over the dataset multiple times
        net.train()
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, expected = data

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, expected)
            loss.backward()
            optimizer.step()
            
            # print statistics
            running_loss += loss.item()
            if i % statistics_count == (statistics_count - 1):    # print 5 times per epoch
                print('[%d, %5d] loss: %.5f' %
                    (epoch + 1, i + 1, running_loss / statistics_count))
                running_loss = 0
                
    return net

def write_img(name, l, ab, expected):
    if torch.cuda.is_available():
        l = l.cpu()
        ab = ab.cpu()
        expected = expected.cpu()
    color_image = torch.cat((l*255, ab), 0).numpy()
    color_image = color_image.transpose((1, 2, 0))
    color_image = color_image.round()
    color_image = np.clip(color_image,0,255)
    color_image = np.uint8(color_image)
    color_image = cv2.cvtColor(color_image, cv2.COLOR_LAB2BGR)

    original = torch.cat((l*255, expected), 0).numpy()
    original = np.transpose(original,(1,2,0))
    original = np.uint8(original)
    original = cv2.cvtColor(original,  cv2.COLOR_LAB2BGR)
    
    gray_image = torch.cat((l*255, l*255, l*255), 0).numpy()
    gray_image = gray_image.transpose((1, 2, 0))
    gray_image = gray_image.round()
    gray_image = np.clip(gray_image,0,255)
    gray_image = np.uint8(gray_image)

    combined_image = np.concatenate((original, gray_image, color_image), 1)
    cv2.imwrite(name,combined_image)