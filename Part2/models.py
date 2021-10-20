import torch
import torch.nn as nn
import torch.nn.functional as F
from datetime import datetime

CHROMREG_MINIBATCH_SIZE = 5

#TODO: make this regressor work well. currently suffering from high loss and for some reason the two outputs are almost always basically the same number
# REF: https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html
class Chrominance_Regressor(nn.Module):
    def __init__(self):
        super().__init__()
        # implementation of convolution downsampling and FC layers for regression output
        self.conv1 = nn.Conv2d(1, 64, 5, stride = 1, padding = 2)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(64, 64, 5, stride = 1, padding = 2)
        self.conv3 = nn.Conv2d(64, 32, 3, stride = 1, padding = 1)
        self.conv4 = nn.Conv2d(32, 16, 3, stride = 1, padding = 1)
        self.conv5 = nn.Conv2d(16, 8, 3, stride = 1, padding = 1)
        self.conv6 = nn.Conv2d(8, 8, 3, stride = 1, padding = 1)
        self.fc1 = nn.Linear(4 * 4 * 8, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 32)
        self.fc4 = nn.Linear(32, 32)
        self.fc5 = nn.Linear(32, 16)
        self.fc6 = nn.Linear(16, 4)
        self.fc7 = nn.Linear(4, 2)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = self.pool(F.relu(self.conv4(x)))
        x = self.pool(F.relu(self.conv5(x)))
        x = F.relu(self.conv6(x))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = F.relu(self.fc5(x))
        x = F.relu(self.fc6(x))
        x = self.fc7(x)
        return x

#Regressor
def train_chrominance_reg(trainloader):
    EPOCH_COUNT = 16
    LEARNING_RATE = 0.0004

    net = Chrominance_Regressor()

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=LEARNING_RATE)

    statistics_count = 7500 / (CHROMREG_MINIBATCH_SIZE * 5)
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
                print('[%d, %5d] loss: %.3f' %
                    (epoch + 1, i + 1, running_loss / statistics_count))
                running_loss = 0.0
    
    #save the model with time based name to prevent model overwrite
    now = datetime.now()
    model_path = now.strftime("part2_%m-%d_%H.%M.regmodel")
    torch.save(net.state_dict(), model_path)
    return net


# TODO: actually make this model and training algorithm - currently basically just a copy of the regressor
class ColorizationNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        # implementation of convolution downsampling and FC layers for regression output
        self.conv1 = nn.Conv2d(1, 32, 5, stride = 1, padding = 2)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 32, 3, stride = 1, padding = 1)
        self.conv3 = nn.Conv2d(32, 16, 3, stride = 1, padding = 1)
        self.conv4 = nn.Conv2d(16, 8, 3, stride = 1, padding = 1)
        self.conv5 = nn.Conv2d(8, 4, 3, stride = 1, padding = 1)
        self.fc1 = nn.Linear(4 * 4 * 4, 32)
        self.fc2 = nn.Linear(32, 32)
        self.fc3 = nn.Linear(32, 16)
        self.fc4 = nn.Linear(16, 12)
        self.fc5 = nn.Linear(12, 2)

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
        x = self.fc5(x)
        return x

def train_colorizer(trainloader):
    EPOCH_COUNT = 16
    LEARNING_RATE = 0.005

    net = Chrominance_Regressor()

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=LEARNING_RATE)

    statistics_count = 7500 / (CHROMREG_MINIBATCH_SIZE * 5)
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
                print('[%d, %5d] loss: %.3f' %
                    (epoch + 1, i + 1, running_loss / statistics_count))
                running_loss = 0.0
    
    #save the model with time based name to prevent model overwrite
    now = datetime.now()
    model_path = now.strftime("part2_%m-%d_%H.%M.colormodel")
    torch.save(net.state_dict(), model_path)
    return net



# import torch.nn.functional as F

# class Conv(nn.Module):
#     def __init__(self):
#             super(Conv, self).init()
#             self.conv1 =  nn.Sequential(nn.Conv2d(1, 2, (128,128),stride = 2, padding = 1) , nn.Relu(),
#                                          nn.Conv2d(1, 2, (64,64),stride = 2, padding = 1) , nn.Relu(),
#                                          nn.Conv2d(1, 2, (32,32),stride = 2, padding = 1) , nn.Relu(),
#                                          nn.Conv2d(1, 2, (16,16),stride = 2, padding = 1 ), nn.Relu(),
#                                          nn.Conv2d(1, 2, (8,8),stride = 2, padding = 1) , nn.Relu(),
#                                          nn.Conv2d(1, 2, (4,4),stride = 2, padding = 1 ), nn.Relu(),
#                                          nn.Conv2d(1, 2, (2,2),stride = 2, padding = 1) , nn.Relu())

#     #add regressor to prdict a and b?
#     #nn.Linear(1, 2)
#     def forward(self, x):
#         x = self.conv1(x)
#         return x

# from keras.models import Sequential
# from keras.layers import Dense, Conv2D, Flatten
# #create model
# model = Sequential()
# #add model layers
# model.add(Conv2D(64, kernel_size=3, activation='relu', input_shape=(128,128,1), data_format="channels_last"))
# model.add(Conv2D(32, kernel_size=3, activation='relu'))
# model.add(Flatten())
# model.add(Dense(10, activation='linear'))


# class ColorizedImageNN(nn.Module):
#     def __init__(self):
#         super(ColorizedImageNN,self).__intit__()
#         self.upsample = nn.Sequential(
#             nn.Conv2d(128,128,kernel_size=5,stride=1,padding=1),
#             nn.ReLU(),
#             nn.UpsamplingNearest2d(scale_factor=2),
#             nn.Conv2d(128,64,kernel_size=3,stride=1,padding=1),
#             nn.ReLU(),
#             nn.Conv2d(64,64,kernel_size=3,stride=1,padding=1),
#             nn.ReLU(),
#             nn.UpsamplingNearest2d(scale_factor=2),
#             nn.Conv2d(64,32,kernel_size=3,stride=1,padding=1),
#             nn.ReLU(),
#             nn.Conv2d(32,32,kernel_size=3,stride=1,padding=1),
#             nn.ReLU(),
#             nn.UpsamplingNearest2d(scale_factor=2),
#             nn.Conv2d(32,16,kernel_size=3,stride=1,padding=1),
#             nn.ReLU(),
#             nn.Conv2d(16,16,kernel_size=3,stride=1,padding=1),
#             nn.ReLU(),
#             nn.UpsamplingNearest2d(scale_factor=2),
#             nn.Conv2d(16,8,kernel_size=3,stride=1,padding=1),
#             nn.ReLU(),
#             nn.Conv2d(8,8,kernel_size=3,stride=1,padding=1),
#             nn.ReLU(),
#             nn.UpsamplingNearest2d(scale_factor=2),
#             nn.Conv2d(8,4,kernel_size=3,stride=1,padding=1),
#             nn.UpsamplingNearest2d(scale_factor=2)
#         )

# def chrominance_regressor(dataset):
#     #Scale L channel to [0,1] either 0 or 1?
#     for i in dataset:
#         dataset[i] = dataset[i]//100
#     criterion = nn.MSECriterion()

#     #7 modules, each with a spatial convolution layer + Relu activation function, sum all modules?
#     NN = Conv()

#     result = Conv(dataset)