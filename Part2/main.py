from albumentations import augmentations
import torch
import cv2
import glob
import numpy as np
import random
import torch.nn as nn
import albumentations as A
import torch.nn.functional as F

class Conv(nn.Module):
    def __init__(self):
            super(Conv, self).init()
            self.conv1 =  nn.Sequential(nn.Conv2d(1, 2, (128,128),stride = 2, padding = 1) , nn.Relu(),
                                         nn.Conv2d(1, 2, (64,64),stride = 2, padding = 1) , nn.Relu(),
                                         nn.Conv2d(1, 2, (32,32),stride = 2, padding = 1) , nn.Relu(),
                                         nn.Conv2d(1, 2, (16,16),stride = 2, padding = 1 ), nn.Relu(),
                                         nn.Conv2d(1, 2, (8,8),stride = 2, padding = 1) , nn.Relu(),
                                         nn.Conv2d(1, 2, (4,4),stride = 2, padding = 1 ), nn.Relu(),
                                         nn.Conv2d(1, 2, (2,2),stride = 2, padding = 1) , nn.Relu())

    #add regressor to prdict a and b?
    #nn.Linear(1, 2)
    def forward(self, x):
        x = self.conv1(x)
        return x


class ColorizedImageNN(nn.Module):
    def __init__(self):
        super(ColorizedImageNN,self).__intit__()
        self.upsample = nn.Sequential(
            nn.Conv2d(128,128,kernel_size=5,stride=1,padding=1),
            nn.ReLU(),
            nn.UpsamplingNearest2d(scale_factor=2),
            nn.Conv2d(128,64,kernel_size=3,stride=1,padding=1),
            nn.ReLU(),
            nn.Conv2d(64,64,kernel_size=3,stride=1,padding=1),
            nn.ReLU(),
            nn.UpsamplingNearest2d(scale_factor=2),
            nn.Conv2d(64,32,kernel_size=3,stride=1,padding=1),
            nn.ReLU(),
            nn.Conv2d(32,32,kernel_size=3,stride=1,padding=1),
            nn.ReLU(),
            nn.UpsamplingNearest2d(scale_factor=2),
            nn.Conv2d(32,16,kernel_size=3,stride=1,padding=1),
            nn.ReLU(),
            nn.Conv2d(16,16,kernel_size=3,stride=1,padding=1),
            nn.ReLU(),
            nn.UpsamplingNearest2d(scale_factor=2),
            nn.Conv2d(16,8,kernel_size=3,stride=1,padding=1),
            nn.ReLU(),
            nn.Conv2d(8,8,kernel_size=3,stride=1,padding=1),
            nn.ReLU(),
            nn.UpsamplingNearest2d(scale_factor=2),
            nn.Conv2d(8,4,kernel_size=3,stride=1,padding=1),
            nn.UpsamplingNearest2d(scale_factor=2)
        )
    #duces memory requirements (32-bit float)

torch.set_default_tensor_type(torch.FloatTensor)

def load_dataset():
    img_dir = "face_images/*.jpg"
    files = glob.glob(img_dir)
    data = []
    #preprocess data
    rand_indeces = torch.randperm(len(files))
    data = [cv2.imread(files[i]) for i in rand_indeces]

    data_stack = np.stack(data, axis=0)
    faces_tensor = torch.tensor(data_stack)

    return faces_tensor

#Augmentations 

transform = A.Compose([
    A.RandomSizedCrop(min_max_height=(50,101),width=128, height=128,p=0.5),
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5)
])

#Augment dataset
def augment(faces_tensor):
    augmented_arr = np.zeros((7500,128,128,3), int)
    for i,orig_img in enumerate(faces_tensor):
        img = orig_img.numpy()
        augmented_arr[i*10] = img
        for j in range(9):
            augmented_arr[i*10 + j + 1] = transform(image=img)["image"]
            if random.randint(1, 10) <= 5:
                augmented_arr[i*10 + j + 1] = np.multiply(augmented_arr[i*10 + j + 1], random.uniform(0.6,1)).round()
    augmented_arr = augmented_arr.astype(np.uint8)
    augmented_tensor = torch.tensor(augmented_arr)

    return augmented_tensor


#Image Conversion to Color Space
def convert_images(faces_tensor):
    converted_tensor = torch.zeros_like(faces_tensor)
    for i,img in enumerate(faces_tensor):
        img = img.numpy()
        img = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        converted_tensor[i] = torch.tensor(img)
        
    return converted_tensor


#Regressor
def chrominance_regressor(dataset):
    #Scale L channel to [0,1] either 0 or 1?
    for i in dataset:
        dataset[i] = dataset[i]//100
    criterion = nn.MSECriterion()

    #7 modules, each with a spatial convolution layer + Relu activation function, sum all modules?
    NN = Conv()

    result = Conv(dataset)

    
#Colorize the Image


def colorize():
    pass
    # 2 color channels a* b* for output
    
    
#Batch Normalization
def batchNorm():
    #module = nn.SpatialBatchNormalization(N [,eps] [, momentum] [,affine])
    module = nn.SpatialBatchNormalization

def show_image(name, img):
    new_img = img.numpy()
    cv2.imshow(name, new_img)
    cv2.waitKey(0)

if __name__ == '__main__':
    dataset = load_dataset()
    print(dataset)
    show_image('Orig Dataset', dataset[0])

    augmented_tensor = augment(dataset)
    print(augmented_tensor)
    show_image('Augmented Dataset', augmented_tensor[3])
    
    converted_tensor = convert_images(augmented_tensor)
    print(converted_tensor)
    show_image('Converted Dataset', converted_tensor[3])

