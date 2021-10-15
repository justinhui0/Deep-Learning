import torch
import cv2
import os
import glob
import numpy as np
import random

# Reduces memory requirements (32-bit float)
torch.set_default_tensor_type(torch.FloatTensor)

def load_dataset():
    img_dir = "face_images/*.jpg"
    files = glob.glob(img_dir)
    data = []
    #preprocess data
    for f1 in files:
        img = cv2.imread(f1)
        img = np.transpose(img, (2,0,1))
        data.append(img)
    faces_tensor = torch.empty((750,3,128,128))

    #randomly shuffle data
    rand_indeces = torch.randperm(len(data))
    for i,rand_i in enumerate(rand_indeces):
        faces_tensor[i] = torch.from_numpy(data[rand_i])

    return faces_tensor

#Augmentations 

def aug_crop(img,scale):
    #Ref: https://towardsdatascience.com/data-augmentation-compilation-with-python-and-opencv-b76b1cd500e0
    height, width = int(img.shape[2]*scale), int(img.shape[1]*scale)
    dimensions = img.shape[1:]
    # Get random range for crop
    x = random.randint(0, img.shape[1] - int(width))
    y = random.randint(0, img.shape[2] - int(height))
    # Crop image
    img = np.transpose(img, (1,2,0))
    croppedImg = img[y:y + height, x:x + width]
    # Convert to numpy
    np_img = croppedImg.numpy()
    # Resize image
    resizedImg = cv2.resize(np_img, dimensions, interpolation = cv2.INTER_AREA)
    # Convert to tensor image
    resizedImg = np.transpose(resizedImg, (2,0,1))
    tensor_image = torch.from_numpy(resizedImg)
    return tensor_image


def aug_flip_horiz(img):
    flipped_img = cv2.flip(img.numpy(), 1)
    torch_flipped = torch.from_numpy(flipped_img)
    return torch_flipped

def aug_flip_vert(img):
    flipped_img = cv2.flip(img.numpy(), 0)
    torch_flipped = torch.from_numpy(flipped_img)
    return torch_flipped
    
def aug_scale(img, scalar):
    return (img * scalar).round()


#Augment dataset
def augment(faces_tensor):
    ten_tensor = torch.empty((7500,3,128,128))
    scale = 0.5
    for i in range(750):
        #add in original faces tensor to ten tensor
        ten_tensor[i*10] = faces_tensor[i]
        #add augmented to tensor
        ten_tensor[i*10+1] = aug_scale(faces_tensor[i], random.uniform(.6,1))
        ten_tensor[i*10+2] = aug_crop(faces_tensor[i], scale)
        ten_tensor[i*10+3] = aug_flip_horiz(faces_tensor[i])
        ten_tensor[i*10+4] = aug_flip_vert(faces_tensor[i])
        ten_tensor[i*10+5] = aug_crop(aug_flip_horiz(faces_tensor[i]),scale )
        ten_tensor[i*10+6] = aug_crop(aug_flip_vert(faces_tensor[i]),scale)
        ten_tensor[i*10+7] = aug_crop(aug_scale(faces_tensor[i], random.uniform(.6,1)), scale)
        ten_tensor[i*10+8] = aug_scale(aug_flip_horiz(faces_tensor[i]), random.uniform(.6,1))
        ten_tensor[i*10+9] = aug_scale(aug_flip_vert(faces_tensor[i]), random.uniform(.6,1))
    return ten_tensor


#Image Conversion to Color Space
def convert_images(faces_tensor):
    converted_tensor = torch.zeros_like(faces_tensor)
    for i,img in enumerate(faces_tensor):
        img = img.numpy()
        img = np.transpose(img, (1,2,0))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        img = np.transpose(img, (2,0,1))
        converted_tensor[i] = torch.from_numpy(img)
        
    return converted_tensor


#Regressor
def chrominance_regressor(dataset):
    pass

if __name__ == '__main__':
    dataset = load_dataset()
    augmented_tensor = augment(dataset)
    converted_tensor = convert_images(augmented_tensor)
    print(dataset)
    print(augmented_tensor)
    print(converted_tensor)
