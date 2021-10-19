from albumentations.augmentations.functional import transpose
import torch
import cv2
import glob
import numpy as np
import random
import torch.nn as nn
import albumentations as A
import models
import torch.optim as optim

#reduces memory requirements (32-bit float)
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
    train_tensor = faces_tensor[:round(0.9 * len(faces_tensor))]
    test_tensor = faces_tensor[round(0.9 * len(faces_tensor)):]

    return (train_tensor,test_tensor)

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

#Prepare converted dataset for training
def prepare_data(converted_tensor):
    np_tens = converted_tensor.numpy()
    trans_arr = np.transpose(np_tens,(0,3,1,2))
    input_vals = trans_arr[:,0:1,:,:] / 100

    a_vals = trans_arr[:,1,:,:]
    b_vals = trans_arr[:,2,:,:]

    a_avg = a_vals.mean(axis=tuple(range(1, a_vals.ndim)))
    b_avg = b_vals.mean(axis=tuple(range(1, b_vals.ndim)))

    output_arr = np.array(list(zip(a_avg,b_avg)))

    input_tens = torch.tensor(input_vals).float()
    output_tens = torch.tensor(output_arr).float()
    return (input_tens,output_tens)

def make_batches(input_tens, output_tens, count):
    input_batches = torch.split(input_tens, count)
    output_batches = torch.split(output_tens, count)

    return tuple(zip(input_batches,output_batches))

#Regressor
def train_chrominance_reg(trainloader):
    net = models.Chrominance_Regressor()

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=0.01)

    for epoch in range(8):  # loop over the dataset multiple times
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
            if i % 135 == 134:    # print every 134 mini-batches
                print('[%d, %5d] loss: %.3f' %
                    (epoch + 1, i + 1, running_loss / 135))
                running_loss = 0.0
    
    torch.save(net.state_dict(), "model")
    return net

    
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
    TRAIN = True

    train_data, test_data = load_dataset()
    print(" --- Data Loaded ---")
    
    if TRAIN:
        augmented_tensor = augment(train_data)
        print(" --- Data Augmented ---")
        
        converted_tensor = convert_images(augmented_tensor)
        print(" --- Data Converted to LAB ---")

        prepared_tuple = prepare_data(converted_tensor)
        training_batches = make_batches(*prepared_tuple, 10)
        print(" --- Data Prepared for Training ---")
        model = train_chrominance_reg(training_batches)
        print(" --- Finished Training ---")
    
    else:
        model = models.Chrominance_Regressor()
        model.load_state_dict(torch.load("model"))
        model.eval()

    lab_data = convert_images(test_data)
    input_tens, output_tens = prepare_data(lab_data)

    results = model(input_tens)
    for i,val in enumerate(zip(output_tens,results)):
        print("Test Image #{}:\tExpected:{}\tActual:{}".format(i+1, val[0].detach().numpy(), val[1].detach().numpy()))
    criterion = nn.MSELoss()
    print(" --- Loss: %f ---" % (criterion(results,output_tens)))