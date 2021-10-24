import torch
import cv2
import glob
import numpy as np
import random
import torch.nn as nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import models


#reduces memory requirements (32-bit float)
torch.set_default_tensor_type(torch.FloatTensor)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    print("Using GPU")
else:
    print("Using CPU")

# defining the Dataset class
class FaceImages(Dataset):
    def __init__(self,train=True, augmentations=None, is_regressor=True):
        self.train = train
        self.is_regressor = is_regressor

        # retreive all face image files
        img_dir = "face_images/*.jpg"
        files = glob.glob(img_dir)
        data = []
        #eandomize order of images each time dataset is loaded
        rand_indeces = torch.randperm(len(files))
        data = [cv2.imread(files[i]) for i in rand_indeces]

        #format data into a tensor of shape (750, 128, 128, 3)
        data_stack = np.stack(data, axis=0)

        faces_tensor = torch.tensor(data_stack)


        #split tensor into training and testing. 90% is training (675 training images, 75 testing images)
        if self.train:
            self.data = faces_tensor[:round(0.9 * len(faces_tensor))]
            if augmentations != None:
                self.data = self.augment(augmentations)
            self.data = self.convert_images(self.data)
        else:
            self.data = faces_tensor[round(0.9 * len(faces_tensor)):]
            self.data = self.convert_images(self.data)
  
    def __len__(self):
        return len(self.data)
  
    def __getitem__(self, index):
        #Prepare converted dataset for training
        img = self.data[index].numpy()
        img = np.transpose(img,(2,0,1))
        
        input_vals = img[0:1,:,:] / 255
        a_vals = img[1,:,:]
        b_vals = img[2,:,:]
        
        if self.is_regressor:
            a_avg = a_vals.mean()
            b_avg = b_vals.mean()
            output_arr = np.array([a_avg,b_avg])
        else:
            output_arr = np.array([a_vals,b_vals])

        #if cuda is avail, use it
        input_tens = torch.tensor(input_vals).float().to(device)
        output_tens = torch.tensor(output_arr).float().to(device)

        return input_tens, output_tens

    def augment(self, augmentations):
        #iterate through provided tensor and apply the above augmentations yielding a tensor 10x as large as the input
        faces_arr = self.data.numpy()
        new_shape = list(faces_arr.shape)
        new_shape[0] *= 10
        augmented_arr = np.zeros(new_shape, int)
        for i,img in enumerate(faces_arr):
            for j, augment in enumerate(augmentations):
                augmented_arr[i*10 + j] = augment(img)
        
        augmented_arr = augmented_arr.astype(np.uint8)

        return torch.tensor(augmented_arr)


    #Image Conversion to L* A* B* Color Space
    def convert_images(self, faces_tensor):
        converted_tensor = torch.zeros_like(faces_tensor)
        for i,img in enumerate(faces_tensor):
            img = img.numpy()
            img = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
            converted_tensor[i] = torch.tensor(img)

        return converted_tensor


#Augmentations 
def random_crop(img, scaling_range):
    # scaling_range = scale factor indicating percentile range of each side that will remain
    #crop image in random position according to selected percentile and rescale to original size
    new_size = round(random.uniform(*scaling_range) * img.shape[0])
    new_x = random.randint(0, img.shape[0] - new_size - 1)
    new_y = random.randint(0, img.shape[1] - new_size - 1)

    cropped_img = img[new_x : new_x + new_size, new_y : new_y + new_size, : ]

    resized_img = cv2.resize(cropped_img, img.shape[:2])

    return resized_img

def flip_horiz(img):
    return np.flip(img, 1)

def scale_rgb(img):
    scale_val = random.uniform(0.6,0.99)
    return np.multiply(img, scale_val).round()

augmentations = [
    lambda x: x,
    lambda x: random_crop(x, (0.65, 0.9)),
    lambda x: flip_horiz(x),
    lambda x: scale_rgb(x),
    lambda x: flip_horiz(random_crop(x, (0.65, 0.9))),
    lambda x: flip_horiz(scale_rgb(x)),
    lambda x: scale_rgb(random_crop(x, (0.65, 0.9))),
    lambda x: flip_horiz(random_crop(scale_rgb(x), (0.65, 0.9))),
    
    lambda x: flip_horiz(random_crop(x, (0.3, 0.7))),
    lambda x: scale_rgb(random_crop(x, (0.3, 0.7))),
]


def chrominance_regressor_main():
    print(" --- Loading Data ---")
    # implementing dataloader on the dataset and printing per batch
    trainset = FaceImages(train=True,augmentations=augmentations)
    trainloader = DataLoader(trainset, batch_size=models.CHROMREG_MINIBATCH_SIZE, shuffle=True)

    print(" --- Finished Loading Data, Training model ---")
    model = models.train_chrominance_reg(trainloader, device)
    print(" --- Finished Training ---")
    

    model.eval()

    testset = FaceImages(train=False)
    testloader = DataLoader(testset, batch_size=models.CHROMREG_MINIBATCH_SIZE, shuffle=False)
    
    total_loss = 0
    count = 0
    for i,data in enumerate(testloader):
        images, outputs = data

        #compare expected and actual results of CNN performance on test data
        results = model(images)
        results = np.multiply(results.detach().numpy(), 255)
        outputs = np.multiply(outputs.numpy(), 255)
        for i,val in enumerate(zip(outputs,results)):
            print("Test Image #{}:\tExpected:{}\tActual:{}".format(i+1, val[0], val[1]))
        criterion = nn.MSELoss()
        loss = criterion(torch.tensor(outputs),torch.tensor(results))
        print(" --- Test Batch #%d Loss: %f ---" % (i+1, loss))
        total_loss += loss
        count += 1

    print("\n --- Average MSE Loss: %f" % (total_loss / count))

    
def colorization_main(model_path=""):
    
    print(" --- Loading Data ---")
    # implementing dataloader on the dataset and printing per batch
    trainset = FaceImages(train=True, augmentations=augmentations, is_regressor=False)
    trainloader = DataLoader(trainset, batch_size=models.COLORIZE_MINIBATCH_SIZE, shuffle=True)

    print(" --- Finished Loading Data, Training model ---")
    model = models.train_colorizer(trainloader, device)
    print(" --- Finished Training ---")
    

    model.eval()

    testset = FaceImages(train=False, is_regressor=False)
    testloader = DataLoader(testset, batch_size=models.COLORIZE_MINIBATCH_SIZE, shuffle=False)
    
    total_loss = 0
    count = 0
    for i,data in enumerate(testloader):
        inputs, expected = data

        #compare expected and actual results of CNN performance on test data
        results = model(inputs)
        results = results.detach()

        for j,val in enumerate(zip(inputs, results, expected)):
            fname = "test_image_results/img{0:03d}.png".format(i * models.COLORIZE_MINIBATCH_SIZE + j)
            models.write_img(fname, val[0] , val[1], val[2])
            print(" --- Wrote image to '{}'".format(fname))

        criterion = nn.MSELoss()
        loss = criterion(expected,results)
        print(" --- Test Batch #%d Loss: %f ---" % (i+1, loss))
        total_loss += loss
        count += 1

    print("\n --- Average MSE Loss: %f" % (total_loss / count))

if __name__ == '__main__':
    regressor = 1

    if regressor == 1:
        chrominance_regressor_main()
    else:
        colorization_main()