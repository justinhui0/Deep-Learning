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

# defining the Dataset class
class FaceImages(Dataset):
    def __init__(self,train=True, augmentations=None):
        self.train = train

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


        a_vals = img[1,:,:] / 255
        b_vals = img[2,:,:] / 255

        a_avg = a_vals.mean()
        b_avg = b_vals.mean()

        #construct and return input and output tensors for training and testing
        output_arr = np.array([a_avg,b_avg])

        input_tens = torch.tensor(input_vals).float()
        output_tens = torch.tensor(output_arr).float()

        return input_tens, output_tens

    def augment(self, augmentations):
        #iterate through provided tensor and apply the above augmentations yielding a tensor 10x as large as the input
        augmented_arr = np.zeros((7500,128,128,3), int)
        faces_arr = self.data.numpy()
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
def random_crop(img):
    # scale factor indicating percentile range of each side that will remain
    scaling_range = (0.45, 0.95)

    #crop image in random position according to selected percentile and rescale to original size
    new_size = round(random.uniform(*scaling_range) * img.shape[0])
    new_x = random.randint(0, img.shape[0] - new_size - 1)
    new_y = random.randint(0, img.shape[1] - new_size - 1)

    cropped_img = img[new_x : new_x + new_size, new_y : new_y + new_size, : ]

    resized_img = cv2.resize(cropped_img, img.shape[:2])

    return resized_img

def flip_vert(img):
    return np.flip(img, 0)

def flip_horiz(img):
    return np.flip(img, 1)

def scale_rgb(img):
    scale_val = random.uniform(0.6,1)
    return np.multiply(img, scale_val).round()

augmentations = [
    lambda x: x,
    lambda x: random_crop(x),
    lambda x: scale_rgb(flip_vert(x)),
    lambda x: scale_rgb(flip_horiz(x)),
    lambda x: scale_rgb(x),
    lambda x: flip_vert(random_crop(x)),
    lambda x: flip_horiz(random_crop(x)),
    lambda x: random_crop(flip_vert(flip_horiz(x))),
    lambda x: scale_rgb(flip_vert(flip_horiz(x))),
    lambda x: scale_rgb(random_crop(flip_vert(flip_horiz(x))))
]

#display an inputted image
def show_image(img):
    if not isinstance(img, np.ndarray):
        new_img = img.numpy()
    else:
        new_img = img
    cv2.imshow("Part 2 Image", new_img)
    cv2.waitKey(0)


def get_model_path(extension):
    # user input to select a saved model or specify if training a new model
    model_names = glob.glob("*." + extension)
    if len(model_names) == 0:
        train = True
    else:
        resp = ""
        while resp != "y" and resp != "n":
            resp = input("Train new model? (y/n): ")
        train = resp == 'y'
    if not train:
        print("Model Options:")
        for i, name in enumerate(model_names):
            print("\t{}. {}".format(i+1, name))
        resp = 0
        while resp < 1 or resp > len(model_names):
            resp_str = input("Enter which numbered model to evaluate: ")
            if resp_str.isnumeric():
                resp = int(resp_str)
        model_path = model_names[resp - 1]
    if train:
        model_path = None
    return model_path

def chrominance_regressor_main(model_path=""):
    if model_path == "":
        model_path = get_model_path("regmodel")

    dataset = FaceImages(train=True,augmentations=augmentations)


    # implementing dataloader on the dataset and printing per batch
    dataloader = DataLoader(dataset, batch_size=models.CHROMREG_MINIBATCH_SIZE, shuffle=True)


    model = models.train_chrominance_reg(dataloader)
    print(" --- Finished Training ---")
    
    # else:
    #     #load selected model
    #     model = models.Chrominance_Regressor()
    #     model.load_state_dict(torch.load(model_path))
    #     model.eval()

    test_dataset = FaceImages(train=False)

    #compare expected and actual results of CNN performance on test data
    results = model(test_dataset)
    for i,val in enumerate(zip(output_tens,results)):
        print("Test Image #{}:\tExpected:{}\tActual:{}".format(i+1, val[0].detach().numpy().round(4), val[1].detach().numpy().round(4)))
    criterion = nn.MSELoss()
    print(" --- Test Loss: %f ---" % (criterion(results,output_tens)))

def colorization_main(model_path=""):
    if model_path == "":
        model_path = get_model_path("colormodel")
    
    train_data, test_data = load_dataset()
    print(" --- Data Loaded ---")
    
    if model_path is None:
        augmented_tensor = augment(train_data)
        print(" --- Data Augmented ---")
        
        converted_tensor = convert_images(augmented_tensor)
        print(" --- Data Converted to LAB ---")

        prepared_tuple = prepare_data(converted_tensor)
        training_batches = make_batches(*prepared_tuple, models.CHROMREG_MINIBATCH_SIZE)
        print(" --- Data Prepared for Training ---")
        
        model = models.train_chrominance_reg(training_batches)
        print(" --- Finished Training ---")
    
    else:
        #load selected model
        model = models.Chrominance_Regressor()
        model.load_state_dict(torch.load(model_path))
        model.eval()

    #convert and prepare test data
    lab_data = convert_images(test_data)
    input_tens, output_tens = prepare_data(lab_data)

    #compare expected and actual results of CNN performance on test data
    results = model(input_tens)
    for i,val in enumerate(zip(output_tens,results)):
        print("Test Image #{}:\tExpected:{}\tActual:{}".format(i+1, val[0].detach().numpy().round(4), val[1].detach().numpy().round(4)))
    criterion = nn.MSELoss()
    print(" --- Test Loss: %f ---" % (criterion(results,output_tens)))

if __name__ == '__main__':
    regressor = 1
    
    if regressor == 1:
        chrominance_regressor_main()
    else:
        colorization_main()