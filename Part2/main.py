import torch
import cv2
import glob
import numpy as np
import random
import torch.nn as nn
import models

#reduces memory requirements (32-bit float)
torch.set_default_tensor_type(torch.FloatTensor)

def load_dataset():
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
    train_tensor = faces_tensor[:round(0.9 * len(faces_tensor))]
    test_tensor = faces_tensor[round(0.9 * len(faces_tensor)):]

    return (train_tensor,test_tensor)

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

#Augment dataset
def augment(faces_tensor):
    #iterate through provided tensor and apply the above augmentations yielding a tensor 10x as large as the input
    augmented_arr = np.zeros((7500,128,128,3), int)
    faces_arr = faces_tensor.numpy()
    for i,img in enumerate(faces_arr):
        for j, augment in enumerate(augmentations):
            augmented_arr[i*10 + j] = augment(img)
    
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
    # transpose array to expected format (7500, 3, 128, 128)
    trans_arr = np.transpose(np_tens,(0,3,1,2))
    #select the luminence values and normalize between [0,1]
    input_vals = trans_arr[:,0:1,:,:] / 100

    # select the A* and B* values then calculate the mean for expected values
    a_vals = trans_arr[:,1,:,:]
    b_vals = trans_arr[:,2,:,:]

    a_avg = a_vals.mean(axis=tuple(range(1, a_vals.ndim)))
    b_avg = b_vals.mean(axis=tuple(range(1, b_vals.ndim)))

    #construct and return input and output tensors for training and testing
    output_arr = np.array(list(zip(a_avg,b_avg)))

    input_tens = torch.tensor(input_vals).float()
    output_tens = torch.tensor(output_arr).float()
    return (input_tens,output_tens)

#split datasets into minibatches of size count
def make_batches(input_tens, output_tens, count):
    input_batches = torch.split(input_tens, count)
    output_batches = torch.split(output_tens, count)

    return tuple(zip(input_batches,output_batches))

#display an inputted image
def show_image(img):
    if not isinstance(img, np.ndarray):
        new_img = img.numpy()
    else:
        new_img = img
    cv2.imshow("Part 2 Image", new_img)
    cv2.waitKey(0)


def main():
    # user input to select a saved model or specify if training a new model
    model_names = glob.glob("*.model")
    if len(model_names) == 0:
        train = True
    else:
        resp = ""
        while resp != "y" and resp != "n":
            resp = input("Train new model? (y/n): ")
        train = resp == 'y'
    if not train:
        for i, name in enumerate(model_names):
            print("{}. {}".format(i+1, name))
        resp = 0
        while resp < 1 or resp > len(model_names):
            resp_str = input("Enter which numbered model to evaluate: ")
            if resp_str.isnumeric():
                resp = int(resp_str)
        model_path = model_names[resp - 1]


    train_data, test_data = load_dataset()
    print(" --- Data Loaded ---")
    
    if train:
        augmented_tensor = augment(train_data)
        print(" --- Data Augmented ---")
        
        converted_tensor = convert_images(augmented_tensor)
        print(" --- Data Converted to LAB ---")

        prepared_tuple = prepare_data(converted_tensor)
        training_batches = make_batches(*prepared_tuple, models.MINIBATCH_SIZE)
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
        print("Test Image #{}:\tExpected:{}\tActual:{}".format(i+1, val[0].detach().numpy(), val[1].detach().numpy()))
    criterion = nn.MSELoss()
    print(" --- Test Loss: %f ---" % (criterion(results,output_tens)))

if __name__ == '__main__':
    main()