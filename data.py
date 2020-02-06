from torch.utils.data import Dataset
import torch
import pandas as pd
from skimage.io import imread
from skimage.color import gray2rgb
import numpy as np
from sklearn.model_selection import train_test_split
import torchvision as tv
import os.path

train_mean = [0.59685254, 0.59685254, 0.59685254]
train_std = [0.16043035, 0.16043035, 0.16043035]
path_to_csv = r'D:\\courses\\Deep Learning\\Exercises\\Exercise4\\src_to_implement\\train.csv'


class ChallengeDataset(Dataset):
    # TODO implement the Dataset class according to the description
    def __init__(self, flag, csv_path, train_size, h=tv.transforms.Compose([tv.transforms.ToTensor()])):
        self.mode = flag
        self.labels = pd.read_csv(csv_path, sep=",")
        self.train_size = train_size
        self.transform = h
        self.train_data, self.valid_data = train_test_split(self.labels, train_size=self.train_size, random_state=11)

    def __getitem__(self, item):
        #Depending on the mode, return train / val data
        if self.mode=="train":
            image_name = self.train_data.iloc[item,0].split(";")[0]
            image_label = self.train_data.iloc[item, 0].split(";")[2:]
        else:
            image_name = self.valid_data.iloc[item, 0].split(";")[0]
            image_label = self.valid_data.iloc[item, 0].split(";")[2:]
        image = imread(os.path.join("/proj/ciptmp/xi01cyki", image_name))
        image = gray2rgb(image)
        image = self.transform(image)
        # need the labels as floats not strings for the torch tensor, both return types torch.tensor
        image_label = list(map(int, image_label))
        image_label = np.array(image_label).astype("float")
        image_label = torch.tensor(image_label)
        return tuple((image, image_label))

    def __len__(self):
        #dependent on mode
        if self.mode == "train":
            length = len(self.train_data)
        else:
            length = len(self.valid_data)
        return length

    def pos_weight(self):
        new_labels = [list(map(int, self.train_data.iloc[item, 0].split(";")[1:])) for item in
                      np.arange(len(self.train_data))]
        new_labels = np.array(new_labels)
        cracks = new_labels[:, 1]
        inactive = new_labels[:, 2]
        # column 0 is poly_wafer, column 1 is crack and and column 2 is inactive
        # w_crack i need to take total no. of zeros in column 1/ total no.s of 1 in column1.
        w_cracks = np.sum(cracks == 0) / np.sum(cracks == 1)
        w_inactive = np.sum(inactive == 0) / np.sum(inactive == 1)
        return w_cracks, w_inactive


def get_train_dataset():
    # TODO
    composed_transform = tv.transforms.Compose([tv.transforms.ToPILImage(),
                                                tv.transforms.RandomHorizontalFlip(p=0.5),
                                                tv.transforms.RandomVerticalFlip(p=0.5),
                                                tv.transforms.ColorJitter(brightness=5, contrast=3),
                                                tv.transforms.ToTensor(),
                                                tv.transforms.Normalize(mean=train_mean, std=train_std)])
    data_set = ChallengeDataset("train", "train.csv", 0.7, h=composed_transform)
    return data_set

def get_validation_dataset():
    composed_transform = tv.transforms.Compose([tv.transforms.ToPILImage(),tv.transforms.ToTensor(), tv.transforms.Normalize(mean=train_mean, std=train_std)])
    data_set = ChallengeDataset("val", "train.csv", 0.7, h=composed_transform)
    return data_set


"""
    noor = data_set.train_data
    train_data = []
    for x in np.arange(len(noor)):
        item = noor.iloc[x].__getattribute__("name")
        train_data.append(data_set.__getitem__(item))

    #FIXME instead of returning fixed dataset (list) return object Dataset, so data_set
    return train_data
    
# this needs to return a dataset *without* data augmentation!

    noor = data_set.valid_data
    valid_data = []
    for x in np.arange(len(noor)):
        item = noor.iloc[x].__getattribute__("name")
        valid_data.append(data_set.__getitem__(item))

"""