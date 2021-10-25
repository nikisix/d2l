from torch.utils.data import Dataset, Sampler, DataLoader
import os
import pandas as pd
import random
import torch
import torchvision
# https://stackoverflow.com/questions/51444059/how-to-iterate-over-two-dataloaders-simultaneously-using-pytorch
# https://stackoverflow.com/questions/5480898/python-list-iter-method-called-on-every-loop


def gen_dog_pair_dataset(n, df):
    """ n = dataset size, labels = df(id, breed)"""
    dataset = dict()
    while True:
        b1, b2 = '', ''  # breeds
        d1, d2 = '', ''  # dogs
        if n == 0:
            return dataset
        flip = .5 < random.random()
        if (flip): # TODO maybe rebalance this based on class-vals
            b1 = random.sample(list(df.breed), 1)[0]
            b2 = None
            d1, d2 = random.sample(list(df[df.breed==b1]['id']), 2)
        else: # different breeds
            b1, b2 = random.sample(list(df.breed), 2)
            d1 = random.sample(list(df[df.breed==b1]['id']), 1)[0]
            d2 = random.sample(list(df[df.breed==b2]['id']), 1)[0]
        ix = [(d1, d2), (d2, d1)][d1 < d2]
        if ix in dataset:
            continue
        else:
            dataset[ix] = flip
            n-=1


class Siamese(Dataset):
    """ A dataset that compares two images:
    labels in the form d[img-sha-1, img-sha-2] = True/False
    { ...
    ('fc6abf69e1581b95734830af88c636a0', '37f1a6d8a5a2e3929afa8bdfc47aea2c')
    : False}
    """
    def __init__(
            self, labels, datapath='../data/kaggle_dog_tiny/train/',
            img_extn='jpg', transform=None):
        self.labels = labels
        self.datapath = datapath
        self.img_extn = img_extn
        if transform:
            self.transform = transform
        else:
            self.transform = lambda x: x

    def __len__(self):
        return len(self.labels) #length of the data

    def __getitem__(self, idx):
        ''' Description
        - Get images and labels here
        - Returned images must be tensor
        - Labels should be int
        Params:
            img1, img2: image tensors
            label: True if both tensors are from the same group
        Returns: img1 (tensor), img2 (tensor), label (bool)
        '''
        label = self.labels[idx]
        img1path, img2path = idx
        img1 = torchvision.io.read_image(
                os.path.join(self.datapath, img1path+'.'+self.img_extn))
        img2 = torchvision.io.read_image(
                os.path.join(self.datapath, img2path+'.'+self.img_extn))
        return self._post(img1, img2, label)

    def __iter__(self):
        # maybe remove self her
        for (img1, img2), label in self.labels.items().__iter__():
            yield self._post(img1, img2, label)

    def _post(self, img1, img2, label):
        ''' Post-process images by converting to float then applying transforms
        float necessary b/c torchvision.transforms. Normalize breaks with int'''
        # img1, img2 = [self.transform(img.unsqueeze(0).float()) for img in [img1, img2]]
        img1, img2 = [self.transform(img.float()) for img in [img1, img2]]
        return (img1, img2, label)


class SiameseSampler(Sampler):
    def __init__(self, labels):
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __iter__(self):
        for ix in self.labels.keys():
            yield ix


df_labels = pd.read_csv('../data/kaggle_dog_tiny/labels.csv')
label_dict = gen_dog_pair_dataset(10, df_labels)
train_ds = Siamese(label_dict)
ssampler = SiameseSampler(label_dict)
# test_tup = list(label_dict.keys())[0]
# print(siamese[test_tup])
train_iter = DataLoader(train_ds, batch_size=2, sampler=SiameseSampler, drop_last=True)
