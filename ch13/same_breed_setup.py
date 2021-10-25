from d2l import torch as d2l
from torch import nn
import torchvision
import pandas as pd

from siamese_dataset import Siamese, gen_dog_pair_dataset
# from kaggle_dog import get_net

def get_net(devices):
    finetune_net = nn.Sequential()
    finetune_net.features = torchvision.models.resnet34(pretrained=True)
    # Define a new output network
    # There are 120 output categories
    finetune_net.output_new = nn.Sequential(nn.Linear(1000, 256),
                                            nn.ReLU(),
                                            nn.Linear(256, 120))
    # Move model to device
    finetune_net = finetune_net.to(devices[0])
    # Freeze feature layer params
    for param in finetune_net.features.parameters():
        param.requires_grad = False
    return finetune_net


transform_train = torchvision.transforms.Compose([
    # Randomly crop the image to obtain an image with an area of 0.08 to 1 of
    # the original area and height to width ratio between 3/4 and 4/3. Then,
    # scale the image to create a new image with a height and width of 224
    # pixels each
    torchvision.transforms.RandomResizedCrop(224, scale=(0.08, 1.0),
                                             ratio=(3.0/4.0, 4.0/3.0)),
    torchvision.transforms.RandomHorizontalFlip(),
    # Randomly change the brightness, contrast, and saturation
    torchvision.transforms.ColorJitter(brightness=0.4,
                                       contrast=0.4,
                                       saturation=0.4),
    # Add random noise
    # torchvision.transforms.ToTensor(),
    # Standardize each channel of the image
    torchvision.transforms.Normalize([0.485, 0.456, 0.406],
                                     [0.229, 0.224, 0.225])
    ])


devices = d2l.try_all_gpus()
net = get_net(devices)

df_labels = pd.read_csv('../data/kaggle_dog_tiny/labels.csv')
label_dict = gen_dog_pair_dataset(10, df_labels)
siamese = Siamese(label_dict, transform=transform_train)
test_tup = list(label_dict.keys())[0]
# print(siamese[test_tup][0])
# print(net(siamese[test_tup][0]))

