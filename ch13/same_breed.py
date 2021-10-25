from d2l import torch as d2l
from torch import nn
import os
import pandas as pd
import torch
import torchvision

from siamese_dataset import Siamese, SiameseSampler, gen_dog_pair_dataset

#@save
d2l.DATA_HUB['dog_tiny'] = (d2l.DATA_URL + 'kaggle_dog_tiny.zip',
                            '0cb91d09b814ecdc07b50f31f8dcad3e81d6a86d')

# If you use the full dataset downloaded for the Kaggle competition, change
# the variable below to False
demo = True
if demo:
    data_dir = d2l.download_extract('dog_tiny')
else:
    data_dir = os.path.join('..', 'data', 'dog-breed-identification')

batch_size = 4 if demo else 128
valid_ratio = 0.1

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
                                     [0.229, 0.224, 0.225])])


transform_test = torchvision.transforms.Compose([
    torchvision.transforms.Resize(256),
    # Crop a square of 224 by 224 from the center of the image
    torchvision.transforms.CenterCrop(224),
    # torchvision.transforms.ToTensor(),  # loading images in as tensors
    torchvision.transforms.Normalize([0.485, 0.456, 0.406],
                                     [0.229, 0.224, 0.225])])

df_labels = pd.read_csv('../data/kaggle_dog_tiny/labels.csv')
label_dict_train = gen_dog_pair_dataset(100, df_labels)  # TODO fine-tune nums
label_dict_valid = gen_dog_pair_dataset(60, df_labels)  # TODO fine-tune nums
label_dict_test = gen_dog_pair_dataset(100, df_labels)

train_ds = Siamese(label_dict_train, transform=transform_train)
train_iter = torch.utils.data.DataLoader(
    train_ds, batch_size,
    sampler=SiameseSampler(label_dict_train), drop_last=True)

valid_ds = Siamese(label_dict_valid, transform=transform_train)  # same txfrm as train
valid_iter = torch.utils.data.DataLoader(
    valid_ds, batch_size,
    sampler=SiameseSampler(label_dict_valid), drop_last=True)

test_ds = Siamese(label_dict_test, transform=transform_test)
test_iter = torch.utils.data.DataLoader(
    test_ds, batch_size,
    sampler=SiameseSampler(label_dict_test), drop_last=True)

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


loss = nn.CrossEntropyLoss(reduction='none')

def evaluate_loss(data_iter, net, devices):
    l_sum, n = 0.0, 0
    for features, labels in data_iter:
        features, labels = features.to(devices[0]), labels.to(devices[0])
        outputs = net(features)
        l = loss(outputs, labels)
        l_sum = l.sum()
        n += labels.numel()
    return l_sum / n


def train(net, train_iter, valid_iter, num_epochs, lr, wd, devices, lr_period,
          lr_decay):
    # Only train the small custom output network
    net = nn.DataParallel(net, device_ids=devices).to(devices[0])
    trainer = torch.optim.SGD((param for param in net.parameters()
                               if param.requires_grad), lr=lr,
                              momentum=0.9, weight_decay=wd)
    scheduler = torch.optim.lr_scheduler.StepLR(trainer, lr_period, lr_decay)
    num_batches, timer = len(train_iter), d2l.Timer()
    animator = d2l.Animator(xlabel='epoch', xlim=[1, num_epochs],
                            legend=['train loss', 'valid loss'])
    for epoch in range(num_epochs):
        metric = d2l.Accumulator(2)
        for i, (img1s, img2s, labels) in enumerate(train_iter):
            timer.start()

            # push onto gpu
            img1s = img1s.to(devices[0])
            img2s = img2s.to(devices[0])
            labels = labels.to(devices[0])

            trainer.zero_grad()
            import ipdb; ipdb.set_trace()  # TODO BREAKPOINT

            # TODO YOU ARE HERE - choose how to cat the imgs
            # Then prepare the network to accept...
            # ipdb> torch.cat((img1s, img2s), 1).shape
            # torch.Size([4, 6, 224, 224])
            # ipdb> torch.cat((img1s, img2s), 0).shape
            # torch.Size([8, 3, 224, 224])
            # ipdb> torch.cat((img1s, img2s), 3).shape
            # torch.Size([4, 3, 224, 448])
            output1s = net(img1s)
            output2s = net(img2s)
            l = loss(output, labels).sum()
            l.backward()
            trainer.step()
            metric.add(l, labels.shape[0])
            timer.stop()
            if (i + 1) % (num_batches // 5) == 0 or i == num_batches - 1:
                animator.add(epoch + (i + 1) / num_batches,
                             (metric[0] / metric[1], None))
        if valid_iter is not None:
            valid_loss = evaluate_loss(valid_iter, net, devices)
            animator.add(epoch + 1, (None, valid_loss))
        scheduler.step()
    if valid_iter is not None:
        print(f'train loss {metric[0] / metric[1]:.3f}, '
              f'valid loss {valid_loss:.3f}')
    else:
        print(f'train loss {metric[0] / metric[1]:.3f}')
    print(f'{metric[1] * num_epochs / timer.sum():.1f} examples/sec '
          f'on {str(devices)}')


devices, num_epochs, lr, wd = d2l.try_all_gpus(), 5, 0.001, 1e-4
lr_period, lr_decay, net = 10, 0.1, get_net(devices)
train(net, train_iter, valid_iter, num_epochs, lr, wd, devices, lr_period,
      lr_decay)


# TODO Prediction
# ref: ./kaggle_dog.py
