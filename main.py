from utils import get_video_files, extract_images_from_video, create_dataset, plot_image
from models import cnn
from loss import FocalLoss, reweight

import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

import numpy as np

def main():
    
    # extra image from video and split into train and test
    files = get_video_files("VideoData")
    for file in files:
        path = "VideoData" + '/' + file
        name = file.split(".")[0]
        extract_images_from_video(video=path, name=name)
    
    # create train and testing dataset
    path_train = "ImageData/train"
    images_train, y_train = create_dataset(path_train, 32, 32)
    path_test = "ImageData/test"
    images_test, y_test = create_dataset(path_test, 32, 32)
    
    # create dict for unique class
    label_dict = {k: v for v, k in enumerate(np.unique(y_train))}
    #print(target_dict)
    # convert class to numeric data
    labels_train = [label_dict[y_train[i]] for i in range(len(y_train))]
    labels_test = [label_dict[y_test[i]] for i in range(len(y_test))]
    #print(target_train)
    
    # create train and test dataset for PyTorch
    train_data = CreateDataset(torch.FloatTensor(images_train), torch.FloatTensor(labels_train))
    test_data = CreateDataset(torch.FloatTensor(images_test), torch.FloatTensor(labels_test))
    
    # define hyperparameter
    batch_size = 128
    learning_rate = 0.001
    reg = 0.00005
    epochs = 20
    
    # train and test dataloader
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)
    
    # use cnn 
    model = cnn()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=reg)
    # use cross-entropy
    critierion = nn.CrossEntropyLoss()
    '''
    # use Focal Loss
    criterion = FocalLoss(weight=per_cls_weights, gamma=1)
    '''
    
    best = 0.0
    for epoch in range(epochs):

        # train loop
        train(epoch, train_loader, model, optimizer, criterion)

        # validation loop
        acc, cm = validate(epoch, test_loader, model, criterion)

        if acc > best:
            best = acc
    
    print('Best Prec @1 Acccuracy: {:.4f}'.format(best))
    
    
    
class CreateDataset(Dataset):

    def __init__(self, images, labels):
        # Add a transform here
        self.images = images
        self.labels = labels

    def __getitem__(self, index):
        return self.images[index], self.labels[index]

    def __len__ (self):
        return len(self.images)
     

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def accuracy(output, target):
    n = target.shape[0]

    _, pred = torch.max(output, dim=-1)

    correct = pred.eq(target).sum() * 1.0

    acc = correct / n

    return acc

def train(epoch, data_loader, model, optimizer, criterion):

    losses = AverageMeter()
    acc = AverageMeter()

    for idx, (data, target) in enumerate(data_loader):

        if torch.cuda.is_available():
            data = data.cuda()
            target = target.cuda()

        optimizer.zero_grad()
        # prediction using forward in the model
        out = model.forward(data)
        # computer loss
        loss = criterion(out, target)
        # backward pass
        loss.backward()
        optimizer.step()

        batch_acc = accuracy(out, target)

        losses.update(loss, out.shape[0])
        acc.update(batch_acc, out.shape[0])

        if idx % 10 == 0:
            print(('Epoch: [{0}][{1}/{2}]\t'
                   'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                   'Prec @1 {top1.val:.4f} ({top1.avg:.4f})\t')
                   .format(epoch, idx, len(data_loader), loss=losses, top1=acc))


def validate(epoch, val_loader, model, criterion):
    losses = AverageMeter()
    acc = AverageMeter()

    num_class = 9
    cm =torch.zeros(num_class, num_class)
    for idx, (data, target) in enumerate(val_loader):

        if torch.cuda.is_available():
            data = data.cuda()
            target = target.cuda()

        with torch.no_grad():
            out = model.forward(data)
            loss = criterion(out, target)

        batch_acc = accuracy(out, target)

        # update confusion matrix
        _, preds = torch.max(out, 1)
        for t, p in zip(target.view(-1), preds.view(-1)):
            cm[t.long(), p.long()] += 1

        losses.update(loss, out.shape[0])
        acc.update(batch_acc, out.shape[0])

    cm = cm / cm.sum(1)
    per_cls_acc = cm.diag().detach().numpy().tolist()
    for i, acc_i in enumerate(per_cls_acc):
        print("Accuracy of Class {}: {:.4f}".format(i, acc_i))

    print("* Prec @1: {top1.avg:.4f}".format(top1=acc))
    return acc.avg, cm


if __name__ == '__main__':
    main()