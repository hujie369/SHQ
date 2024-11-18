#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 31 09:22:53 2018

@author: tshzzz
"""
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
import os
import argparse
from utils import txt_logger
import torch.optim.lr_scheduler as lr_scheduler
import time
from models import mobilenet
from quantize_utils import load_qnet
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--model', type=str, default='mobilenet')
    parser.add_argument('--output', type=str, default='./pre/')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=0.005, help='learning rate')
    parser.add_argument('-dataset', help='which dataset to use', type=str, choices=['cifar10', 'imagenet'], default='cifar10')
    args = parser.parse_args()
    return args

args = parse_args()

def load_pretrain(pred_dict,model):
    model_dict = model.state_dict()
    for (k,v),(k1,v1) in zip(pred_dict.items(),model_dict.items()):
        if v.shape == v1.shape:
            model_dict[k1] = v
    return model_dict

class Cutout(object):
    """Randomly mask out one or more patches from an image.
    Args:
        n_holes (int): Number of patches to cut out of each image.
        length (int): The length (in pixels) of each square patch.
    """

    def __init__(self, n_holes, length):
        self.n_holes = n_holes
        self.length = length

    def __call__(self, img):
        """
        Args:
            img (Tensor): Tensor image of size (C, H, W).
        Returns:
            Tensor: Image with n_holes of dimension length x length cut out of it.
        """
        h = img.size(1)
        w = img.size(2)

        mask = np.ones((h, w), np.float32)

        for n in range(self.n_holes):
            y = np.random.randint(h)
            x = np.random.randint(w)

            y1 = np.clip(y - self.length // 2, 0, h)
            y2 = np.clip(y + self.length // 2, 0, h)
            x1 = np.clip(x - self.length // 2, 0, w)
            x2 = np.clip(x + self.length // 2, 0, w)

            mask[y1: y2, x1: x2] = 0.

        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        img = img * mask

        return img

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if args.dataset == 'cifar10':
    train_transforms = transforms.Compose([])
    normalize = transforms.Normalize(mean=[x / 255.0 for x in [125.3, 123.0, 113.9]],
                                        std=[x / 255.0 for x in [63.0, 62.1, 66.7]])
    train_transforms.transforms.append(transforms.RandomCrop(32, padding=4))
    train_transforms.transforms.append(transforms.RandomHorizontalFlip())
    train_transforms.transforms.append(transforms.ToTensor())
    train_transforms.transforms.append(normalize)
    # train_transforms.transforms.append(Cutout(n_holes=1, length=16))

    transform = transforms.Compose([
        transforms.ToTensor(),
        normalize])
    trainloader, testloader, n_class = None, None, None

    trainsets = torchvision.datasets.CIFAR10(root='../cifar10/', train=True,
                                            download=True, transform=train_transforms)
    trainloader = torch.utils.data.DataLoader(trainsets, batch_size=512,
                                                shuffle=True, num_workers=16)

    evalsets = torchvision.datasets.CIFAR10(root='../cifar10/', train=False,
                                            download=True, transform=transform)
    evalloader = torch.utils.data.DataLoader(evalsets, batch_size=512,
                                                shuffle=False, num_workers=16)
    n_class = 10
if args.dataset == 'imagenet':
    from imagenet_loader import data_loader
    trainloader, evalloader = data_loader(root='../ImageNet', batch_size=args.batch_size)
    n_class = 1000



def mkdir(out_dir):
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)

def eval_train(model,loader):
    model.eval()    
    correct = 0.0
    for img,label in loader:
        img = img.to(device)
        pred = model(img)
        _,pred_idx = torch.max(pred,1)
        correct += torch.sum(pred_idx.cpu() == label)
    eval_acc = float(correct) / len(loader.dataset)
    model.train()
    return eval_acc


def train(model,
          optimizer,
          criterion,
          epochs,
          out_dir,
          logger,
          trainloader,
          evalloader):

    step = 0
    best_acc = 0
    scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=(epochs // 9) + 1)

    for epoch in range(epochs):
        correct = 0.0
        total_loss = 0
        for img,label in trainloader:
            optimizer.zero_grad()
            label = label.to(device)
            img = img.to(device)
            pred = model(img)
            
            _,pred_idx = torch.max(pred,1)
            correct += torch.sum(pred_idx == label)
            
            loss = criterion(pred,label)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * img.size(0)
            step += 1

        total_loss /= len(trainloader.dataset)
        logger.add_scalar('train loss', total_loss, step)
        train_acc = float(correct) / len(trainloader.dataset)        
        logger.add_scalar('train score', train_acc, step)
        eval_acc = eval_train(model,evalloader)
        logger.add_scalar('eval score', eval_acc, step)
        if eval_acc > best_acc:
            best_acc = eval_acc
            model_path = os.path.join(out_dir, 'best.pth')
            torch.save(model.state_dict(), model_path)
            # torch.save(model, model_path)
        model_path = os.path.join(out_dir, 'model.pth')
        # torch.save(model, model_path)
        torch.save(model.state_dict(), model_path)
        logger.print_info(epoch)
        scheduler.step(epoch)


# GNAS idea
# if __name__ == '__main__':
#     print(args.model)
#     torch.backends.cudnn.benchmark = True
#     torch.cuda.manual_seed(100)

#     if args.model == 'mobilenet':
#         model = mobilenet.qmobilenet(class_num=n_class)
#     else:
#         model = locals()[args.model](10)
#     model = model.to(device)
#     if False:
#         # model = torch.load('./pre/mobilenet/best.pth')
#         load_qnet(model, './pre/mobilenet/best.pth')
#         print(model.modules)
#     else:
#         optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9)
#         criterion = nn.CrossEntropyLoss()
#         epochs = args.epochs
#         mkdir(args.output)
#         out_dir = args.output+args.model+args.dataset
#         mkdir(out_dir)
#         logger = txt_logger(out_dir, 'training', 'log.txt')
#         start = time.time()
#         train(model,optimizer,criterion,epochs,out_dir,logger,trainloader,evalloader)
#         end = time.time()
#         time_elapsed = end - start
#         logger.logger.info('total time {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))

if __name__ == '__main__':
    from utils import train, test
    print(args.model)
    torch.backends.cudnn.benchmark = True
    torch.cuda.manual_seed(100)

    if args.model == 'mobilenet':
        model = mobilenet.qmobilenet(class_num=n_class)
    else:
        model = locals()[args.model](n_class)
    model = model.to(device)
    if False:
        # model = torch.load('./pre/mobilenet/best.pth')
        load_qnet(model, './pre/mobilenet/best.pth')
        print(model.modules)
    else:
        # optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9)
        # criterion = nn.CrossEntropyLoss()
        epochs = args.epochs
        mkdir(args.output)
        out_dir = args.output+args.model+args.dataset
        mkdir(out_dir)
        logger = txt_logger(out_dir, 'training', 'log.txt')
        start = time.time()
        optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
        train(model, trainloader, optimizer, epoch=epochs, device=device, verbose=True)
        # train(model,optimizer,criterion,epochs,out_dir,logger,trainloader,evalloader)
        top1 = eval_train(model, evalloader)
        print(f'top1:{top1}')
        model_path = os.path.join(out_dir, 'mix_best.pth')
        torch.save(model.state_dict(), model_path)
        end = time.time()
        time_elapsed = end - start
        logger.logger.info('total time {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))

    
    
    

    