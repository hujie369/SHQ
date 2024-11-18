import conf
import torch
from torch.nn.functional import softmax
import torch.optim as optim
import warnings
from progress.bar import Bar
import torchvision
from torchvision.transforms import transforms
from torch import log
import torchvision.transforms.functional as F
import torch.nn as nn

from quantize_utils import QConv2d, load_qnet

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

def has_childrean(module):
    try:
        next(module.children())
        return True
    except StopIteration:
        return False


def train(model, dataloader, optimizer, epoch=3, loss_fn=torch.nn.CrossEntropyLoss(), device='cpu', verbose=False, testloader=None, dataset='cifar10'):
    max_top1 = 0.0
    if device == 'cpu':
        warnings.warn('using cpu for training can be very slow', RuntimeWarning)
    model = model.to(device)
    with Bar('Training:', max=epoch*len(dataloader), suffix='%(percent)d%%') as bar:
        for e in range(epoch):
            for i, (data, target) in enumerate(dataloader):
                data, target = data.to(device), target.to(device)
                optimizer.zero_grad()
                out = model(data)
                loss = loss_fn(out, target)
                loss.backward()
                optimizer.step()
                # if verbose:
                #     if i % 50 == 0:
                #         print('Epoch: {} [{}/{}]\tLoss: {:.6f}'.format(
                #             e, i * len(data), len(dataloader.dataset), loss.item()
                #         ))
                bar.next()
            top1,_ = test(model,testloader,'cuda')
            if dataset == 'cifar10':
                top1 = top1 * 10
            if verbose:
                import matplotlib.pyplot as plt
                print(f'epoch:{e},top1:{top1}')
                plt.xlabel('epoch')
                plt.ylabel('acc')
                plt.title('train')
                plt.xlim(0, epoch)
                plt.ylim(0, 1)
                plt.scatter(e, top1.cpu(),s= 3)
                
            if top1 > max_top1:
                max_top1 = top1
                # if verbose == True:
                #     torch.save({'model': model.state_dict()}, f'cifar.pth')
                # if verbose == True:
                #     torch.save(model, f'./pre/mobilenet/act_cifar.pth')
    # if verbose:
    #     plt.savefig(f'./train_pic/cifar10_{max_top1}.png')
    return max_top1


def test(model, dataloader, device='cpu'):
    model = model.to(device)
    model.eval()
    correct_1 = 0.0
    correct_5 = 0.0

    with torch.no_grad():
        # for image, label in Bar('Testing').iter(dataloader):
        for image, label in dataloader:

            if device != 'cpu':
                image, label = image.to(device), label.to(device)

            output = model(image)
            _, pred = output.topk(5, 1, largest=True, sorted=True)

            label = label.view(label.size(0), -1).expand_as(pred)
            correct = pred.eq(label).float()

            #compute top 5
            # correct_5 += correct[:, :5].sum()

            #compute top1
            correct_1 += correct[:, :1].sum()



        # TODO:cifar & imagenet acc diff
        # top1, top5 = correct_1 / len(dataloader.dataset) * 10, correct_5 / len(dataloader.dataset) * 10
        top1, top5 = correct_1 / len(dataloader.dataset), correct_5 / len(dataloader.dataset)
        # print(top1, top5)
    return top1, top5


def finetune(qmodel, trainloader, epochs=1, device='cpu', need_test=False, testloader=None, verbose=False, dataset='cifar10'):
    '''
    finetune qmodel for #epochs(default 1) using SGD optimizer lr=0.0001, momentum=0.9
    '''
    optimizer = optim.SGD(qmodel.parameters(), lr=0.01, momentum=0.9)
    return train(qmodel, trainloader, optimizer, epoch=epochs, device=device, verbose=verbose, testloader=testloader, dataset=dataset)
    # if need_test:
    #     if testloader is None:
    #         raise ValueError('need Test, but test dataloader is not given')
    #     top1, top5 = test(qmodel, testloader, device=device)
        
    # return top1, top5


def cifar10DataLoader(root='../cifar10', train=True, normalized=False, **kwargs):
    
    '''
    include Cutout
    '''
    train_transforms = transforms.Compose([])
    normalize = transforms.Normalize(mean=[x / 255.0 for x in [125.3, 123.0, 113.9]],
                                        std=[x / 255.0 for x in [63.0, 62.1, 66.7]])
    train_transforms.transforms.append(transforms.RandomCrop(32, padding=4))
    train_transforms.transforms.append(transforms.RandomHorizontalFlip())
    train_transforms.transforms.append(transforms.ToTensor())
    train_transforms.transforms.append(normalize)
    train_transforms.transforms.append(Cutout(n_holes=1, length=16))

    transform = transforms.Compose([
        transforms.ToTensor(),
        normalize])
    
    if train:

        trainsets = torchvision.datasets.CIFAR10(root='../cifar10/', train=True,
                                                download=True, transform=train_transforms)
        trainloader = torch.utils.data.DataLoader(trainsets, batch_size=256,
                                                    shuffle=True, num_workers=16)
        return trainloader
    
    else:

        evalsets = torchvision.datasets.CIFAR10(root='../cifar10/', train=False,
                                                download=True, transform=transform)
        evalloader = torch.utils.data.DataLoader(evalsets, batch_size=256,
                                                    shuffle=False, num_workers=16)
        return evalloader
    

    # transform = transforms.Compose([transforms.Resize([36,36]),
    #                                 transforms.RandomCrop([32,32]),
    #                                 transforms.RandomHorizontalFlip(),
    #                                 transforms.ToTensor(),
    #                                 transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    #                                 ]) if normalized else transforms.ToTensor()

    # dataLoader = torch.utils.data.DataLoader(
    #     dataset=torchvision.datasets.CIFAR10(
    #         root=root,
    #         download=False,
    #         transform=transform,
    #         train=train
    #     ),
    #     **kwargs
    # )
    # return dataLoader


def getCifar10Model(name):
    '''
    mobilenet
    '''
    if name == 'mobilenet':
        from models.mobilenet import qmobilenet
        model = qmobilenet(class_num=10)
        load_qnet(model, conf.mobilenet_path)
        print('模型已载入')
    else:
        raise NotImplementedError(f'{name} is not supported yet')
    return model


### log_txt
import logging
import os
import sys
import numpy as np


class txt_logger(object):

    def __init__(self, save_dir, name, filename):

        logger = logging.getLogger(name)
        logger.setLevel(logging.DEBUG)

        ch = logging.StreamHandler(stream=sys.stdout)
        ch.setLevel(logging.DEBUG)
        formatter = logging.Formatter("%(asctime)s %(name)s %(levelname)s: %(message)s")

        if save_dir:
            fh = logging.FileHandler(os.path.join(save_dir, filename))
            fh.setLevel(logging.DEBUG)
            fh.setFormatter(formatter)
            logger.addHandler(fh)
        self.logger = logger

        self.info = {}

    def add_scalar(self, tag, value, step=None):
        if tag in self.info:
            self.info[tag].append(value)
        else:
            self.info[tag] = [value]


    def print_info(self, epoch):

        info_line = 'epoch {}: '.format(epoch)
        for i in self.info.keys():
            info = np.array(self.info[i]).mean()
            info_line += i + ':' + str(round(info,4)) + ', '

        print(info_line)
        self.logger.info(
            info_line
        )
        self.info = {}
