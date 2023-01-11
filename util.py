import numpy as np
import os
import torch
from backdoors import *
from models import *
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


EPSILON = 1e-7

_dataset_name = ['default', 'cifar10', 'gtsrb', 'imagenet']

_mean = {
    'default':  [0.5, 0.5, 0.5],
    'cifar10':  [0.4914, 0.4822, 0.4465],
    'gtsrb':    [0.3337, 0.3064, 0.3171],
    'imagenet': [0.485, 0.456, 0.406],
}

_std = {
    'default':  [0.5, 0.5, 0.5],
    'cifar10':  [0.2023, 0.1994, 0.2010],
    'gtsrb':    [0.2672, 0.2564, 0.2629],
    'imagenet': [0.229, 0.224, 0.225],
}

_size = {
    'cifar10':  (32, 32),
    'gtsrb':    (32, 32),
    'imagenet': (224, 224),
}

_num = {
    'cifar10':  10,
    'gtsrb':    43,
    'imagenet': 1000,
}


def get_norm(dataset):
    assert dataset in _dataset_name, _dataset_name
    mean = torch.FloatTensor(_mean[dataset])
    std  = torch.FloatTensor(_std[dataset])
    normalize   = transforms.Normalize(mean, std)
    unnormalize = transforms.Normalize(- mean / std, 1 / std)
    return normalize, unnormalize


def get_resize(size):
    if isinstance(size, str):
        assert size in _dataset_name, _dataset_name
        size = _size[size]
    return transforms.Resize(size)


def get_processing(dataset, augment=True, tensor=False, size=None):
    normalize, unnormalize = get_norm(dataset)

    transforms_list = []
    if size is not None:
        transforms_list.append(get_resize(size))
    if augment:
        transforms_list.append(transforms.RandomCrop(_size[dataset], padding=4))
        transforms_list.append(transforms.RandomHorizontalFlip())
    if not tensor:
        transforms_list.append(transforms.ToTensor())
    transforms_list.append(normalize)

    preprocess = transforms.Compose(transforms_list)
    deprocess  = transforms.Compose([unnormalize])
    return preprocess, deprocess

    
def get_dataset(args, train=True, augment=True):
    transform, _ = get_processing(args.dataset, train & augment)
    if args.dataset == 'cifar10':
        dataset = datasets.CIFAR10(args.datadir, train, transform,
                                   download=False)
    elif args.dataset == 'svhn':
        split = 'train' if train else 'test'
        dataset = datasets.SVHN(args.datadir, split, transform, download=False)
    return dataset


def get_loader(args, train=True):
    dataset = get_dataset(args, train)
    dataloader = DataLoader(dataset=dataset, batch_size=args.batch_size,
                            num_workers=4, shuffle=train)
    return dataloader


def get_model(network, pretrained=False):
    if network == 'resnet18':
        model = resnet18()
    elif network == 'preresnet18':
        model = preresnet18()
    elif network == 'vgg11':
        model = vgg11()
    return model


def get_classes(dataset):
    return _num[dataset]


def get_size(dataset):
    return _size[dataset]


def get_backdoor(attack, shape, normalize=None, device=None, args=None):
    if args is not None:
        base_path = f'ckpt/{args.dataset}_{args.network}'
    else:
        base_path = ''

    if 'refool' in attack:
        backdoor = Refool(shape, attack.split('_')[1], device=device)
    elif attack == 'wanet':
        backdoor = WaNet(shape, device=device)
        noise_path    = f'{base_path}_wanet_noise.pt'
        identity_path = f'{base_path}_wanet_identity.pt'
        if os.path.exists(noise_path) & os.path.exists(identity_path):
            backdoor.noise_grid    = torch.load(noise_path).to(device)
            backdoor.identity_grid = torch.load(identity_path).to(device)
        else:
            torch.save(backdoor.noise_grid.cpu(),    noise_path)
            torch.save(backdoor.identity_grid.cpu(), identity_path)
    elif attack == 'invisible':
        backdoor = Invisible()
    elif attack in ['blend', 'sig', 'polygon']:
        backdoor = Other(attack, device=None)
    elif attack == 'filter':
        backdoor = Filter()
    elif attack == 'inputaware':
        backdoor = InputAware(normalize, device=device)
        mask_path = f'{base_path}_inputaware_mask.pt'
        genr_path = f'{base_path}_inputaware_pattern.pt'
        if os.path.exists(mask_path) & os.path.exists(genr_path):
            backdoor.net_mask = torch.load(mask_path).to(device)
            backdoor.net_genr = torch.load(genr_path).to(device)
    elif attack == 'dynamic':
        backdoor = Dynamic(normalize, device=device)
        genr_path = f'{base_path}_dynamic_pattern.pt'
        if os.path.exists(genr_path):
            backdoor.net_genr = torch.load(genr_path).to(device)
    elif 'dfst' in attack:
        backdoor = DFST(normalize, device=device)
        genr_path = f'{base_path}_dfst_generator.pt'
        if os.path.exists(genr_path):
            backdoor.genr_a2b = torch.load(genr_path).to(device)
    else:
        backdoor = None
    return backdoor
