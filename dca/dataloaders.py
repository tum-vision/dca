from typing import Tuple
from os.path import join as pjoin
import torch
from torch.utils.data import DataLoader, Subset
import torchvision.transforms as transforms
import torchvision.datasets as datasets


def dup_collate_fn(dups: int):

    def collate_fn(data):
        imgs, gts = tuple(zip(*data))
        t = torch.stack(imgs, dim=0)
        return t.repeat(dups, *(1,)*(t.ndim-1)), torch.as_tensor(gts)

    return collate_fn


class CIFAR10Info:
    outclass = 10
    imgshape = (3, 32, 32)
    counts = {'train': 50000, 'test': 10000}
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)


def get_cifar10_train_loaders(
    data_dir: str, train_val_split: float, workers: int, pin_memory: bool,
    tbatch: int, vbatch: int, tdups: int = 1, vdups: int = 1
) -> Tuple[DataLoader, DataLoader]:
    cifar10_dir = pjoin(data_dir, 'cifar10')
    normalize = transforms.Normalize(mean=CIFAR10Info.mean,
                                     std=CIFAR10Info.std)

    train_data = datasets.CIFAR10(root=cifar10_dir, train=True,
                                  transform=transforms.Compose([
                                      transforms.RandomHorizontalFlip(),
                                      transforms.RandomCrop(32, 4),
                                      transforms.ToTensor(),
                                      normalize]), download=True)

    val_data = datasets.CIFAR10(root=cifar10_dir, train=True,
                                transform=transforms.Compose([
                                    transforms.ToTensor(),
                                    normalize]), download=True)

    nb_train = int(len(train_data) * train_val_split)
    train_indices = list(range(nb_train))
    val_indices = list(range(nb_train, len(train_data)))

    train_loader = DataLoader(
        Subset(train_data, train_indices), batch_size=tbatch,
        num_workers=workers, pin_memory=pin_memory, shuffle=True,
        collate_fn=dup_collate_fn(tdups)
    ) if tdups > 1 else DataLoader(
        Subset(train_data, train_indices), batch_size=tbatch,
        num_workers=workers, pin_memory=pin_memory, shuffle=True)

    val_loader = DataLoader(
        Subset(val_data, val_indices), batch_size=vbatch,
        num_workers=workers, pin_memory=pin_memory, shuffle=False,
        collate_fn=dup_collate_fn(vdups)
    ) if vdups > 1 else DataLoader(
        Subset(val_data, val_indices), batch_size=vbatch,
        num_workers=workers, pin_memory=pin_memory, shuffle=False)

    return train_loader, val_loader


def get_cifar10_test_loader(
    data_dir: str, workers: int, pin_memory: bool, batch: int, dups: int = 1
) -> DataLoader:
    cifar10_dir = pjoin(data_dir, 'cifar10')
    normalize = transforms.Normalize(mean=CIFAR10Info.mean,
                                     std=CIFAR10Info.std)

    test_data = datasets.CIFAR10(root=cifar10_dir, train=False,
                                 transform=transforms.Compose([
                                     transforms.ToTensor(),
                                     normalize,
                                 ]), download=True)

    test_loader = DataLoader(
        test_data, batch_size=batch, num_workers=workers, shuffle=False,
        pin_memory=pin_memory, collate_fn=dup_collate_fn(dups)
    ) if dups > 1 else DataLoader(
        test_data, batch_size=batch, num_workers=workers, shuffle=False,
        pin_memory=pin_memory)
    return test_loader


class SVHNInfo:
    outclass = 10
    imgshape = (3, 32, 32)
    split = ('train', 'test', 'extra')
    counts = {'train': 73257, 'test': 26032, 'extra': 531131}
    mean = (0.4376821, 0.4437697, 0.47280442)
    std = (0.19803012, 0.20101562, 0.19703614)


def get_svhn_train_loaders(
    data_dir: str, train_val_split: float, workers: int, pin_memory: bool,
    tbatch: int, vbatch: int, tdups: int = 1, vdups: int = 1
) -> Tuple[DataLoader, DataLoader]:
    svhn_dir = pjoin(data_dir, 'svhn')
    normalize = transforms.Normalize(SVHNInfo.mean, SVHNInfo.std)

    train_data = datasets.SVHN(
        root=svhn_dir, split='train', transform=transforms.Compose([
            transforms.RandomHorizontalFlip(), transforms.RandomCrop(32, 4),
            transforms.ToTensor(), normalize]), download=True)

    val_data = datasets.SVHN(
        root=svhn_dir, split='train', transform=transforms.Compose([
            transforms.ToTensor(), normalize]), download=True)

    nb_train = int(len(train_data) * train_val_split)
    train_indices = list(range(nb_train))
    val_indices = list(range(nb_train, len(train_data)))

    train_loader = DataLoader(
        Subset(train_data, train_indices), batch_size=tbatch,
        num_workers=workers, pin_memory=pin_memory, shuffle=True,
        collate_fn=dup_collate_fn(tdups)
    ) if tdups > 1 else DataLoader(
        Subset(train_data, train_indices), batch_size=tbatch,
        num_workers=workers, pin_memory=pin_memory, shuffle=True)

    val_loader = DataLoader(
        Subset(val_data, val_indices), batch_size=vbatch,
        num_workers=workers, pin_memory=pin_memory, shuffle=False,
        collate_fn=dup_collate_fn(vdups)
    ) if vdups > 1 else DataLoader(
        Subset(val_data, val_indices), batch_size=vbatch,
        num_workers=workers, pin_memory=pin_memory, shuffle=False)

    return train_loader, val_loader


def get_svhn_test_loader(
    data_dir: str, workers: int, pin_memory: bool, batch: int, dups: int = 1
) -> DataLoader:
    svhn_dir = pjoin(data_dir, 'svhn')
    normalize = transforms.Normalize(SVHNInfo.mean, SVHNInfo.std)

    test_data = datasets.SVHN(
        root=svhn_dir, split='test', transform=transforms.Compose([
            transforms.ToTensor(), normalize]), download=True)

    test_loader = DataLoader(
        test_data, batch_size=batch, num_workers=workers, shuffle=False,
        pin_memory=pin_memory, collate_fn=dup_collate_fn(dups)
    ) if dups > 1 else DataLoader(
        test_data, batch_size=batch, num_workers=workers, shuffle=False,
        pin_memory=pin_memory)
    return test_loader
