from torchvision import transforms
from torch.utils.data import DataLoader

from config import settings
from data.CIFAR100 import NC_CIFAR100
from data.CIFAR100_val import CIFAR100_val
from data.CUB200 import NC_CUB200
from data.CUB200_val import NC_CUB200_val
from data.miniImageNet import NC_miniImageNet
from data.miniImageNet_val import NC_miniImageNet_val


def data_loader(args):
    batch = args.batch_size
    if args.dataset == 'CIFAR100':
        mean_vals = settings.mean_vals_cifar
        std_vals = settings.std_vals_cifar
        tsfm_train = transforms.Compose([transforms.ToPILImage(),
                                         transforms.Resize((settings.CIFAR_size, settings.CIFAR_size)),
                                         transforms.RandomCrop(settings.CIFAR_size, padding=12),
                                         transforms.RandomHorizontalFlip(),
                                         transforms.ToTensor(),
                                         transforms.Normalize(mean_vals, std_vals)
                                         ])
    if args.dataset == 'CUB200':
        mean_vals = settings.mean_vals
        std_vals = settings.std_vals
        tsfm_train = transforms.Compose([  # transforms.ToPILImage(),
            transforms.Resize((256, 256)),
            transforms.RandomCrop(224, padding=0),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean_vals, std_vals)
        ])
    if args.dataset == 'miniImageNet':
        mean_vals = settings.mean_vals
        std_vals = settings.std_vals
        tsfm_train = transforms.Compose([  # transforms.ToPILImage(),
            transforms.RandomResizedCrop(settings.miniImage_size),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
            transforms.ToTensor(),
            transforms.Normalize(mean_vals, std_vals)
        ])

    if args.dataset == 'CIFAR100':
        img_train = NC_CIFAR100(args, transform=tsfm_train)
    if args.dataset == 'CUB200':
        img_train = NC_CUB200(args, transform=tsfm_train)
    if args.dataset == 'miniImageNet':
        img_train = NC_miniImageNet(args, transform=tsfm_train)

    if args.sesses >0:
        return img_train, tsfm_train
    else:
        train_loader = DataLoader(img_train, batch_size=batch, shuffle=True, num_workers=8)
        return train_loader


def val_loader(args):
    batch = args.batch_size
    if args.dataset == 'CIFAR100':
        size = settings.CIFAR_size
        mean_vals = settings.mean_vals_cifar
        std_vals = settings.std_vals_cifar
        tsfm_train = transforms.Compose([transforms.ToPILImage(),
                                         transforms.Resize((size, size)),  # 224
                                         transforms.ToTensor(),
                                         transforms.Normalize(mean_vals, std_vals)
                                         ])
    if args.dataset == 'CUB200':
        size = settings.CUB_size
        mean_vals = settings.mean_vals
        std_vals = settings.std_vals
        tsfm_train = transforms.Compose([  # transforms.ToPILImage(),
            transforms.Resize((256,256)),
            transforms.CenterCrop((224,224)),
            transforms.ToTensor(),
            transforms.Normalize(mean_vals, std_vals)
        ])
    if args.dataset == 'miniImageNet':
        size = settings.miniImage_size
        mean_vals = settings.mean_vals
        std_vals = settings.std_vals
        tsfm_train = transforms.Compose([  # transforms.ToPILImage(),
            transforms.Resize(92),  # 92
            transforms.CenterCrop(size),
            transforms.ToTensor(),
            transforms.Normalize(mean_vals, std_vals)
        ])

    if args.dataset == 'CIFAR100':
        img_val = CIFAR100_val(args, transform=tsfm_train)
    if args.dataset == 'CUB200':
        img_val = NC_CUB200_val(args, transform=tsfm_train)
    if args.dataset == 'miniImageNet':
        img_val = NC_miniImageNet_val(args, transform=tsfm_train)

    val_loader = DataLoader(img_val, batch_size=batch, shuffle=False, num_workers=8)

    return val_loader
