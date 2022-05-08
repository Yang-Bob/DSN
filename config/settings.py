import os
import os.path as osp
# -----------------------------------------------------------------------------
# Config definition
# -----------------------------------------------------------------------------

# -----------------------------------------------------------------------------
# Dataset
# -----------------------------------------------------------------------------
ROOT_DIR = osp.abspath(osp.join(osp.dirname(__file__), '..', '..'))


CIFAR100_Index_DIR = osp.abspath('./data_list/CIFAR100/Index_list/')
CIFAR100_DIR = osp.abspath('/home/YangBoyu/Dataset/cifar100/cifar-100-python')

CUB200_Index_DIR = osp.abspath('./data_list/CUB200/Index_list/')
CUB200_Datasets_Dir = osp.abspath('/home/YangBoyu/Dataset/Birds200')

miniImagenet_Index_DIR = osp.abspath('./data_list/miniImageNet/Index_list/')
miniImagenet_Datasets_Dir = osp.abspath('/home/YangBoyu/Dataset/mini_imagenet/images')

# Image Size
CIFAR_size=32
CUB_size=224
miniImage_size=84

#Session Length
CIFAR100_SessLen = [60, 5, 5, 5, 5, 5, 5, 5, 5]
CUB200_SessLen = [100, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10]
miniImagenet_SessLen = [60, 5, 5, 5, 5, 5, 5, 5, 5]


# Normalization
mean_vals = [0.485, 0.456, 0.406]
std_vals = [0.229, 0.224, 0.225]

mean_vals_cifar = [0.5071, 0.4866, 0.4409]
std_vals_cifar = [0.267, 0.256, 0.276]

#mean_vals = [0.4914, 0.4822, 0.4465]
#std_vals = [0.2023, 0.1994, 0.2010]

# -----------------------------------------------------------------------------
# model
# -----------------------------------------------------------------------------
LR = 3.5e-4
SNAPSHOT_DIR = os.path.join('snapshots')
