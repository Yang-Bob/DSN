import os
import random
import pickle
import numpy as np
from config import settings


def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict


class NC_CIFAR100():
    def __init__(self, args, transform=None, c_way=5, k_shot=5):
        self.name = 'NC_CIFAR100'
        self.root = settings.CIFAR100_DIR
        self.IndexDir = os.path.join(settings.CIFAR100_Index_DIR, args.seed)
        self.Img, self.Label = self.Read_CIFAR(os.path.join(self.root, 'train'))
        self.transform = transform
        self.count = 0
        self.Set_Session(args)

    def Set_Session(self, args):
        self.sess = args.sess
        self.sess_img, self.sess_label = self.Read_Index_Sess()
        self.len = len(self.sess_img)
        # print(len(self.Index_list))

    def get_data(self):
        return self.sess_img, self.sess_label

    def Read_CIFAR(self, data_dir):
        a = unpickle(data_dir)
        X = a[b'data'].reshape(50000, 3, 32, 32).transpose(0, 2, 3, 1)
        Y = a[b'fine_labels']

        return X, Y

    def Read_Index_Sess(self):
        idx = []
        label = []
        image = []
        f = open(self.IndexDir + '/session_' + str(self.sess + 1) + '.txt', 'r')
        while True:
            lines = f.readline()
            if not lines:
                break
            tmp = int(lines.strip())
            idx.append(tmp)
            label.append(self.Label[tmp])
            image.append(self.Img[tmp])
        return image, label
    
    def load_frame(self, index):
        Image = self.sess_img[index]
        Label = self.sess_label[index]

        return Image, Label
    
    def __len__(self):
        return self.len

    def __getitem__(self, index):
        Image, Label = self.load_frame(index)
        if self.transform is not None:
            Image = self.transform(Image)
        self.count = self.count + 1

        return Image, Label


