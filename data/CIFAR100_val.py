import os
import numpy as np
from config import settings


def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict


class CIFAR100_val():
    def __init__(self, args, transform=None,c_way=5, k_shot=5):
        self.name = 'NC_CIFAR100'
        self.root = settings.CIFAR100_DIR
        self.IndexDir = os.path.join(settings.CIFAR100_Index_DIR, args.seed)
        self.Img, self.Label = self.Read_CIFAR(os.path.join(self.root, 'test'))
        self.transform = transform
        self.count = 0
        self.Set_Session(args)

    def Set_Session(self, args):
        self.sess = args.sess
        self.Index_list_all = []
        for sess in range(self.sess + 1):
            self.Index_list = self.Read_Index_Sess(sess)
            self.Index_list_all += self.Index_list
        self.len = len(self.Index_list_all)

    def Update_Session(self, sess):
        self.Index_list = self.Read_Index_Sess(sess)
        self.Index_list_all = self.Index_list
        self.len = len(self.Index_list)

    def Read_CIFAR(self, data_dir):
        a = unpickle(data_dir)
        X = a[b'data'].reshape(10000, 3, 32, 32).transpose(0, 2, 3, 1)
        Y = a[b'fine_labels']

        return X, Y

    def Read_Index_Sess(self, sess):
        idx = []
        f = open(self.IndexDir + '/test_' + str(sess + 1) + '.txt', 'r')
        while True:
            lines = f.readline()
            if not lines:
                break
            idx.append(int(lines.strip()))

        return idx

    def Random_choose(self):
        Index = np.random.choice(self.Index_list, 1, replace=False)[0]

        return Index

    def load_frame(self, Index):
        Image = self.Img[Index]
        Label = self.Label[Index]

        return Image, Label

    def __len__(self):
        return self.len

    def __getitem__(self, index):
        Index = self.Index_list_all[index]
        Image = self.Img[Index]
        Label = self.Label[Index]
        if self.transform is not None:
            Image = self.transform(Image)
        self.count = self.count + 1

        return Image, Label
