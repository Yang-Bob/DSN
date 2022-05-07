import os
import torch
import random
import pickle
import PIL.Image as Image
import numpy as np
from config import settings

random.seed(123)

class NC_CUB200():
    def __init__(self, args, transform=None, c_way=5, k_shot=5):
        self.name = 'NC_CUB200'
        self.Datasets_dir = settings.CUB200_Datasets_Dir
        self.IndexDir = os.path.join(settings.CUB200_Index_DIR,args.seed)
        self.transform = transform
        self.count = 0
        self.K_shot = k_shot
        self.Set_Session(args)

    def Set_Session(self, args):
        self.sess = args.sess
        self.img, self.label = self.Read_Index_Sess()
        self.len = len(self.img)
        #print(len(self.img))

    def get_data(self):
        return self.img, self.label

    def Read_Index_Sess(self):
        idx = []
        label = []
        image = []
        f = open(self.IndexDir + '/session_' + str(self.sess + 1) + '.txt', 'r')
        while True:
            lines = f.readline()
            if not lines:
                break
            id, l = lines.split()
            idx.append(id)
            label.append(int(l)-1)

            img = Image.open(os.path.join(self.Datasets_dir, id))
            img = np.array(img)
            if len(img.shape) == 2:
                img = np.stack([img]*3, 2)
            image.append(img)

        if self.sess>0:
            idx_np = np.array(idx)
            label_np = np.array(label)
            img_np = np.array(image)

            label_sample = []
            img_sample = []

            cls = np.unique(label)
            for cl in cls:
                img_sample = img_sample + (random.sample(list(img_np[label_np == cl]), self.K_shot))
                label_sample = label_sample + ([cl] * self.K_shot)
            return img_sample, label_sample
        return image, label


    def load_frame(self, idx):
        img = self.img[idx]
        img = Image.fromarray(img, mode='RGB')
        Label = self.label[idx]

        return img, Label

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        Image, Label = self.load_frame(idx)
        if self.transform is not None:
            Image = self.transform(Image)
        self.count = self.count + 1

        return Image, Label
