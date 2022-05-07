import os
import PIL.Image as Image
import numpy as np
from config import settings


class NC_CUB200_val():
    def __init__(self, args, transform=None):
        self.name = 'NC_CUB200'
        self.Datasets_dir = settings.CUB200_Datasets_Dir
        self.IndexDir = os.path.join(settings.CUB200_Index_DIR, args.seed)
        self.transform = transform
        # self.count = 0
        self.image = {}
        self.label = {}
        self.len ={}
        self.Set_Session(args)


    def Set_Session(self, args):
        self.sess = args.sess
        for sess in range(self.sess + 1):
            self.Update_Session(sess)

    def Update_Session(self, sess):
        self.sess = sess

        if self.sess not in self.label.keys():
            Index_list, label = self.Read_Index_Sess(sess)
            self.label[sess] = label
            self.len[sess] = len(Index_list)
        if self.sess not in self.image.keys():
            self.image[self.sess]=[]
            for Index in Index_list:
                img = Image.open(os.path.join(self.Datasets_dir, Index))
                img = np.array(img)
                if len(img.shape) == 2:
                    img = np.stack([img] * 3, 2)
                img = Image.fromarray(img, mode='RGB')
                self.image[sess].append(img)

    def Read_Index_Sess(self, sess):
        idx = []
        label = []
        f = open(self.IndexDir + '/test_' + str(sess + 1) + '.txt', 'r')
        while True:
            lines = f.readline()
            if not lines:
                break
            id, l = lines.split()
            idx.append(id)
            label.append(int(l) - 1)
            # idx.append(int(lines.strip()))

        return idx, label

    def Random_choose(self):
        Index = np.random.choice(self.Index_list, 1, replace=False)[0]

        return Index

    def load_frame(self, idx):
        img = self.image[self.sess][idx]
        Label = self.label[self.sess][idx]

        return img, Label

    def __len__(self):
        return self.len[self.sess]

    def __getitem__(self, idx):
        Image, Label = self.load_frame(idx)
        if self.transform is not None:
            Image = self.transform(Image)

        return Image, Label
