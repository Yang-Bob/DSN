import numpy as np

class Counting_train():
    def __init__(self, args):
        self.all_label = np.zeros(20)
        self.hit_label = np.zeros(20)
        self.base_num = args.base_num
        self.inc_len = args.inc_len
        self.sess = args.sess
    def count(self, pred, label):
        for k in range(len(pred)):
            if label[k].item() < self.base_num:
                k_idx = 0
            else:
                k_idx = (label[k].item() - self.base_num) // self.inc_len + 1

            self.all_label[k_idx] = self.all_label[k_idx] + 1
            if pred[k].item() == label[k].item():
                self.hit_label[k_idx] = self.hit_label[k_idx] + 1
    def get_train_acc(self):
        Train_ACC_Sess = []
        for i in range(self.sess + 1):
            if self.all_label[i] != 0:
                tmp = self.hit_label[i] / self.all_label[i]
                Train_ACC_Sess.append(tmp)
            else:
                Train_ACC_Sess.append(0.0)
        return Train_ACC_Sess
