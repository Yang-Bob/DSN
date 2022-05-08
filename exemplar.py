import random
import torch
import numpy as np
from torch.utils.data import Dataset
import PIL.Image as Image
import collections
from collections import Counter
import time
from multiprocessing.dummy import Pool as ThreadPool
from threading import Thread


class MyThread(Thread):
    def __init__(self, func, args):
        Thread.__init__(self)
        self.func = func
        self.args = args
        self.result = self.func(self.args)

    def get_result(self):
        return self.result

class Exemplar:
    def __init__(self, args):
        self.mean = {}
        self.cov = {}
        self.val = {}
        self.dataset = args.dataset
        self.k_shot = args.sample_k
        self.base_num = args.base_num
        self.p = torch.ones(args.label_num).long() * self.k_shot
        self.newsample_num = args.newsample_num
        self.oldsample_num_min = args.oldsample_num_min
        self.basesample_num_min=args.basesample_num_min


    def update(self, memory_mean, memory_cov):
        lam = 0.8
        for key in memory_mean.keys():

            if key not in self.mean.keys():
                self.mean[key] = memory_mean[key]
                self.cov[key] = memory_cov[key]
            else:
                self.mean[key] = lam*self.mean[key]+(1-lam)*memory_mean[key]
                self.cov[key] = lam*self.cov[key]+(1-lam)*memory_cov[key]

    def get_exemplar_train(self):
        # exemplar_feature, exemplar_label = self.multi_process_sampling()
        exemplar_feature, exemplar_label = self.multi_thread_sampling()
        # exemplar_feature, exemplar_label = self.general_sampling()
        return exemplar_feature, exemplar_label
    
    def sampling(self, key):
        exemplar_feature = []
        exemplar_label = []
        ger_mean = self.mean[key]
        ger_cov = self.cov[key]
        if key >= self.memory_lidx:
            ger_num = self.newsample_num
        elif (key < self.memory_lidx) and (key >= self.base_num):
            if self.dataset == 'CUB200':
                ger_num = min(self.oldsample_num_min, self.p[key].item())
            else:
                ger_num = min(self.oldsample_num_min, self.p[key].item())
        else:
            if self.dataset == 'CUB200':
                ger_num = min(self.basesample_num_min, self.p[key].item())
            else:
                ger_num = min(self.basesample_num_min, self.p[key].item())
        ger_feature = np.random.multivariate_normal(mean=ger_mean, cov=ger_cov, size=ger_num)
        ger_feature = np.float32(ger_feature)

        for i in range(ger_num):
            exemplar_feature.append(ger_feature[i].squeeze())

            if key >= self.memory_lidx:
                exemplar_label.append(-1 * key)
            else:
                exemplar_label.append(key)

        self.p[key] = self.k_shot
        return exemplar_feature, exemplar_label

    def multi_process_sampling(self,processes=5):
        exemplar_feature = []
        exemplar_label = []
        pool = ThreadPool(processes=processes)
        # pfunc = partial(func, param1)
        out  = pool.map(self.sampling, self.mean.keys())
        for out_iter in out:
            exemplar_feature.extend(out_iter[0])
            exemplar_label.extend(out_iter[1])
        pool.close()
        pool.join()

        return exemplar_feature, exemplar_label

    def general_sampling(self):
        exemplar_feature = []
        exemplar_label = []
        for key in self.mean.keys():
            out = self.sampling(key)
            exemplar_feature.extend(out[0])
            exemplar_label.extend(out[1])
        return exemplar_feature, exemplar_label

    def multi_thread_sampling(self):
        exemplar_feature = []
        exemplar_label = []
        threads = []
        # key_len = len(self.mean.keys())
        for key in self.mean.keys():
            t = MyThread(self.sampling, key)
            t.start()
            threads.append(t)
        for t in threads:
            t.join()
        for t in threads:
            out = t.get_result()
            exemplar_feature.extend(out[0])
            exemplar_label.extend(out[1])
        return exemplar_feature, exemplar_label

    def get_len(self):
        return len(self.train)


class BatchData(Dataset):
    def __init__(self, args, images, labels, input_transform=None, IOF='image'):
        self.images = images
        self.labels = labels
        self.input_transform = input_transform
        self.IOF = IOF
        self.args = args

    def __getitem__(self, index):
        image = self.images[index]
        label = self.labels[index]

        if self.IOF == 'image':
            if self.args.dataset != 'CIFAR100':
                image = Image.fromarray(image, mode='RGB')
            if self.input_transform is not None:
                image = self.input_transform(image)
        label = torch.LongTensor([label])
        return image, label

    def __len__(self):
        return len(self.images)

class Distribution:
    def __init__(self):
        self.all_label = torch.zeros(20)
        self.hit_label = torch.zeros(20)
        self.alpha_cov = collections.defaultdict(list)
        self.alpha_sample = collections.defaultdict(list)
    
    def statistic(self, args, pred, label, label_tmp, output, memory_lidx):
        for k in range(len(pred)):
            if label[k].item() < args.base_num:
                k_idx = 0
            else:
                k_idx = (label[k].item() - args.base_num) // args.inc_len + 1

            # calculate cov's alpha and sample strategy
            if (k_idx == args.sess) and (label_tmp[k].item() >= 0):
                x = output[k].clone().detach()
                tmp = torch.zeros(args.base_num + 1)
                tmp[:args.base_num] = x[:args.base_num]
                tmp[args.base_num] = x[label[k].item()]
                tmp = torch.softmax(tmp, dim=0)
                self.alpha_cov[label[k].item()].append(tmp.cpu().numpy())

                tmp2 = torch.zeros(memory_lidx + 1)
                tmp2[:memory_lidx] = x[:memory_lidx]
                tmp2[memory_lidx] = x[label[k].item()]
                tmp2 = torch.softmax(tmp2, dim=0)
                self.alpha_sample[label[k].item()].append(tmp2.cpu().numpy())
    
    def statistic_cov(self, args, memory_lidx, memory_cov, base_cov):
        for i in range(memory_lidx, memory_lidx + args.inc_len):
            alpha_tmp = np.array(self.alpha_cov[i])
            alpha_tmp = np.mean(alpha_tmp, axis=0)
            memory_cov[i] = memory_cov[i] * alpha_tmp[args.base_num]
            for j in range(args.base_num):
                memory_cov[i] = memory_cov[i] + base_cov[j] * alpha_tmp[j]
        return memory_cov
    
    def statisitc_sample(self, args, exemplar, k):
        for i in range(exemplar.memory_lidx, exemplar.memory_lidx + args.inc_len):
            sample_tmp = np.array(self.alpha_sample[i])
            sample_tmp = np.mean(sample_tmp, axis=0)
            sample_val, sample_idx = torch.topk(torch.from_numpy(sample_tmp[:exemplar.memory_lidx]), k)
            for j in range(k):
                if sample_val[j] >= max(0, sample_tmp[exemplar.memory_lidx]):
                    exemplar.p[sample_idx[j]] = exemplar.p[sample_idx[j]] + 1
