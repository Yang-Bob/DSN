import os
import json
import argparse
import time

import torch
import torch.optim as optim

from data.LoadData import data_loader
from data.LoadData import val_loader
from utils import Log
from utils import Restore
from models import *
from config import settings

def get_arguments():
    parser = argparse.ArgumentParser(description='Incremental')
    parser.add_argument("--sesses", type=int, default='0', help='0 is base train, incremental from 1,2,3,...,8')
    parser.add_argument("--max_epoch", type=int, default='200')
    parser.add_argument("--batch_size", type=int, default='128')
    parser.add_argument("--dataset", type=str, default='CUB200')
    parser.add_argument("--arch", type=str, default='DSN', help='quickcnn, resnet')
    parser.add_argument("--lr", type=float, default=0.1)  # 0.1
    parser.add_argument("--r", type=float, default=15)
    parser.add_argument("--gamma", type=float, default=4)
    parser.add_argument("--seed", type=str, default='Seed_1')  # Seed_3
    parser.add_argument("--gpu", type=str, default='4')  #
    parser.add_argument("--pretrained", type=bool, default=False)
    parser.add_argument("--decay_epoch", type=int, nargs='+', default=[80, 120, 160])

    return parser.parse_args()


def test(args, network, val_data):
    TP = 0.0
    All = 0.0
    network.eval()
    for i, data in enumerate(val_data):
        img, label = data
        img, label = img.cuda(), label.cuda()
        with torch.no_grad():
            out, output = network(img, args.sess)
        _, pred = torch.max(output, dim=1)

        TP += torch.eq(pred, label).sum().float().item()
        All += torch.eq(label, label).sum().float().item()

    acc = float(TP) / All
    network.train()
    return acc


def train(args):
    lr = args.lr
    network = eval(args.arch).OneModel(args)
    print(network)
    network.cuda()
    optimizer = optim.SGD(network.parameters(), lr=lr, momentum=0.9, weight_decay=0.0005, nesterov=True)
    for sess in range(args.sesses + 1):
        args.sess = sess
        train_loader = data_loader(args)
        val_data = val_loader(args)
        dataset_len = train_loader.dataset.__len__()
        ACC = 0
        Best_ACC = 0
        ACC_list = []
        loss_list = []
        begin_time = time.time()
        for epoch in range(args.max_epoch):
            if epoch in args.decay_epoch:
                for param_group in optimizer.param_groups:
                    param_group['lr'] = param_group['lr'] * 0.1
            for i, data in enumerate(train_loader):
                img, label = data
                img, label = img.cuda(), label.cuda()
                out, output = network(img, args.sess)

                _, pred = torch.max(output, dim=1)

                loss = network.get_loss(16.0*output, label)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                acc = torch.eq(pred, label).sum().float().item() / torch.eq(label, label).sum().float().item()
                all_step = int((dataset_len / args.batch_size))
                Time = time.time()
                print('epoch: %d' % epoch, 'step: %d/%d' % (i, all_step), 'loss: %f' % loss, 'ACC_val: %f' % ACC,
                      'acc_train: %f' % acc, 'Time: %f' % ((Time - begin_time) / 60))
            ACC = test(args, network, val_data)
            ACC_list.append(ACC)
            loss_list.append(loss.data.item())
            if Best_ACC <= ACC:
                Best_ACC = ACC
                Restore.save_model(args, network, filename='.pth.tar')
                print('Update Best_ACC %f' % Best_ACC)
            print('epoch: %d' % epoch, 'acc_val: %f' % ACC)
            Log.log(args, ACC_list, 'acc', sup='Sess0')
            Log.log(args, loss_list, 'loss', sup='Sess0')
        # Restore.save_model(args, network, filename='.pth.tar')
    print('End')


if __name__ == '__main__':
    args = get_arguments()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    if args.dataset == 'CUB200':
        args.pretrained=True
    print('Running parameters:\n')
    print(json.dumps(vars(args), indent=4, separators=(',', ':')))
    train(args)
