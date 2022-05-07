import os
import json
import argparse
import time
import pandas as pd
import numpy as np
import copy

import torch
import torch.optim as optim
from config import settings
from data.LoadData_change import data_loader
from data.LoadData_change import val_loader
from models import *
from utils import Restore
from utils.Counting import Counting_train
from exemplar import BatchData
from exemplar import Exemplar
from exemplar import Distribution
from torch.utils.data import DataLoader

import collections
from collections import Counter

decay_epoch = [1000]
decay = 0.5


def get_arguments():
    parser = argparse.ArgumentParser(description='Incremental')
    parser.add_argument("--sesses", type=int, default='10', help='0 is base train, incremental from 1,2,3,...,8')
    parser.add_argument("--start_sess", type=int, default='1')
    parser.add_argument("--max_epoch", type=int, default='100')  # 180
    parser.add_argument("--batch_size", type=int, default='128')
    parser.add_argument("--dataset", type=str, default='CUB200')
    parser.add_argument("--arch", type=str, default='DSN')  #
    parser.add_argument("--lr", type=float, default=0.08)  # 0.005 0.002
    parser.add_argument("--r", type=float, default=0.1)  # 0.01
    parser.add_argument("--gamma", type=float, default=0.6)  # 0.01
    parser.add_argument("--lamda", type=float, default=1.0)  # 0.01
    parser.add_argument("--seed", type=str, default='Seed_1')  # 0.01 #Seed_1
    parser.add_argument("--gpu", type=str, default='5')
    parser.add_argument("--pretrained", type=str, default='False')
    parser.add_argument("--label_num", type=int, default='200')
    parser.add_argument("--base_num", type=int, default='100')  # 180
    parser.add_argument("--inc_len", type=int, default='10')
    parser.add_argument("--DS", type=str, default='True', help='Distribution Support')
    parser.add_argument("--delay_estimation", type=int, default='20')
    parser.add_argument("--delay_testing", type=int, default='3')
    parser.add_argument("--newsample_num", type=int, default=2)
    parser.add_argument("--oldsample_num_min", type=int, default=3)
    parser.add_argument("--basesample_num_min", type=int, default=3)
    parser.add_argument("--top_k", type=int, default='1')
    parser.add_argument("--sample_k", type=int, default='1')
    # parser.add_argument("--decay_epoch", nargs='+', type=int, default=[50])

    return parser.parse_args()


def test(args, network, val_data):
    TP = 0.0
    All = 0.0
    network.eval()
    val_data.dataset.Update_Session(0)
    for i, data in enumerate(val_data):
        img, label = data
        img, label = img.cuda(), label.cuda()
        out, output = network(img, sess=args.sess, Mode='test')
        _, pred = torch.max(output, dim=1)
        TP += torch.eq(pred, label).sum().float().item()
        All += torch.eq(label, label).sum().float().item()

    acc = float(TP) / All
    network.train()
    return acc


def test_continue(args, network,val_data):
    acc_list = []
    network.eval()
    for sess in range(args.sess + 1):
        TP = 0.0
        All = 0.0
        val_data.dataset.Update_Session(sess)
        for i, data in enumerate(val_data):
            img, label = data
            img, label = img.cuda(), label.cuda()
            out, output = network(img, args.sess, Mode='test')

            _, pred = torch.max(output, dim=1)
            TP += torch.eq(pred, label).sum().float().item()
            All += torch.eq(label, label).sum().float().item()

        acc = float(TP) / All
        acc_list.append(acc)
    network.train()
    return acc_list


def acc_list2string(acc_list):
    acc_str = ''
    for idx, item in enumerate(acc_list):
        acc_str += 'Sess%d: %.4f, ' % (idx, item)

    return acc_str


def Trans_ACC(args, acc_list):
    if args.dataset == 'CUB200':
        SessLen = settings.CUB200_SessLen
    if args.dataset == 'CIFAR100':
        SessLen = settings.CIFAR100_SessLen
    if args.dataset == 'miniImageNet':
        SessLen = settings.miniImagenet_SessLen
    ACC = 0
    ACC_A = 0 #new session
    ACC_M = 0 #old session
    num = 0
    for idx, acc in enumerate(acc_list):
        ACC += acc * SessLen[idx]
        num += SessLen[idx]
        if idx == args.sess:
            ACC_A += acc
        else:
            ACC_M += acc * SessLen[idx]
    ACC = ACC / num
    ACC_M = ACC_M / (num - SessLen[idx])
    return ACC, ACC_A, ACC_M


def freeze_model(model):
    for param in model.parameters():
        param.requires_grad = False
    return model


def extract_feature(data_loader, model):
    feature_dict = collections.defaultdict(list)
    model.eval()

    for i, (x, y) in enumerate(data_loader):
        x = x.cuda()
        y = y.cuda()
        with torch.no_grad():
            outputs = model.get_feature(x)
        tmp = outputs.clone().detach()
        for out, label in zip(tmp, y):
            x = out.cpu().numpy()
            feature_dict[label.item()].append(x)
    model.train()
    return feature_dict

def init_feature_space(args, network):
    exemplar = Exemplar(args)
    args.sess = 0
    img_train, input_transform = data_loader(args)
    train_x, train_y = img_train.get_data()
    base_loader = DataLoader(BatchData(args, train_x, train_y, input_transform), batch_size=128, shuffle=True,
                             num_workers=8)
    base_feature = extract_feature(base_loader, network)
    base_mean = {}
    base_cov = {}
    memory_mean = {}
    memory_cov = {}
    for key in base_feature.keys():
        feature = np.array(base_feature[key])
        mean = np.mean(feature, axis=0)
        cov = np.cov(feature.T)
        memory_mean[key] = mean
        memory_cov[key] = cov
        base_mean[key] = mean
        base_cov[key] = cov
    exemplar.update(memory_mean, memory_cov)

    return exemplar, base_mean,base_cov

def update_feature_space(args,network, exemplar, init=False):
    img_train, img_transform = data_loader(args)
    train_x, train_y = img_train.get_data()
    train_loader = DataLoader(BatchData(args, train_x, train_y, input_transform=img_transform), batch_size=128,
                              shuffle=True, num_workers=8)
    dataset_len = train_loader.dataset.__len__()

    # new session's distribution init
    sess_loader = DataLoader(BatchData(args, train_x, train_y, img_transform), batch_size=128, shuffle=True,
                             num_workers=8)
    sess_feature = extract_feature(sess_loader, network)
    train_x = []
    train_y = []
    memory_mean = {}
    memory_cov = {}
    for key in sess_feature.keys():
        feature = np.array(sess_feature[key])
        mean = np.mean(feature, axis=0)
        cov = np.cov(feature.T)
        memory_mean[key] = mean
        memory_cov[key] = cov

        for i in range(len(sess_feature[key])):
            train_x.append(sess_feature[key][i])
            train_y.append(key)

    # new session's distribution save
    exemplar.update(memory_mean, memory_cov)
    exemplar.memory_lidx = args.base_num + args.inc_len * (args.sess - 1)
    if init and args.DS=='True':
        exec('network.fc_aux' + str(args.sess + 2) + '.weight.data.copy_(network.fc1.weight.data)')
        temp = np.zeros((args.inc_len, 512))
        for key in memory_mean.keys():
            temp[key-args.base_num-args.sess*args.inc_len] = memory_mean[key]
        fea = torch.tensor(temp).cuda().to(torch.float32)

        # initialize classifier
        fea = network._l2norm(network.fc1(fea), dim=1)
        exec('network.fc' + str(args.sess + 2) + '.weight.data.copy_(fea.data)')

    return exemplar, train_x, train_y, memory_mean,memory_cov, dataset_len

def train(args):
    if args.dataset == 'CUB200':
        args.label_num = 200
        args.base_num = 100
        args.inc_len = 10
    else:
        args.label_num = 100
        args.base_num = 60
        args.inc_len = 5

    ACC_list = []

    lr = args.lr
    network = eval(args.arch).OneModel(args)  # fc:fc1  fw:sess-1 fc

    network.cuda()
    network_Old = eval(args.arch).OneModel(args)  # OLD NETWORK
    network_Old.cuda()
    best_model = eval(args.arch).OneModel(args)
    best_model.cuda()
    # optimizer = optim.SGD(network.parameters(), lr=lr, momentum=0.9, weight_decay=0.0005, nesterov=True)
    # use following
    # optimizer = optim.SGD(network.parameters(), lr=lr, momentum=0.9, dampening=0.5, weight_decay=0)

    print(network)

    log_dir = os.path.join('./log', args.dataset, args.arch, args.seed)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    args.sess = 0
    val_data = val_loader(args)
    if args.start_sess > 0:
        Restore.load(args, network, filename='Sess%d' % (args.start_sess - 1) + '.pth.tar')
        exemplar, base_mean, base_cov = init_feature_space(args, network)
        args.sess = args.start_sess - 1
        ACC = test(args, network, val_data)
        ACC_list.append(ACC)
        print('Sess: %d' % args.sess, 'acc_val: %f' % ACC)

    # Initialize feature space
    begin_time = time.time()
    best_model.load_state_dict(network.state_dict())
    for sess in range(args.start_sess, args.sesses + 1):
        args.sess = sess
        # Restore.load(args, network, filename='Sess%d' % (sess - 1) + '.pth.tar')
        network.load_state_dict(best_model.state_dict())
        network_Old.load_state_dict(network.state_dict())
        network_Old = freeze_model(network_Old)
        network_Old.eval()

        param_list1 = eval('network.fc'+str(args.sess + 2)+'.parameters()')
        param_list2 = eval('network.fc_aux'+str(args.sess + 2)+'.parameters()')
        optimizer = optim.SGD([{"params":param_list1},{"params":param_list2}], lr=lr, momentum=0.9, dampening=0.5, weight_decay=0)

        # Update feature space
        exemplar, train_x, train_y,memory_mean,memory_cov, dataset_len = update_feature_space(args,network, exemplar,True)
        Best_ACC = 0

        for epoch in range(args.max_epoch):
            if epoch % args.delay_estimation == 0:
                exemplar, train_x, train_y, memory_mean, memory_cov, dataset_len = update_feature_space(args, network, exemplar)
            if args.DS=='True':
                # memory sample
                if epoch % args.delay_estimation==0: #delay updating samples
                    train_xs, train_ys = exemplar.get_exemplar_train()
                    print(len(train_ys))
                    train_xs.extend(train_x)
                    train_ys.extend(train_y)
            else:
                train_xs = train_x
                train_ys = train_y

            train_loader = DataLoader(BatchData(args, train_xs, train_ys, input_transform=None, IOF='feature'),
                                      batch_size=args.batch_size, shuffle=True, num_workers=8)

            # statistic
            distribution = Distribution()
            counting_train = Counting_train(args)

            if epoch in decay_epoch:
                for param_group in optimizer.param_groups:
                    param_group['lr'] = param_group['lr'] * decay
            for i, data in enumerate(train_loader):
                img_tmp, label_tmp = data
                img_tmp = img_tmp.to(torch.float32)
                img_tmp, label_tmp = img_tmp.cuda(), label_tmp.cuda()

                img_old = torch.zeros(0, img_tmp.shape[1]).to(torch.float32).cuda()
                label_old = torch.zeros(0).long().cuda()
                img_new = torch.zeros(0, img_tmp.shape[1]).to(torch.float32).cuda()
                label_new = torch.zeros(0).long().cuda()

                for i in range(len(label_tmp)):
                    if (label_tmp[i].item() < exemplar.memory_lidx) and (label_tmp[i].item() >= 0):
                        img_old = torch.cat([img_old, img_tmp[i].unsqueeze(dim=0)], dim=0)
                        label_old = torch.cat([label_old, label_tmp[i]], dim=0)
                    else:
                        img_new = torch.cat([img_new, img_tmp[i].unsqueeze(dim=0)], dim=0)
                        label_new = torch.cat([label_new, label_tmp[i]], dim=0)

                label_tmp = torch.cat([label_old, label_new], dim=0)
                label = torch.abs(label_tmp)
                img = torch.cat([img_old, img_new], dim=0)

                # _, output = network(img, args.sess, epoch, IOF='feature')
                # Change Here
                Compression=True
                if img_new.shape[0] != 0:
                    _, output_newimg = network(img_new, args.sess, epoch, IOF='feature')
                    output = output_newimg
                if img_old.shape[0] != 0:
                    _, output_oldimg = network(img_old, args.sess, epoch, Mode='test', IOF='feature')
                    if img_new.shape[0] == 0:
                        output = output_oldimg
                        Compression=False
                    else:
                        output = torch.cat([output_oldimg, output_newimg], dim=0)
                #
                with torch.no_grad():
                    _, outputold= network_Old(img, args.sess - 1, epoch, Mode='test', IOF='feature')
                _, pred = torch.max(output, dim=1)
                loss = network.get_loss(16.0*output, label, 16.0*outputold.detach(), compression=Compression)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                network.finish_train()
                # statistic
                if args.DS=='True':
                    distribution.statistic(args, pred, label, label_tmp, output, exemplar.memory_lidx)
                counting_train.count(pred, label)


            # Updating distribution statistic
            if args.DS=='True':
                # statistic cov
                memory_cov = distribution.statistic_cov(args, exemplar.memory_lidx, memory_cov, base_cov)
                exemplar.update(memory_mean, memory_cov)
                # statisitc sample
                distribution.statisitc_sample(args, exemplar, args.top_k)

            # Train accuracy
            Train_ACC_Sess = counting_train.get_train_acc()
            Train_ACC_Sess_str = acc_list2string(Train_ACC_Sess)

            # information
            Time = time.time()
            p_st_1 = 'Training--' + ' Sess: %d' % args.sess + ' epoch: %d' % epoch + '                  ' + '%s' % Train_ACC_Sess_str+ ' Time cost: %.2fm ' %((Time - begin_time)/60)
            print(p_st_1)

            # Test accuracy
            if epoch>=0:#args.max_epoch/args.delay_testing:
                ACC_Sess = test_continue(args, network, val_data)
            else:
                ACC_Sess=[0]*(args.sess+1)
            ACC_Sess_str = acc_list2string(ACC_Sess)
            ACC, ACC_A, ACC_M = Trans_ACC(args, ACC_Sess)

            # information
            Time = time.time()
            p_st_2 = 'Testing--' + ' Sess: %d' % args.sess + ' epoch: %d' % epoch + ' acc_val: %f' % ACC + ' %s' % ACC_Sess_str + ' Time cost: %.2fm ' %((Time - begin_time)/60)+ '\n'
            print(p_st_2)

            '''if (ACC > Best_ACC) and (ACC_A >= 0.2):'''
            if Best_ACC <= ACC:
                Best_ACC = ACC
                Best_st = p_st_2
                best_mean = memory_mean
                best_cov = memory_cov
                Restore.save_model(args, network, filename='.pth.tar')
                best_model.load_state_dict(network.state_dict())

        network.finish_train()
        # Restore.save_model(args, network, filename='.pth.tar')
        with open(log_dir + '/log' + args.gpu + '.txt', 'a') as file_obj:
            file_obj.write(Best_st)

        ACC_list.append(Best_ACC)
        Best_ACC_Sess_str = acc_list2string(ACC_list)
        print('best acc:%s' %Best_ACC_Sess_str)
        exemplar.update(best_mean, best_cov)
#         args.max_epoch = args.max_epoch + 10

    timestamp = time.strftime("%m%d-%H%M", time.localtime())
    print('ACC:', ACC_list)
    print('End')


if __name__ == '__main__':
    args = get_arguments()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    print('Running parameters:\n')
    print(json.dumps(vars(args), indent=4, separators=(',', ':')))
    if args.dataset=='CUB200':
        args.sesses=10
    else:
        args.sesses=8
    train(args)
