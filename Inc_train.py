import os
import json
import argparse
import time
import numpy as np
import copy

import torch
import torch.optim as optim
from config import settings
from data.LoadData_change import data_loader
from data.LoadData_change import val_loader
from models import *
from utils import Restore
from exemplar import BatchData
from exemplar import Exemplar
from torch.utils.data import DataLoader

import collections
from collections import Counter

decay_epoch = [1000]
decay = 0.5


def get_arguments():
    parser = argparse.ArgumentParser(description='Incremental')
    parser.add_argument("--sesses", type=int, default='10', help='0 is base train, incremental from 1,2,3,...,8')
    parser.add_argument("--start_sess", type=int, default='1')
    parser.add_argument("--max_epoch", type=int, default='80')  # 180
    parser.add_argument("--batch_size", type=int, default='120')
    parser.add_argument("--dataset", type=str, default='CIFAR100')
    parser.add_argument("--arch", type=str, default='DSN')  #
    parser.add_argument("--lr", type=float, default=0.08)  # 0.005 0.002
    parser.add_argument("--r", type=float, default=0.1)  # 0.01
    parser.add_argument("--gamma", type=float, default=0.6)  # 0.01
    parser.add_argument("--lamda", type=float, default=1.0)  # 0.01
    parser.add_argument("--seed", type=str, default='Seed_3')  # 0.01 #Seed_1
    parser.add_argument("--gpu", type=str, default='4')
    parser.add_argument("--newsample_num", type=int, default=2)
    parser.add_argument("--pretrained", type=str, default='False')
    # parser.add_argument("--decay_epoch", nargs='+', type=int, default=[50])

    return parser.parse_args()


def test(args, network):
    TP = 0.0
    All = 0.0
    val_data = val_loader(args)
    network.eval()
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


def test_continue(args, network):
    val_data = val_loader(args)
    acc_list = []
    network.eval()
    for sess in range(args.sess + 1):
        TP = 0.0
        All = 0.0
        val_data.dataset.Update_Session(sess)
        for i, data in enumerate(val_data):
            img, label = data
            img, label = img.cuda(), label.cuda()

            if sess == 0:
                out, output = network(img, args.sess, Mode='test')
            else:
                out, output = network(img, args.sess, Mode='test', trans='Tukey')
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
        acc_str += 'Sess%d: %.3f, ' % (idx, item)

    return acc_str


def Trans_ACC(args, acc_list):
    if args.dataset == 'CUB200':
        SessLen = settings.CUB200_SessLen
    if args.dataset == 'CIFAR100':
        SessLen = settings.CIFAR100_SessLen
    if args.dataset == 'miniImageNet':
        SessLen = settings.miniImagenet_SessLen
    ACC = 0
    ACC_A = 0
    ACC_M = 0
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


def extract_feature(data_loader, model, trans='normal'):
    feature_dict = collections.defaultdict(list)
    model.eval()

    for i, (x, y) in enumerate(data_loader):
        x = x.cuda()
        y = y.cuda()
        outputs = model.get_feature(x, trans)
        tmp = outputs.clone().detach()

        for out, label in zip(tmp, y):
            x = out.cpu().numpy()
            feature_dict[label.item()].append(x)

    model.train()
    return feature_dict


def train(args):
    if args.dataset == 'CUB200':
        label_num = 200
        base_num = 100
        inc_len = 10
    if args.dataset == 'CIFAR100':
        label_num = 100
        base_num = 60
        inc_len = 5

    ACC_list = []
    ACC_list_train = []
    lr = args.lr
    network = eval(args.arch).OneModel(args)  # fc:fc1  fw:sess-1 fc
    network.cuda()
    network_Old = eval(args.arch).OneModel(args)  # OLD NETWORK
    network_Old.cuda()
    optimizer = optim.SGD(network.parameters(), lr=lr, momentum=0.9, weight_decay=0.0005, nesterov=True)
    print(network)

    log_dir = os.path.join('./log', args.dataset, args.arch, args.seed)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    if args.start_sess > 0:
        Restore.load(args, network, filename='Sess%d' % (args.start_sess - 1) + '.pth.tar')
        args.sess = args.start_sess - 1
        ACC = test(args, network)
        ACC_list.append(ACC)
        print('Sess: %d' % args.sess, 'acc_val: %f' % ACC)

    # memory
    exemplar = Exemplar(1, args, label_num, base_num)
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

    # statistic var
    memory_lidx = base_num - inc_len

    for sess in range(args.start_sess, args.sesses + 1):
        Restore.load(args, network, filename='Sess%d' % (sess - 1) + '.pth.tar')
        network_Old.load_state_dict(network.state_dict())
        network_Old = freeze_model(network_Old)
        network_Old.eval()
        args.sess = sess

        img_train, img_transform = data_loader(args)
        train_x, train_y = img_train.get_data()
        train_loader = DataLoader(BatchData(args, train_x, train_y, input_transform=img_transform), batch_size=128,
                                  shuffle=True, num_workers=8)
        dataset_len = train_loader.dataset.__len__()

        # statistic var update
        memory_lidx = memory_lidx + inc_len

        # new session's distribution init
        sess_loader = DataLoader(BatchData(args, train_x, train_y, img_transform), batch_size=128, shuffle=True,
                                 num_workers=8)
        sess_feature = extract_feature(sess_loader, network, 'Tukey')
        train_x = []
        train_y = []
        memory_mean = {}
        memory_cov = {}
        best_mean = {}
        best_cov = {}
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

        Best_ACC = 0
        loss_list = []
        begin_time = time.time()
        for epoch in range(args.max_epoch):
            # memory sample
            train_xs, train_ys = exemplar.get_exemplar_train(memory_lidx)
            print(len(train_ys))
            train_xs.extend(train_x)
            train_ys.extend(train_y)

            train_loader = DataLoader(BatchData(args, train_xs, train_ys, input_transform=None, IOF='feature'),
                                      batch_size=args.batch_size, shuffle=True, num_workers=8)

            # statistic
            all_label = torch.zeros(20)
            hit_label = torch.zeros(20)
            alpha_cov = collections.defaultdict(list)
            alpha_sample = collections.defaultdict(list)

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
                    if (label_tmp[i].item() < memory_lidx) and (label_tmp[i].item() >= 0):
                        img_old = torch.cat([img_old, img_tmp[i].unsqueeze(dim=0)], dim=0)
                        label_old = torch.cat([label_old, label_tmp[i]], dim=0)
                    else:
                        img_new = torch.cat([img_new, img_tmp[i].unsqueeze(dim=0)], dim=0)
                        label_new = torch.cat([label_new, label_tmp[i]], dim=0)

                label_tmp = torch.cat([label_old, label_new], dim=0)
                label = torch.abs(label_tmp)

                flag_old = True
                flag_new = True

                if label_new.shape[0] != 0:
                    _, output_new = network(img_new, args.sess, epoch, IOF='feature')
                    with torch.no_grad():
                        _, outputold_new = network_Old(img_new, args.sess - 1, epoch, Mode='test', IOF='feature')
                    _, pred_new = torch.max(output_new, dim=1)
                    flag_new = False

                if label_old.shape[0] != 0:
                    _, output_old = network(img_old, args.sess, epoch, Mode='train', IOF='feature')
                    with torch.no_grad():
                        _, outputold_old = network_Old(img_old, args.sess - 1, epoch, Mode='test', IOF='feature')
                    _, pred_old = torch.max(output_old, dim=1)
                    flag_old = False

                if flag_old:
                    output_old = torch.empty(0, output_new.shape[1]).cuda()
                    outputold_old = torch.empty(0, outputold_new.shape[1]).cuda()
                    pred_old = torch.zeros(0).long().cuda()

                if flag_new:
                    output_new = torch.empty(0, output_old.shape[1]).cuda()
                    outputold_new = torch.empty(0, outputold_old.shape[1]).cuda()
                    pred_new = torch.zeros(0).long().cuda()

                output = torch.cat([output_old, output_new], dim=0)
                outputold = torch.cat([outputold_old, outputold_new], dim=0)
                pred = torch.cat([pred_old, pred_new], dim=0)

                loss = network.get_loss(output, label, outputold.detach())
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                network.finish_train()

                # statistic
                for k in range(len(pred)):
                    if label[k].item() < base_num:
                        k_idx = 0
                    else:
                        k_idx = (label[k].item() - base_num) // inc_len + 1

                    all_label[k_idx] = all_label[k_idx] + 1
                    if pred[k].item() == label[k].item():
                        hit_label[k_idx] = hit_label[k_idx] + 1

                    # calculate cov's alpha and sample strategy
                    if (k_idx == sess) and (label_tmp[k].item() >= 0):
                        x = output[k].clone().detach()
                        tmp = torch.zeros(base_num + 1)
                        tmp[:base_num] = x[:base_num]
                        tmp[base_num] = x[label[k].item()]
                        tmp = torch.softmax(tmp, dim=0)
                        alpha_cov[label[k].item()].append(tmp.cpu().numpy())

                        tmp2 = torch.zeros(memory_lidx + 1)
                        tmp2[:memory_lidx] = x[:memory_lidx]
                        tmp2[memory_lidx] = x[label[k].item()]
                        tmp2 = torch.softmax(tmp2, dim=0)
                        alpha_sample[label[k].item()].append(tmp2.cpu().numpy())

                acc = torch.eq(pred, label).sum().float().item() / torch.eq(label, label).sum().float().item()
                all_step = int((dataset_len / args.batch_size))
                Time = time.time()
                print('Training--', 'Sess: %d' % args.sess, 'epoch: %d' % epoch, 'step: %d/%d' % (i, all_step),
                      'loss: %f' % loss, 'ACC_val: %f' % ACC, 'acc_train: %f' % acc,
                      'Time: %f' % ((Time - begin_time) / 60))
            ACC_Sess = test_continue(args, network)
            ACC_Sess_str = acc_list2string(ACC_Sess)
            ACC, ACC_A, ACC_M = Trans_ACC(args, ACC_Sess)
            loss_list.append(loss.data.item())

            # statistic cov
            for i in range(memory_lidx, memory_lidx + inc_len):
                alpha_tmp = np.array(alpha_cov[i])
                alpha_tmp = np.mean(alpha_tmp, axis=0)
                # import pdb;pdb.set_trace()
                memory_cov[i] = memory_cov[i] * alpha_tmp[base_num]
                for j in range(base_num):
                    memory_cov[i] = memory_cov[i] + base_cov[j] * alpha_tmp[j]

            exemplar.update(memory_mean, memory_cov)

            # statisitc sample
            for i in range(memory_lidx, memory_lidx + inc_len):
                sample_tmp = np.array(alpha_sample[i])
                sample_tmp = np.mean(sample_tmp, axis=0)
                sample_val, sample_idx = torch.topk(torch.from_numpy(sample_tmp[:memory_lidx]), 1)
                for j in range(1):
                    if sample_val[j] >= max(0, sample_tmp[memory_lidx] - 0.02):
                        exemplar.p[sample_idx[j]] = exemplar.p[sample_idx[j]] + 1

            Train_ACC_Sess = []
            for i in range(sess + 1):
                if all_label[i].long().item() != 0:
                    tmp = hit_label[i] / all_label[i]
                    Train_ACC_Sess.append(tmp)
                else:
                    Train_ACC_Sess.append(0.0)

            Train_ACC_Sess_str = acc_list2string(Train_ACC_Sess)
            p_st_1 = 'Training--' + ' Sess: %d' % args.sess + ' epoch: %d' % epoch + '                  ' + '%s' % Train_ACC_Sess_str
            print(p_st_1)

            p_st_2 = 'Testing--' + ' Sess: %d' % args.sess + ' epoch: %d' % epoch + ' acc_val: %f' % ACC + ' %s' % ACC_Sess_str + '\n'
            print(p_st_2)

            '''if (ACC > Best_ACC) and (ACC_A >= 0.2):'''
            if ACC_A > ACC_M:
                if Best_ACC <= ACC:
                    Best_ACC = ACC
                    Best_st = p_st_2
                    best_mean = memory_mean
                    best_cov = memory_cov
                    Restore.save_model(args, network, filename='.pth.tar')

        network.finish_train()
        # Restore.save_model(args, network, filename='.pth.tar')
        with open(log_dir + '/log' + args.gpu + '.txt', 'a') as file_obj:
            file_obj.write(Best_st)

        ACC_list.append(ACC)
        ACC_list_train.append(acc)
        exemplar.update(best_mean, best_cov)
        args.max_epoch = args.max_epoch + 10

    timestamp = time.strftime("%m%d-%H%M", time.localtime())
    print('ACC:', ACC_list)
    print('End')


if __name__ == '__main__':
    args = get_arguments()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    print('Running parameters:\n')
    print(json.dumps(vars(args), indent=4, separators=(',', ':')))
    train(args)
