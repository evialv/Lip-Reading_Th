# coding: utf-8
import os
import time
import random
import logging
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets
from torch.autograd import Variable
from torchvision.transforms import Compose

from lr_scheduler import *
from model import *
from dataset import *

import multiprocessing as mp
import multiprocess as mul
# from audio_only.preprocessing import *
from preprocessing import *
import torch
# torch.cuda.empty_cache()

# try:
#     set_start_method('spawn')
# except RuntimeError:
#     pass

SEED = 1
torch.manual_seed(SEED)
# torch.cuda.manual_seed(SEED)
np.random.seed(SEED)


# class AddNoise(object):
#     """Add SNR noise [-1, 1]
#     """
#
#     def __init__(self, noise, snr_levels=[-5, 0, 5, 10, 15, 20, 9999]):
#         assert noise.dtype in [np.float32, np.float64], "noise only supports float data type"
#
#         self.noise = noise
#         self.snr_levels = snr_levels
#
#     def get_power(self, clip):
#         clip2 = clip.copy()
#         clip2 = clip2 **2
#         return np.sum(clip2) / (len(clip2) * 1.0)
#
#     def __call__(self, signal):
#         assert signal.dtype in [np.float32, np.float64], "signal only supports float32 data type"
#         snr_target = random.choice(self.snr_levels)
#         if snr_target == 9999:
#             return signal
#         else:
#             # -- get noise
#             start_idx = random.randint(0, len(self.noise)-len(signal))
#             noise_clip = self.noise[start_idx:start_idx+len(signal)]
#
#             sig_power = self.get_power(signal)
#             noise_clip_power = self.get_power(noise_clip)
#             factor = (sig_power / noise_clip_power ) / (10**(snr_target / 10.0))
#             desired_signal = (signal + noise_clip*np.sqrt(factor)).astype(np.float32)
#             return desired_signal


# class NormalizeUtterance():
#     """Normalize per raw audio by removing the mean and divided by the standard deviation
#     """
#     def __call__(self, signal):
#         signal_std = 0. if np.std(signal)==0. else np.std(signal)
#         signal_mean = np.mean(signal)
#         return (signal - signal_mean) / signal_std


# if torch.cuda.is_available():
#     torch.set_default_tensor_type('torch.cuda.FloatTensor')


def get_preprocessing_pipelines():
    # -- preprocess for the video stream
    preprocessing = {}
    # -- LRW config

    preprocessing['train'] = Compose([
        AddNoise(noise=np.load('../babbleNoise_resample_16K.npy')),
        NormalizeUtterance()])

    preprocessing['val'] = NormalizeUtterance()

    preprocessing['test'] = NormalizeUtterance()


    return preprocessing

def data_loader(args):
    preprocessing = get_preprocessing_pipelines()
    dsets = {x: MyDataset(x, args.dataset,preprocessing_func=preprocessing[x]) for x in ['train', 'val', 'test']}
    dset_loaders = {x: torch.utils.data.DataLoader(dsets[x], batch_size=args.batch_size, shuffle=True, num_workers=args.workers,generator=torch.Generator(device='cpu')) for x in ['train', 'val', 'test']}
    dset_sizes = {x: len(dsets[x]) for x in ['train', 'val', 'test']}
    print('\nStatistics: train: {}, val: {}, test: {}'.format(dset_sizes['train'], dset_sizes['val'], dset_sizes['test']))
    return dset_loaders, dset_sizes


def reload_model(model, logger, path=""):
    if not bool(path):
        logger.info('train from scratch')
        return model
    else:
        model_dict = model.state_dict()
        pretrained_dict = torch.load(path)
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
        logger.info('*** model has been successfully loaded! ***')
        return model


def showLR(optimizer):
    lr = []
    for param_group in optimizer.param_groups:
        lr += [param_group['lr']]
    return lr


def train_test(model, dset_loaders, criterion, epoch, phase, optimizer, args, logger, use_gpu, save_path):
    if phase == 'val' or phase == 'test':
        model.eval()
    if phase == 'train':
        model.train()
    if phase == 'train':
        logger.info('-' * 10)
        logger.info('Epoch {}/{}'.format(epoch, args.epochs - 1))
        logger.info('Current Learning rate: {}'.format(showLR(optimizer)))

    running_loss, running_corrects, running_all = 0., 0., 0.
    x=dset_loaders[phase]

    for batch_idx, (inputs, targets) in enumerate(dset_loaders[phase]):
        inputs = inputs.float()
        if use_gpu:
            if phase == 'train':
                inputs, targets = Variable(inputs.cuda()), Variable(targets.cuda())
            if phase == 'val' or phase == 'test':
                inputs, targets = Variable(inputs.cuda(), volatile=True), Variable(targets.cuda())
        else:
            if phase == 'train':
                inputs, targets = Variable(inputs), Variable(targets)
            if phase == 'val' or phase == 'test':
                inputs, targets = Variable(inputs, volatile=True), Variable(targets)
        outputs = model(inputs)
        if args.every_frame:
            outputs = torch.mean(outputs, 1)
        _, preds = torch.max(F.softmax(outputs, dim=1).data, 1)
        loss = criterion(outputs, targets)
        if phase == 'train':
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        # stastics
        running_loss += loss.data * inputs.size(0)
        running_corrects += torch.sum(preds == targets.data)
        running_all += len(inputs)
        if batch_idx == 0:
            since = time.time()
        elif batch_idx % args.interval == 0 or (batch_idx == len(dset_loaders[phase])-1):
            print('Process: [{:5.0f}/{:5.0f} ({:.0f}%)]\tLoss: {:.4f}\tAcc:{:.4f}\tCost time:{:5.0f}s\tEstimated time:{:5.0f}s\r'.format(
                running_all,
                len(dset_loaders[phase].dataset),
                100. * batch_idx / (len(dset_loaders[phase])-1),
                running_loss / running_all,
                running_corrects / running_all,
                time.time()-since,
                (time.time()-since)*(len(dset_loaders[phase])-1) / batch_idx - (time.time()-since))),
    print
    logger.info('{} Epoch:\t{:2}\tLoss: {:.4f}\tAcc:{:.4f}'.format(
        phase,
        epoch,
        running_loss / len(dset_loaders[phase].dataset),
        running_corrects / len(dset_loaders[phase].dataset))+'\n')
    if phase == 'train':
        torch.save(model.state_dict(), save_path+'/'+args.mode+'_'+str(epoch+1)+'.pt')
        return model


def test_adam(args, use_gpu):
    if args.every_frame and args.mode != 'temporalConv':
        save_path = '/mnt/hdd2/alvanaki/trained_audio/' + args.mode + '_every_frame'
    elif not args.every_frame and args.mode != 'temporalConv':
        save_path = '/mnt/hdd2/alvanaki/trained_audio/' + args.mode + '_last_frame'
    elif args.mode == 'temporalConv':
        save_path = '/mnt/hdd2/alvanaki/trained_audio/' + args.mode
    else:
        raise Exception('No model is found!')
    if not os.path.isdir(save_path):
        os.mkdir(save_path)
    # logging info
    filename = save_path+'/'+args.mode+'_'+str(args.lr)+'.txt'
    logger_name = "mylog"
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)
    fh = logging.FileHandler(filename, mode='a')
    fh.setLevel(logging.INFO)
    logger.addHandler(fh)
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    logger.addHandler(console)

    # torch.cuda.empty_cache()

    # framelen = 29
    framelen = 27

    model = lipreading(mode=args.mode, inputDim=512, hiddenDim=512, nClasses=args.nClasses, frameLen=framelen, every_frame=args.every_frame)
    # # reload model
    # model = reload_model(model, logger, args.path)
    # define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    if args.mode == 'temporalConv' or args.mode == 'finetuneGRU':
        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=0.)
    elif args.mode == 'backendGRU':
        for param in model.parameters():
            param.requires_grad = False
        for param in model.gru.parameters():
            param.requires_grad = True
        optimizer = optim.Adam([
            {'params': model.gru.parameters(), 'lr': args.lr}
        ], lr=0., weight_decay=0.)
    else:
        raise Exception('No model is found!')

    dset_loaders, dset_sizes = data_loader(args)
    scheduler = AdjustLR(optimizer, [args.lr], sleep_epochs=5, half=5, verbose=1)
    if args.test:
        train_test(model, dset_loaders, criterion, 0, 'val', optimizer, args, logger, use_gpu, save_path)
        train_test(model, dset_loaders, criterion, 0, 'test', optimizer, args, logger, use_gpu, save_path)
        return
    for epoch in range(args.epochs):
        scheduler.step(epoch)
        model = train_test(model, dset_loaders, criterion, epoch, 'train', optimizer, args, logger, use_gpu, save_path)
        train_test(model, dset_loaders, criterion, epoch, 'val', optimizer, args, logger, use_gpu, save_path)


def main():
    # Settings
    parser = argparse.ArgumentParser(description='Pytorch Audio-only BBC-LRW Example')
    parser.add_argument('--nClasses', default=500, type=int, help='the number of classes')
    parser.add_argument('--path', default='', help='path to model')
    parser.add_argument('--dataset', default='/mnt/hdd2/alvanaki/audio_dataset_full', help='path to dataset')
    parser.add_argument('--mode', default='temporalConv', help='temporalConv, backendGRU, finetuneGRU')
    parser.add_argument('--every-frame', default=False, action='store_true', help='predicition based on every frame')
    parser.add_argument('--lr', default=0.0003, type=float, help='initial learning rate')
    parser.add_argument('--batch-size', default=36, type=int, help='mini-batch size (default: 36)')
    parser.add_argument('--workers', default=0, type=int, help='number of data loading workers (default: 4)')
    parser.add_argument('--epochs', default=30, type=int, help='number of total epochs')
    parser.add_argument('--interval', default=10, type=int, help='display interval')
    parser.add_argument('--test', default=False, action='store_true', help='perform on the test phase')
    args = parser.parse_args()

    # os.environ['CUDA_VISIBLE_DEVICES'] = '1'
    # os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
    # use_gpu = torch.cuda.is_available()
    use_gpu=False
    test_adam(args, use_gpu)


if __name__ == '__main__':
    mp.set_start_method('spawn')
    main()
