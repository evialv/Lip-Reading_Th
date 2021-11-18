# encoding: utf-8
import os
import glob
import random
import numpy as np
import librosa
import sys

class MyDataset():
    def __init__(self, folds, path, preprocessing_func=None,annonation_direc=None):
        self.folds = folds
        self.path = path
        self.clean = 1 / 7.
        self.preprocessing_func = preprocessing_func
        self._annonation_direc = annonation_direc
        with open('../label_sorted.txt') as myfile:
            self.label_fp = myfile.read().splitlines()
        self.filenames = glob.glob(os.path.join(self.path, '*', self.folds, '*.npz'))
        self.list = {}
        for i, x in enumerate(self.filenames):
            target = x.split('/')[-3]
            for j, elem in enumerate(self.label_fp):
                if elem == target:
                    self.list[i] = [x]
                    self.list[i].append(j)
        print('Load {} part'.format(self.folds))


    def load_data(self, filename):
        try:
            if filename.endswith('npz'):
                return np.load(filename)['data']
            elif filename.endswith('mp4'):
                return librosa.load(filename, sr=16000)[0][-19456:]
            else:
                return np.load(filename)
        except IOError:
            print( "Error when reading file: {}".format(filename) )
            sys.exit()

    # def normalisation(self, inputs):
    #     inputs_std = np.std(inputs)
    #     if inputs_std == 0.:
    #         inputs_std = 1.
    #     return (inputs - np.mean(inputs))/inputs_std

    def __getitem__(self, idx):
        # noise_prop = (1-self.clean)/6.
        # temp = random.random()
        # if self.folds == 'train':
        #     if temp < noise_prop:
        #         self.list[idx][0] = self.list[idx][0][:36]+'NoisyAudio/-5dB/'+self.list[idx][0][42:]
        #     elif temp < 2 * noise_prop:
        #         self.list[idx][0] = self.list[idx][0][:36]+'NoisyAudio/0dB/'+self.list[idx][0][42:]
        #     elif temp < 3 * noise_prop:
        #         self.list[idx][0] = self.list[idx][0][:36]+'NoisyAudio/5dB/'+self.list[idx][0][42:]
        #     elif temp < 4 * noise_prop:
        #         self.list[idx][0] = self.list[idx][0][:36]+'NoisyAudio/10dB/'+self.list[idx][0][42:]
        #     elif temp < 5 * noise_prop:
        #         self.list[idx][0] = self.list[idx][0][:36]+'NoisyAudio/15dB/'+self.list[idx][0][42:]
        #     elif temp < 6 * noise_prop:
        #         self.list[idx][0] = self.list[idx][0][:36]+'NoisyAudio/20dB/'+self.list[idx][0][42:]
        #     else:
        #         self.list[idx][0] = self.list[idx][0]
        # elif self.folds == 'val' or self.folds == 'test':
        #     self.list[idx][0] = self.list[idx][0]
        inputs = np.load(self.list[idx][0])['data']
        preprocess_data = self.preprocessing_func(inputs)

        labels = self.list[idx][1]
        return preprocess_data, labels

    def __len__(self):
        return len(self.filenames)

