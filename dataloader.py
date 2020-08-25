import os
from os.path import isdir, exists, abspath, join

import random
from random import randint
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image,ImageEnhance
from scipy.ndimage.interpolation import map_coordinates
from scipy.ndimage.filters import gaussian_filter
import torchvision

class DataLoader():
    def __init__(self, root_dir='data', batch_size=2, test_percent=.1):
        self.batch_size = batch_size
        self.test_percent = test_percent

        self.root_dir = abspath(root_dir)
        self.data_dir = join(self.root_dir, 'scans')
        self.labels_dir = join(self.root_dir, 'labels')

        self.files = os.listdir(self.data_dir)

        self.data_files = [join(self.data_dir, f) for f in self.files]
        self.label_files = [join(self.labels_dir, f) for f in self.files]

    def __iter__(self):
        n_train = self.n_train()

        if self.mode == 'train':
            current = 0
            endId = n_train
        elif self.mode == 'test':
            current = n_train
            endId = len(self.data_files)

        while current < endId:


            # todo: load images and labels
            # hint: scale images between 0 and 1
            # hint: if training takes too long or memory overflow, reduce image size!

            #get image
            data_image = Image.open(self.data_files[current])

            #crop the image
            data_resize = data_image.resize((572, 572))

            #data augmentation
            data_aug=self.applyDataAugmentation(data_resize)
            data_aug=np.divide(np.asarray(data_aug,dtype=np.float32),255.)

            #mirroring image
            img_crop = (3 * 572 - 572) // 2
            img_flip=np.flipud(data_aug)
            img_rot=np.rot90(data_aug,k=1,axes=(0,1))
            concat1=np.concatenate((np.flipud(img_rot),img_flip,np.flipud(img_rot)),axis=1)
            concat2 = np.concatenate((img_rot, data_aug, img_rot), axis=1)
            concat3 = np.concatenate((np.flipud(img_rot), img_flip, np.flipud(img_rot)), axis=1)
            concat_image=np.concatenate((concat1,concat2,concat3),axis=0)
            dim1,dim2=concat_image.shape
            data_aug=concat_image[img_crop:dim1-572, 572:dim2-572]

            #normalization
            data_aug = (data_aug - np.min(data_aug)) / (np.max(data_aug) - np.min(data_aug))*255

            #mask label
            mask_name = self.label_files[current]
            current += 1
            mask_img = Image.open(mask_name)

            #Crop
            label_image=mask_img.resize((388,388))
            label_image = self.applyDataAugmentation(label_image)
            label_image = np.asarray(label_image,dtype=np.float32)

            yield (data_aug, label_image)

    def setMode(self, mode):
        self.mode = mode

    def n_train(self):
        data_length = len(self.data_files)
        return np.int_(data_length - np.floor(data_length * self.test_percent))

    def applyDataAugmentation(self, img):

        # Add the Brightness for the image, add the Horizontal flip, Vertical Flip
        self.data_resize = torchvision.transforms.functional.adjust_hue(img, 0.5)
        self.data_resize = torchvision.transforms.functional.hflip(self.data_resize)
        self.data_resize = torchvision.transforms.functional.vflip(self.data_resize)
        return self.data_resize