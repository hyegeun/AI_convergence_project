# https://github.com/htdt/hyp_metric/blob/master/proxy_anchor/dataset/Inshop.py
from .base import *

import numpy as np, os, sys, pandas as pd, csv, copy
import torch
import torchvision
import PIL.Image
import glob


class Inshop_Dataset(torch.utils.data.Dataset):
    def __init__(self, root, mode, transform = None):
        self.root = 'unicom/' + root + '/aiproject'
        # self.root = root + '/aiproject'
        self.mode = mode
        self.transform = transform
        self.train_ys, self.train_im_paths = [], []
        self.query_ys, self.query_im_paths = [], []
        self.gallery_ys, self.gallery_im_paths = [], []

        print(self.root)
        train_list = glob.glob(f"{os.path.join(self.root, 'image_train')}/*.jpg")
        # query_list = glob.glob(f"{os.path.join(self.root, 'query_dir')}/*.jpg")
        query_list = glob.glob(f"{os.path.join('./', 'image_query')}/*.jpg")
        gallery_list = glob.glob(f"{os.path.join(self.root, 'test_dir')}/*.jpg")

        for item in train_list:
            self.train_im_paths.append(item)
            img_file = item.split('/')[-1]
            self.train_ys.append(int(img_file.split('_')[0]))
        
        for item in query_list:
            self.query_im_paths.append(item)
            img_file = item.split('/')[-1]
            self.query_ys.append(int(img_file.split('.')[0]))
            # self.query_ys.append(img_file.split('.')[0])
    
        for item in gallery_list:
            self.gallery_im_paths.append(item)
            img_file = item.split('/')[-1]
            # self.gallery_ys.append(int(img_file.split('_')[0]))
            self.gallery_ys.append(img_file.split('.')[0])

        if self.mode == 'train':
            self.im_paths = self.train_im_paths
            self.ys = self.train_ys
        elif self.mode == 'query':
            self.im_paths = self.query_im_paths
            self.ys = self.query_ys
        elif self.mode == 'gallery':
            self.im_paths = self.gallery_im_paths
            self.ys = self.gallery_ys

    def nb_classes(self):
        return len(set(self.ys))
            
    def __len__(self):
        return len(self.ys)
            
    def __getitem__(self, index):
        
        def img_load(index):
            im = PIL.Image.open(self.im_paths[index])
            # convert gray to rgb
            if len(list(im.split())) == 1 : im = im.convert('RGB') 
            if self.transform is not None:
                im = self.transform(im)
            return im
        
        im = img_load(index)
        target = self.ys[index]

        return im, target