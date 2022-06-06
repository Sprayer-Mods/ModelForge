# ModelForge by Sprayer Mods, GPL-3.0 license
"""
A redefinition of dataloaders.py with the sole purpose of creating a 
dataloader that can load sequential video frames.
"""
import glob
import hashlib
import json
import math
import os
import random
import shutil
import time
from itertools import repeat
from multiprocessing.pool import Pool, ThreadPool
from pathlib import Path
from threading import Thread
from urllib.parse import urlparse
from zipfile import ZipFile

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from PIL import ExifTags, Image, ImageOps
from torch.utils.data import DataLoader, Dataset, dataloader, distributed
from tqdm import tqdm

from utils.augmentations import Albumentations, augment_hsv, copy_paste, letterbox, mixup, random_perspective
from utils.general import (DATASETS_DIR, LOGGER, NUM_THREADS, check_dataset, check_requirements, check_yaml, clean_str,
                           cv2, segments2boxes, xyn2xy, xywh2xyxy, xywhn2xyxy, xyxy2xywhn)
from utils.torch_utils import torch_distributed_zero_first
from utils.dataloaders import *

HELP_URL = 'https://github.com/ultralytics/yolov5/wiki/Train-Custom-Data'
IMG_FORMATS = 'bmp', 'dng', 'jpeg', 'jpg', 'mpo', 'png', 'tif', 'tiff', 'webp'  # include image suffixes
VID_FORMATS = 'asf', 'avi', 'gif', 'm4v', 'mkv', 'mov', 'mp4', 'mpeg', 'mpg', 'ts', 'wmv'  # include video suffixes
BAR_FORMAT = '{l_bar}{bar:10}{r_bar}{bar:-10b}'  # tqdm bar format
LOCAL_RANK = int(os.getenv('LOCAL_RANK', -1))  # https://pytorch.org/docs/stable/elastic/run.html

def create_sequence_idxs(nt, 
                         index):
    """
    Creates a list of sequence indexes. Can be overlapping.
    Probability of overlapping sequences increases with smaller datasets.
    """
    return list(range(index, index+nt))


def create_seq_dataloader(path,
                        imgsz,
                        batch_size,
                        nt,
                        seq_batch,
                        stride,
                        single_cls=False,
                        hyp=None,
                        augment=False,
                        cache=False,
                        pad=0.0,
                        rect=False,
                        rank=-1,
                        workers=8,
                        prefix='',
                        shuffle=False):
    if rect and shuffle:
        LOGGER.warning('WARNING: --rect is incompatible with DataLoader shuffle, setting shuffle=False')
        shuffle = False

    with torch_distributed_zero_first(rank):  # init dataset *.cache only once if DDP
        dataset = LoadSeqAndLabels(
            path,
            imgsz,
            batch_size,
            nt,
            seq_batch=seq_batch,
            augment=augment,  # augmentation
            hyp=hyp,  # hyperparameters
            rect=rect,  # rectangular batches
            cache_images=cache,
            single_cls=single_cls,
            stride=int(stride),
            pad=pad,
            prefix=prefix)

    batch_size = min(batch_size, len(dataset))
    nd = torch.cuda.device_count()  # number of CUDA devices
    nw = min([os.cpu_count() // max(nd, 1), batch_size if batch_size > 1 else 0, workers])  # number of workers
    sampler = None if rank == -1 else distributed.DistributedSampler(dataset, shuffle=shuffle)
    loader = InfiniteSeqDataLoader
    return loader(dataset,
                  batch_size=batch_size,
                  shuffle=shuffle and sampler is None,
                  num_workers=nw,
                  sampler=sampler,
                  pin_memory=True,
                  collate_fn=LoadSeqAndLabels.collate_fn if nt > 1 else 
                             LoadImagesAndLabels.collate_fn), dataset



class InfiniteSeqDataLoader(dataloader.DataLoader):
    """ Dataloader that reuses workers

    Uses same syntax as vanilla DataLoader
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        object.__setattr__(self, 'batch_sampler', _RepeatSeqSampler(self.batch_sampler))
        self.iterator = super().__iter__()

    def __len__(self):
        return len(self.batch_sampler.sampler)

    def __iter__(self):
        for _ in range(len(self)):
            yield next(self.iterator)


class _RepeatSeqSampler:
    """ Sampler that repeats forever

    Args:
        sampler (Sampler)
    """

    def __init__(self, sampler):
        self.sampler = sampler # RandomSampler

    def __iter__(self):
        while True:
            yield from iter(self.sampler)


class LoadSeqAndLabels(LoadImagesAndLabels):
    """
    Data loader: loads sequential video frames
    """
    cache_version = 0.6  # dataset labels *.cache version
    rand_interp_methods = [cv2.INTER_NEAREST, cv2.INTER_LINEAR, cv2.INTER_CUBIC, cv2.INTER_AREA, cv2.INTER_LANCZOS4]

    def __init__(self,
                 path,
                 img_size=640,
                 batch_size=16,
                 nt=8,
                 seq_batch=False,
                 augment=False,
                 hyp=None,
                 rect=False,
                 cache_images=False,
                 single_cls=False,
                 stride=32,
                 pad=0.0,
                 prefix=''):
        super().__init__(path,
                        img_size,
                        batch_size,
                        augment,
                        hyp,
                        rect,
                        False,
                        cache_images,
                        single_cls,
                        stride,
                        pad,
                        prefix)
        if self.mosaic:
            LOGGER.warning('WARNING: mosaic data augmentation is not yet supported for videos')
        self.mosaic = False
        self.nt = nt
        self.epoch = 0
        self.seq_batch = seq_batch
        self.batch_size = batch_size
        
        self.start_indices = [np.random.randint(0, len(self.im_files) - nt) for _ in range(batch_size)]
        self.init_indices = self.start_indices # Used to reset indices after batch
        self.max_val = len(self.im_files)-self.nt


    def next_batch(self):
        if self.seq_batch:
            self.start_indices = [i+1 % self.max_val for i in self.start_indices]

    def next_epoch(self):
        self.start_indices = self.init_indices

    def wrap_index(self, index):
        return index % self.batch_size

    def __len__(self):
        return self.max_val

    def __getitem__(self, index):
        """
        Returns a sequence of images starting with the given index.
        To prevent overflow indices, they're clamped to a maximum of the dataset length.
            - s_imgs -> list of tensors
            - s_labels -> list of tensors
            - s_shapes -> list of ints
        """
        if not self.seq_batch:
            index = index if index <= self.max_val else self.max_val
            index = self.indices[index]  # linear, shuffled, or image_weights
            self.start_indices = create_sequence_idxs(self.nt, index) # array of indices
        else:
            index = self.start_indices[self.wrap_index(index)]

        hyp = self.hyp
        mosaic = self.mosaic and random.random() < hyp['mosaic']
        s_imgs = []
        s_labels = []
        s_shapes = []
        s_paths = self.im_files[index:index+self.nt]
        max_nl = 0

        for ii in range(index, index + self.nt):
            if mosaic:
                # Load mosaic
                img, labels = self.load_mosaic(ii)
                shapes = None

                # MixUp augmentation
                if random.random() < hyp['mixup']:
                    img, labels = mixup(img, labels, *self.load_mosaic(random.randint(0, self.n - 1)))

            else:
                # Load image
                img, (h0, w0), (h, w) = self.load_image(ii)

                # Letterbox
                shape = self.batch_shapes[self.batch[index]] if self.rect else self.img_size  # final letterboxed shape
                img, ratio, pad = letterbox(img, shape, auto=False, scaleup=self.augment)
                shapes = (h0, w0), ((h / h0, w / w0), pad)  # for COCO mAP rescaling

                labels = self.labels[index].copy()
                if labels.size:  # normalized xywh to pixel xyxy format
                    labels[:, 1:] = xywhn2xyxy(labels[:, 1:], ratio[0] * w, ratio[1] * h, padw=pad[0], padh=pad[1])

                if self.augment:
                    img, labels = random_perspective(img,
                                                    labels,
                                                    degrees=hyp['degrees'],
                                                    translate=hyp['translate'],
                                                    scale=hyp['scale'],
                                                    shear=hyp['shear'],
                                                    perspective=hyp['perspective'])

            nl = len(labels)  # number of labels
            if nl:
                if nl > max_nl: max_nl = nl
                labels[:, 1:5] = xyxy2xywhn(labels[:, 1:5], w=img.shape[1], h=img.shape[0], clip=True, eps=1E-3)

            if self.augment:
                # Albumentations
                img, labels = self.albumentations(img, labels)
                nl = len(labels)  # update after albumentations

                # HSV color-space
                augment_hsv(img, hgain=hyp['hsv_h'], sgain=hyp['hsv_s'], vgain=hyp['hsv_v'])

                # Flip up-down
                if random.random() < hyp['flipud']:
                    img = np.flipud(img)
                    if nl:
                        labels[:, 2] = 1 - labels[:, 2]

                # Flip left-right
                if random.random() < hyp['fliplr']:
                    img = np.fliplr(img)
                    if nl:
                        labels[:, 1] = 1 - labels[:, 1]

                # Cutouts
                # labels = cutout(img, labels, p=0.5)
                # nl = len(labels)  # update after cutout

            labels_out = torch.zeros((nl, 6))
            if nl:
                labels_out[:, 1:] = torch.from_numpy(labels)

            # Convert
            img = img.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
            img = np.ascontiguousarray(img)

            s_imgs.append(torch.from_numpy(img))
            s_labels.append(labels_out)
            s_shapes.append(shapes)

        if self.nt == 1:
            s_imgs, s_labels, s_paths, s_shapes = s_imgs[0], s_labels[0], s_paths[0], s_shapes[0]

        return s_imgs, s_labels, s_paths, s_shapes

    @staticmethod
    def collate_fn(batch): 
        b_ims, b_labels, paths, shapes = zip(*batch) 
        i_out, l_out = [], [] 
        for i, s_labels in enumerate(b_labels):                      
            for ii, lb in enumerate(s_labels):                     
                lb[:, 0] = ii # add target image index for build_targets()
            l_out.append(torch.cat(s_labels, 0)) # collate seq
            i_out.append(torch.stack(b_ims[i], 0)) # seq. of images

        return torch.stack(i_out, 0), torch.cat(l_out, 0), paths, shapes # collate batch


