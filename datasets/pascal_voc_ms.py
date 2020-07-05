"""
Multi-scale dataloader
Credit PSA
"""

from PIL import Image
from .pascal_voc import PascalVOC

import math
import numpy as np
import torch
import os.path
import torchvision.transforms.functional as F


def load_img_name_list(dataset_path, index = 0):

    img_gt_name_list = open(dataset_path).read().splitlines()
    img_name_list = [img_gt_name.split(' ')[index].strip('/') for img_gt_name in img_gt_name_list]

    return img_name_list

def load_label_name_list(dataset_path):
    return load_img_name_list(dataset_path, index = 1)

class VOC12ImageDataset(PascalVOC):

    def __init__(self, img_name_list_path, voc12_root):
        super().__init__()

        self.img_name_list = load_img_name_list(img_name_list_path)
        self.voc12_root = voc12_root
        self.batch_size = 1

    def __len__(self):
        return len(self.img_name_list)

    def __getitem__(self, idx):
        fullpath = os.path.join(self.voc12_root, self.img_name_list[idx])
        img = Image.open(fullpath).convert("RGB")
        return fullpath, img

class VOC12ClsDataset(VOC12ImageDataset):

    def __init__(self, img_name_list_path, voc12_root):
        super(VOC12ClsDataset, self).__init__(img_name_list_path, voc12_root)
        self.label_list = load_label_name_list(img_name_list_path)

    def __len__(self):
        return self.batch_size * len(self.img_name_list)

    def _pad(self, image):
        w, h = image.size

        pad_mask = Image.new("L", image.size)
        pad_height = self.pad_size[0] - h
        pad_width = self.pad_size[1] - w

        assert pad_height >= 0 and pad_width >= 0

        pad_l = max(0, pad_width // 2)
        pad_r = max(0, pad_width - pad_l)
        pad_t = max(0, pad_height // 2)
        pad_b = max(0, pad_height - pad_t)

        image = F.pad(image, (pad_l, pad_t, pad_r, pad_b), fill=0, padding_mode="constant")
        pad_mask = F.pad(pad_mask, (pad_l, pad_t, pad_r, pad_b), fill=1, padding_mode="constant")

        return image, pad_mask, [pad_t, pad_l]

    def __getitem__(self, idx):
        name, img = super(VOC12ClsDataset, self).__getitem__(idx)
        
        label_fullpath = self.label_list[idx]
        assert len(label_fullpath) < 256, "Expected label path less than 256 for padding"

        mask = Image.open(os.path.join(self.voc12_root, label_fullpath))
        mask = np.array(mask)

        labels = torch.zeros(self.NUM_CLASS - 1)

        # it will also be sorted
        unique_labels = np.unique(mask)

        # ambigious
        if unique_labels[-1] == self.CLASS_IDX['ambiguous']:
            unique_labels = unique_labels[:-1]

        # background
        if unique_labels[0] == self.CLASS_IDX['background']:
            unique_labels = unique_labels[1:]

        assert unique_labels.size > 0, 'No labels found in %s' % self.masks[index]
        unique_labels -= 1 # shifting since no BG class
        labels[unique_labels.tolist()] = 1
        
        return name, img, labels, mask.astype(np.int)


class MultiscaleLoader(VOC12ClsDataset):

    def __init__(self, img_list, cfg, transform):
        super().__init__(img_list, cfg.DATA_ROOT)

        self.scales = cfg.SCALES
        self.pad_size = cfg.PAD_SIZE
        self.use_flips = cfg.FLIP
        self.transform = transform

        self.batch_size = len(self.scales)
        if self.use_flips:
            self.batch_size *= 2

        print("Inference batch size: ", self.batch_size)
        assert self.batch_size == cfg.BATCH_SIZE

    def __getitem__(self, idx):
        im_idx = idx // self.batch_size
        sub_idx = idx % self.batch_size

        scale = self.scales[sub_idx // (2 if self.use_flips else 1)]
        flip = self.use_flips and sub_idx % 2

        name, img, label, mask = super().__getitem__(im_idx)

        target_size = (int(round(img.size[0]*scale)),
                       int(round(img.size[1]*scale)))

        s_img = img.resize(target_size, resample=Image.CUBIC)

        if flip:
            s_img = F.hflip(s_img)

        w, h = s_img.size
        im_msc, ignore, pads_tl = self._pad(s_img)
        pad_t, pad_l = pads_tl

        im_msc = self.transform(im_msc)
        img = F.to_tensor(self.transform(img))

        pads = torch.Tensor([pad_t, pad_l, h, w])

        ignore = np.array(ignore).astype(im_msc.dtype)[..., np.newaxis]
        im_msc = F.to_tensor(im_msc * (1 - ignore))

        return name, img, im_msc, pads, label, mask


class CropLoader(VOC12ClsDataset):

    def __init__(self, img_list, cfg, transform):
        super().__init__(img_list, cfg.DATA_ROOT)

        self.use_flips = cfg.FLIP
        self.transform = transform

        self.grid_h, self.grid_w = cfg.CROP_GRID_SIZE
        self.crop_h, self.crop_w = cfg.CROP_SIZE
        self.pad_size = cfg.PAD_SIZE

        self.stride_h = int(math.ceil(self.pad_size[0] / self.grid_h))
        self.stride_w = int(math.ceil(self.pad_size[1] / self.grid_w))

        assert self.stride_h <= self.crop_h and \
                self.stride_w <= self.crop_w

        self.batch_size = self.grid_h * self.grid_w
        if self.use_flips:
            self.batch_size *= 2

        print("Inference batch size: ", self.batch_size)
        assert self.batch_size == cfg.BATCH_SIZE


    def __getitem__(self, index):
        image_index = index // self.batch_size
        batch_index = index % self.batch_size
        grid_index = batch_index // (2 if self.use_flips else 1)

        index_h = grid_index // self.grid_w
        index_w = grid_index % self.grid_w
        flip = self.use_flips and batch_index % 2 == 0

        name, image, label, mask = super().__getitem__(image_index)

        image_pad, pad_mask, pads = self._pad(image)
        assert image_pad.size[0] == self.pad_size[1] and \
                image_pad.size[1] == self.pad_size[0]

        s_h = index_h * self.stride_h
        e_h = min(s_h + self.crop_h, self.pad_size[0])
        s_h = e_h - self.crop_h

        s_w = index_w * self.stride_w
        e_w = min(s_w + self.crop_w, self.pad_size[1])
        s_w = e_w - self.crop_w

        image_pad = self.transform(image_pad)
        pad_mask = np.array(pad_mask).astype(image_pad.dtype)[..., np.newaxis]
        image_pad *= 1 - pad_mask

        image_pad = F.to_tensor(image_pad)
        image_crop = image_pad[:, s_h:e_h, s_w:e_w].clone()

        pads = torch.LongTensor([s_h, e_h, s_w, e_w] + pads)

        if flip:
            image_crop = image_crop.flip(-1)

        image = F.to_tensor(self.transform(image))

        return name, image, image_crop, pads, label, mask
