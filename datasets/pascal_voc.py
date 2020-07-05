import os
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image, ImagePalette
from .utils import colormap
import datasets.transforms as tf

class PascalVOC(Dataset):

    CLASSES = [
        'background', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 
        'bus', 'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse',
        'motorbike', 'person', 'potted-plant', 'sheep', 'sofa', 'train',
        'tv/monitor', 'ambiguous'
    ]

    CLASS_IDX = {
            'background': 0, 
            'aeroplane': 1,
            'bicycle': 2,
            'bird': 3,
            'boat': 4,
            'bottle': 5,
            'bus': 6,
            'car': 7,
            'cat': 8,
            'chair': 9,
            'cow': 10,
            'diningtable': 11,
            'dog': 12,
            'horse': 13,
            'motorbike': 14,
            'person': 15,
            'potted-plant': 16,
            'sheep': 17,
            'sofa': 18,
            'train': 19,
            'tv/monitor': 20,
            'ambiguous': 255
            }

    CLASS_IDX_INV = {
            0: 'background', 
            1: 'aeroplane',
            2: 'bicycle',
            3: 'bird',
            4: 'boat',
            5: 'bottle',
            6: 'bus',
            7: 'car',
            8: 'cat',
            9: 'chair',
           10: 'cow',
           11: 'diningtable',
           12: 'dog',
           13: 'horse',
           14: 'motorbike',
           15: 'person',
           16: 'potted-plant',
           17: 'sheep',
           18: 'sofa',
           19: 'train',
           20: 'tv/monitor',
          255: 'ambiguous'}

    NUM_CLASS = 21

    MEAN = (0.485, 0.456, 0.406)
    STD = (0.229, 0.224, 0.225)

    def __init__(self):
        super().__init__()
        self._init_palette()

    def _init_palette(self):
        self.cmap = colormap()
        self.palette = ImagePalette.ImagePalette()
        for rgb in self.cmap:
            self.palette.getcolor(rgb)

    def get_palette(self):
        return self.palette

    def denorm(self, image):

        if image.dim() == 3:
            assert image.dim() == 3, "Expected image [CxHxW]"
            assert image.size(0) == 3, "Expected RGB image [3xHxW]"

            for t, m, s in zip(image, self.MEAN, self.STD):
                t.mul_(s).add_(m)
        elif image.dim() == 4:
            # batch mode
            assert image.size(1) == 3, "Expected RGB image [3xHxW]"

            for t, m, s in zip((0,1,2), self.MEAN, self.STD):
                image[:, t, :, :].mul_(s).add_(m)

        return image


class VOCSegmentation(PascalVOC):

    def __init__(self, cfg, split, test_mode, root=os.path.expanduser('./data')):
        super(VOCSegmentation, self).__init__()

        self.cfg = cfg
        self.root = root
        self.split = split
        self.test_mode = test_mode

        # train/val/test splits are pre-cut
        if self.split == 'train':
            _split_f = os.path.join(self.root, 'train_augvoc.txt')
        elif self.split == 'train_voc':
            _split_f = os.path.join(self.root, 'train_voc.txt')
        elif self.split == 'val':
            _split_f = os.path.join(self.root, 'val_voc.txt')
        elif self.split == 'test':
            _split_f = os.path.join(self.root, 'test.txt')
        else:
            raise RuntimeError('Unknown dataset split.')

        assert os.path.isfile(_split_f), "%s not found" % _split_f

        self.images = []
        self.masks = []
        with open(_split_f, "r") as lines:
            for line in lines:
                _image, _mask = line.strip("\n").split(' ')
                _image = os.path.join(self.root, _image)
                assert os.path.isfile(_image), '%s not found' % _image
                self.images.append(_image)
            
                if self.split != 'test':
                    _mask = os.path.join(self.root, _mask.lstrip('/'))
                    assert os.path.isfile(_mask), '%s not found' % _mask
                    self.masks.append(_mask)

        if self.split != 'test':
            assert (len(self.images) == len(self.masks))
            if self.split == 'train':
                assert len(self.images) == 10582
            elif self.split == 'val':
                assert len(self.images) == 1449

        self.transform = tf.Compose([tf.MaskRandResizedCrop(self.cfg.DATASET), \
                                     tf.MaskHFlip(), \
                                     tf.MaskColourJitter(p = 1.0), \
                                     tf.MaskNormalise(self.MEAN, self.STD), \
                                     tf.MaskToTensor()])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):

        image = Image.open(self.images[index]).convert('RGB')
        mask  = Image.open(self.masks[index])

        unique_labels = np.unique(mask)

        # ambigious
        if unique_labels[-1] == self.CLASS_IDX['ambiguous']:
            unique_labels = unique_labels[:-1]

        # ignoring BG
        labels = torch.zeros(self.NUM_CLASS - 1)
        if unique_labels[0] == self.CLASS_IDX['background']:
            unique_labels = unique_labels[1:]
        unique_labels -= 1 # shifting since no BG class

        assert unique_labels.size > 0, 'No labels found in %s' % self.masks[index]
        labels[unique_labels.tolist()] = 1

        # general resize, normalize and toTensor
        image, mask = self.transform(image, mask)

        return image, labels, os.path.basename(self.images[index])

    @property
    def pred_offset(self):
        return 0
