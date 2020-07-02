"""
Evaluating class activation maps from a given snapshot
Supports multi-scale fusion of the masks
Based on PSA
"""

import os
import sys
import numpy as np
import scipy
import torch.multiprocessing as mp
from tqdm import tqdm
from PIL import Image

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as tf
from torch.utils.data import DataLoader
from torch.backends import cudnn
cudnn.enabled = True
cudnn.benchmark = False
cudnn.deterministic = True

from opts import get_arguments
from core.config import cfg, cfg_from_file, cfg_from_list
from models import get_model

from utils.checkpoints import Checkpoint
from utils.timer import Timer
from utils.dcrf import crf_inference
from utils.inference_tools import get_inference_io

def check_dir(base_path, name):

    # create the directory
    fullpath = os.path.join(base_path, name)
    if not os.path.exists(fullpath):
        os.makedirs(fullpath)

    return fullpath

def HWC_to_CHW(img):
    return np.transpose(img, (2, 0, 1))

if __name__ == '__main__':

    # loading the model
    args = get_arguments(sys.argv[1:])

    # reading the config
    cfg_from_file(args.cfg_file)
    if args.set_cfgs is not None:
        cfg_from_list(args.set_cfgs)

    # initialising the dirs
    check_dir(args.mask_output_dir, "vis")
    check_dir(args.mask_output_dir, "crf")

    # Loading the model
    model = get_model(cfg.NET, num_classes=cfg.TEST.NUM_CLASSES)
    checkpoint = Checkpoint(args.snapshot_dir, max_n = 5)
    checkpoint.add_model('enc', model)
    checkpoint.load(args.resume)

    for p in model.parameters():
        p.requires_grad = False

    # setting the evaluation mode
    model.eval()

    assert hasattr(model, 'normalize')
    transform = tf.Compose([np.asarray, model.normalize])

    WriterClass, DatasetClass = get_inference_io(cfg.TEST.METHOD)

    dataset = DatasetClass(args.infer_list, cfg.TEST, transform=transform)

    dataloader = DataLoader(dataset, shuffle=False, num_workers=args.workers, \
                            pin_memory=True, batch_size=cfg.TEST.BATCH_SIZE)

    model = nn.DataParallel(model).cuda()

    timer = Timer()
    N = len(dataloader)

    palette = dataset.get_palette()
    pool = mp.Pool(processes=args.workers)
    writer = WriterClass(cfg.TEST, palette, args.mask_output_dir)

    for iter, (img_name, img_orig, images_in, pads, labels, gt_mask) in enumerate(tqdm(dataloader)):

        # cutting the masks
        masks = []

        with torch.no_grad():
            cls_raw, masks_pred = model(images_in)

            if not cfg.TEST.USE_GT_LABELS:
                cls_sigmoid = torch.sigmoid(cls_raw)
                cls_sigmoid, _ = cls_sigmoid.max(0)
                #cls_sigmoid = cls_sigmoid.mean(0)
                # threshold class scores
                labels = (cls_sigmoid > cfg.TEST.FP_CUT_SCORE)
            else:
                labels = labels[0]

        # saving the raw npy
        image = dataset.denorm(img_orig[0]).numpy()
        masks_pred = masks_pred.cpu()
        labels = labels.type_as(masks_pred)

        #writer.save(img_name[0], image, masks_pred, pads, labels, gt_mask[0])
        pool.apply_async(writer.save, args=(img_name[0], image, masks_pred, pads, labels, gt_mask[0]))

        timer.update_progress(float(iter + 1) / N)
        if iter % 100 == 0:
            msg = "Finish time: {}".format(timer.str_est_finish())
            tqdm.write(msg)
            sys.stdout.flush()

    pool.close()
    pool.join()
