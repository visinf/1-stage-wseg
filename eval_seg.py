"""
Evaluating the masks

TODO:
    Parallelise with 

    from multiprocessing import Pool

    ConfM = ConfusionMatrix(class_num)
    f = ConfM.generateM
    pool = Pool() 
    m_list = pool.map(f, data_list)
    pool.close() 
    pool.join() 

"""

import sys
import os
import numpy as np
import argparse
import scipy

from tqdm import tqdm
from datasets.pascal_voc import PascalVOC
from PIL import Image 
from utils.metrics import Metric

# Defining
parser = argparse.ArgumentParser(description="Mask Evaluation")

parser.add_argument("--data", type=str, default='./data/annotation',
                    help="The prefix for data directory")
parser.add_argument("--filelist", type=str, default='./data/val.txt',
                    help="A text file containing the paths to masks")
parser.add_argument("--masks", type=str, default='./masks',
                    help="A path to generated masks")
parser.add_argument("--oracle-from", type=str, default="",
                    help="Use GT mask but down- then upscale them")
parser.add_argument("--log-scores", type=str, default='./scores.log',
                    help="Logging scores for invididual images")

def check_args(args):
    """Check the files/directories exist"""

    assert os.path.isdir(args.data), \
            "Directory {} does not exist".format(args.data)
    assert os.path.isfile(args.filelist), \
            "File {} does not exist".format(args.filelist)
    if len(args.oracle_from) > 0:
        vals = args.oracle_from.split('x')
        assert len(vals) == 2, "HxW expected"
        h, w = vals
        assert int(h) > 2, "Meaningless resolution"
        assert int(w) > 2, "Meaningless resolution"
    else:
        assert os.path.isdir(args.masks), \
            "Directory {} does not exist".format(args.masks)

def format_num(x):
    return round(x*100., 1)


def get_stats(M, i):

    TP = M[i, i]
    FN = np.sum(M[i, :]) - TP # false negatives
    FP = np.sum(M[:, i]) - TP # false positives

    return TP, FN, FP

def summarise_one(class_stats, M, name, labels):

    for i in labels:

        # skipping the ambiguous
        if i == 255:
            continue

        # category name
        TP, FN, FP = get_stats(M, i)
        score = TP - FN - FP

        class_stats[i].append((name, score))

def summarise_per_class(class_stats, filename):
    
    data = ""
    for cat in PascalVOC.CLASSES:

        if cat == "ambiguous":
            continue

        i = PascalVOC.CLASS_IDX[cat]
        sorted_by_score = sorted(class_stats[i], key=lambda x: -x[1])
        data += cat + "\n"
        for name, score in sorted_by_score:
            data += "{:05d} | {}\n".format(int(score), name)

    with open(filename, 'w') as f:
        f.write(data)

def summarise_stats(M):

    eps = 1e-20

    mean = Metric()
    mean.add_metric(Metric.IoU)
    mean.add_metric(Metric.Precision)
    mean.add_metric(Metric.Recall)

    mean_bkg = Metric()
    mean_bkg.add_metric(Metric.IoU)
    mean_bkg.add_metric(Metric.Precision)
    mean_bkg.add_metric(Metric.Recall)

    head_fmt = "{:>12} | {:>5}" + " | {:>5}"*3
    row_fmt = "{:>12} | {:>5}" + " | {:>5.1f}"*3
    split = "-"*44

    def print_row(fmt, row):
        print(fmt.format(*row))

    print_row(head_fmt, ("Class", "#", "IoU", "Pr", "Re"))
    print(split)

    for cat in PascalVOC.CLASSES:

        if cat == "ambiguous":
            continue

        i = PascalVOC.CLASS_IDX[cat]

        TP, FN, FP = get_stats(M, i)

        iou = 100. * TP / (eps + FN + FP + TP)
        pr = 100. * TP / (eps + TP + FP)
        re = 100. * TP / (eps + TP + FN)

        mean_bkg.update_value(Metric.IoU, iou)
        mean_bkg.update_value(Metric.Precision, pr)
        mean_bkg.update_value(Metric.Recall, re)

        if cat != "background":
            mean.update_value(Metric.IoU, iou)
            mean.update_value(Metric.Precision, pr)
            mean.update_value(Metric.Recall, re)

        count = int(np.sum(M[i, :]))
        print_row(row_fmt, (cat, count, iou, pr, re))


    print(split)
    sys.stdout.write("mIoU: {:.2f}\t".format(mean.summarize(Metric.IoU)))
    sys.stdout.write("  Pr: {:.2f}\t".format(mean.summarize(Metric.Precision)))
    sys.stdout.write("  Re: {:.2f}\n".format(mean.summarize(Metric.Recall)))

    print(split)
    print("With background: ")
    sys.stdout.write("mIoU: {:.2f}\t".format(mean_bkg.summarize(Metric.IoU)))
    sys.stdout.write("  Pr: {:.2f}\t".format(mean_bkg.summarize(Metric.Precision)))
    sys.stdout.write("  Re: {:.2f}\n".format(mean_bkg.summarize(Metric.Recall)))


def evaluate_one(conf_mat, mask_gt, mask):

    gt = mask_gt.reshape(-1)
    pred = mask.reshape(-1)
    conf_mat_one = np.zeros_like(conf_mat)

    assert(len(gt) == len(pred))

    for i in range(len(gt)):
        if gt[i] < conf_mat.shape[0]:
            conf_mat[gt[i], pred[i]] += 1.0
            conf_mat_one[gt[i], pred[i]] += 1.0

    return conf_mat_one

def read_mask_file(filepath):
    return np.array(Image.open(filepath))

def oracle_lower(mask, h, w, alpha):

    mask_dict = {}
    labels = np.unique(mask)
    new_mask = np.zeros_like(mask)
    H, W = mask.shape

    # skipping background
    for l in labels:
        if l in (0, 255):
            continue

        mask_l = (mask == l).astype(np.float)
        mask_down = scipy.misc.imresize(mask_l, (h, w), interp='bilinear')
        mask_up = scipy.misc.imresize(mask_down, (H, W), interp='bilinear')
        new_mask[mask_up > alpha] = l

    return new_mask

def get_image_name(name):
    base = os.path.basename(name)
    base = base.replace(".jpg", "")
    return base

def evaluate_all(args):

    with_oracle = False
    if len(args.oracle_from) > 0:
        oh, ow = [int(x) for x in args.oracle_from.split("x")]
        with_oracle = (oh > 1 and ow > 1)

    if with_oracle:
        print(">>> Using oracle {}x{}".format(oh, ow))

    # initialising the confusion matrix
    conf_mat = np.zeros((21, 21))
    class_stats = {}
    for class_idx in range(21):
        class_stats[class_idx] = []

    # count of the images
    num_im = 0

    # opening the filelist
    with open(args.filelist) as fd:

        for line in tqdm(fd.readlines()):

            files = [x.strip('/ \n') for x in line.split(' ')]

            if len(files) < 2:
                print("No path to GT mask found in line\n")
                print("\t{}".format(line))
                continue

            filepath_gt = os.path.join(args.data, files[1])
            if not os.path.isfile(filepath_gt):
                print("File not found (GT): {}".format(filepath_gt))
                continue

            mask_gt = read_mask_file(filepath_gt)

            if with_oracle:
                mask = oracle_lower(mask_gt, oh, ow, alpha=0.5)
            else:
                basename = os.path.basename(files[1])
                filepath = os.path.join(args.masks, basename)
                if not os.path.isfile(filepath):
                    print("File not found: {}".format(filepath))
                    continue

                mask = read_mask_file(filepath)

            if mask.shape != mask_gt.shape:
                print("Mask shape mismatch in {}: ".format(basename), \
                        mask.shape, " vs ", mask_gt.shape)
                continue

            conf_mat_one = evaluate_one(conf_mat, mask_gt, mask)

            image_name = get_image_name(files[0])
            image_labels = np.unique(mask_gt)
            summarise_one(class_stats, conf_mat_one, image_name, image_labels)

            num_im += 1

    
    print("# of images: {}".format(num_im))
    summarise_per_class(class_stats, args.log_scores)

    return conf_mat

if __name__ == "__main__":

    args = parser.parse_args(sys.argv[1:])
    check_args(args)
    stats = evaluate_all(args)
    summarise_stats(stats)
