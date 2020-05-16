##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
## Created by: Hang Zhang
## ECE Department, Rutgers University
## Email: zhang.hang@rutgers.edu
## Copyright (c) 2017
##
## This source code is licensed under the MIT-style license found in the
## LICENSE file in the root directory of this source tree
##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

import threading
import numpy as np
import torch

from sklearn.metrics import average_precision_score

class Metric(object):

    # synonyms
    IoU = "IoU"
    MaskIoU = "IoU"

    Precision = "Precision"
    Recall = "Recall"
    ClassAP = "ClassAP"

    def __init__(self):
        self.data = {}
        self.count = {}
        self.fn = {}

        # initising the functions
        self.fn[Metric.MaskIoU] = self.mask_iou_
        self.fn[Metric.Precision] = self.precision_
        self.fn[Metric.Recall] = self.recall_
        self.fn[Metric.ClassAP] = self.class_ap_

    def add_metric(self, m):
        assert m in self.fn, "Unknown metric with key {}".format(m)

        self.data[m] = 0.
        self.count[m] = 0.

    def metrics(self):
        return self.data.keys()
    
    def print_summary(self):

        keys_sorted = sorted(self.data.keys())

        for m in keys_sorted:
            print("{}: {:5.4f}".format(m, self.summarize(m)))

    def reset_stat(self, m=None):

        if m is None:
            # resetting everything
            for m in self.data:
                self.data[m] = 0.
                self.count[m] = 0.
        else:
            assert m in self.fn, "Unknown metric with key {}".format(m)

            self.data[m] = 0.
            self.count[m] = 0.

    def update_value(self, m, value, count=1.):

        self.data[m] += value
        self.count[m] += count

    def update(self, gt, pred):

        for m in self.data:
            self.data[m] += self.fn[m](gt, pred)
            self.count[m] += 1.

    def merge(self, metric):

        for m in metric.data:
            if not m in self.data:
                self.reset_stat(m)

            self.update_value(m, metric.data[m], metric.count[m])

    def merge_summary(self, metric):

        for m in metric.data:
            if not m in self.data:
                self.reset_stat(m)

            mean_value = metric.summarize(m)
            self.update_value(m, mean_value, 1.)

    def summarize(self, m):
        if not m in self.count or self.count[m] == 0.:
            return 0.

        return self.data[m] / self.count[m]

    @staticmethod
    def mask_iou_(a, b):
        # computing the mask IoU
        isc = (a * b).sum()
        unn = (a + b).sum()
        z = unn - isc

        if z == 0.:
            return 0.

        return isc / z

    @staticmethod
    def precision_(gt, p):
        # computing the mask IoU
        acc = (gt * p).sum()
        sss = p.sum()

        if sss == 0.:
            return 0.

        return acc / sss

    @staticmethod
    def recall_(gt, p):
        # computing the mask IoU
        acc = (gt * p).sum()
        sss = gt.sum()

        if sss == 0.:
            return 0.

        return acc / sss

    @staticmethod
    def class_ap_(gt, p):

        # this return AP for each class
        ap = average_precision_score(gt, p, average=None)

        # return the average
        return np.mean(ap[gt.sum(0) > 0])


def compute_jaccard(preds_masks_all, targets_masks_all, num_classes=21):

    tps = np.zeros((num_classes, ))
    fps = np.zeros((num_classes, ))
    fns = np.zeros((num_classes, ))
    counts = np.zeros((num_classes, ))

    for mask_pred, mask_gt in zip(preds_masks_all, targets_masks_all):

        bs, h, w = mask_pred.size()
        assert bs == mask_gt.size(0), "Batch size mismatch"
        assert h == mask_gt.size(1), "Width mismatch"
        assert w == mask_gt.size(2), "Height mismatch"

        mask_pred = mask_pred.view(bs, 1, -1)
        mask_gt = mask_gt.view(bs, 1, -1)

        # ignore ambiguous
        mask_pred[mask_gt == 255] = 255

        for label in range(num_classes):
            mask_pred_ = (mask_pred == label).float()
            mask_gt_ = (mask_gt == label).float()

            tps[label] += (mask_pred_ * mask_gt_).sum().item()
            diff = mask_pred_ - mask_gt_
            fps[label] += np.maximum(0., diff).float().sum().item()
            fns[label] += np.maximum(0., -diff).float().sum().item()

    jaccards = [None]*num_classes
    precision = [None]*num_classes
    recall   = [None]*num_classes
    for i in range(num_classes):
        tp = tps[i]
        fn = fns[i]
        fp = fps[i]
        jaccards[i]  = tp / max(1e-3, fn + fp + tp)
        precision[i] = tp / max(1e-3, tp + fp)
        recall[i]    = tp / max(1e-3, tp + fn)

    return jaccards, precision, recall

