import os
import torch
import numpy as np
import scipy.misc

import torch.nn.functional as F

from PIL import Image
from utils.dcrf import crf_inference

from datasets.pascal_voc_ms import MultiscaleLoader, CropLoader 

class ResultWriter:
    
    def __init__(self, cfg, palette, out_path, verbose=True):
        self.cfg = cfg
        self.palette = palette
        self.root = out_path
        self.verbose = verbose

    def _mask_overlay(self, mask, image, alpha=0.3):
        """Creates an overlayed mask visualisation"""
        mask_rgb = self.__mask2rgb(mask)
        return alpha * image + (1 - alpha) * mask_rgb

    def __mask2rgb(self, mask):
        im = Image.fromarray(mask).convert("P")
        im.putpalette(self.palette)
        mask_rgb = np.array(im.convert("RGB"), dtype=np.float)
        return mask_rgb / 255.

    def _merge_masks(self, masks, labels, pads):
        """Combines masks at multiple scales
        
        Args:
            masks: list of masks obtained at different scales
            (already scaled to the original)
        Returns:
            pred: combined single mask
            pred_crf: refined mask with CRF
        """
        raise NotImplementedError

    def save(self, img_path, img_orig, all_masks, labels, pads, gt_mask):

        img_name = os.path.basename(img_path).rstrip(".jpg")

        # converting original image to [0, 255]
        img_orig255 = np.round(255. * img_orig).astype(np.uint8)
        img_orig255 = np.transpose(img_orig255, [1,2,0])
        img_orig255 = np.ascontiguousarray(img_orig255)

        merged_mask = self._merge_masks(all_masks, pads, labels, img_orig255.shape[:2])
        pred = np.argmax(merged_mask, 0)

        # CRF
        pred_crf = crf_inference(img_orig255, merged_mask, t=10, scale_factor=1, labels=21)
        pred_crf = np.argmax(pred_crf, 0)

        filepath = os.path.join(self.root, img_name + '.png')
        scipy.misc.imsave(filepath, pred.astype(np.uint8))

        filepath = os.path.join(self.root, "crf", img_name + '.png')
        scipy.misc.imsave(filepath, pred_crf.astype(np.uint8))

        if self.verbose:
            mask_gt = gt_mask.numpy()
            masks_all = np.concatenate([pred, pred_crf, mask_gt], 1).astype(np.uint8)
            images = np.concatenate([img_orig]*3, 2)
            images = np.transpose(images, [1,2,0])
            
            overlay = self._mask_overlay(masks_all, images)
            filepath = os.path.join(self.root, "vis", img_name + '.png')
            overlay255 = np.round(overlay * 255.).astype(np.uint8)
            scipy.misc.imsave(filepath, overlay255)

class MergeMultiScale(ResultWriter):

    def _cut(self, x_chw, pads):
        pad_h, pad_w, h, w = [int(p) for p in pads]
        return x_chw[:, pad_h:(pad_h + h), pad_w:(pad_w + w)]

    def _merge_masks(self, masks, labels, pads, imsize_hw):

        mask_list = []
        for i, mask in enumerate(masks.split(1, dim=0)):

            # removing the padding
            mask_cut = self._cut(mask[0], pads[i]).unsqueeze(0)

            # normalising the scale
            mask_cut = F.interpolate(mask_cut, imsize_hw, mode='bilinear', align_corners=False)[0]

            # flipping if necessary
            if self.cfg.FLIP and i % 2 == 1:
                mask_cut = torch.flip(mask_cut, (-1, ))

            # getting the max response
            mask_cut[1:, ::] *= labels[:, None, None]
            mask_list.append(mask_cut)

        mean_mask = sum(mask_list).numpy() / len(mask_list)

        # discounting BG
        #mean_mask[0, ::] *= 0.5
        mean_mask[0, ::] = np.power(mean_mask[0, ::], self.cfg.BG_POW)

        return mean_mask

class MergeCrops(ResultWriter):

    def _cut(self, x_chw, pads):
        pad_h, pad_w, h, w = [int(p) for p in pads]
        return x_chw[:, pad_h:(pad_h + h), pad_w:(pad_w + w)]

    def _merge_masks(self, masks, labels, coords, imsize_hw):
        num_classes = masks.size(1)

        masks_sum = torch.zeros([num_classes, *imsize_hw]).type_as(masks)
        counts = torch.zeros(imsize_hw).type_as(masks)

        for ii, (mask, pads) in enumerate(zip(masks.split(1), coords.split(1))):

            mask = mask[0]
            s_h, e_h, s_w, e_w = pads[0][:4]
            pad_t, pad_l = pads[0][4:]

            if self.cfg.FLIP and ii % 2 == 0:
                mask = mask.flip(-1)

            # crop mask, if needed
            m_h = 0 if s_h > 0 else pad_t
            m_w = 0 if s_w > 0 else pad_l

            # due to padding
            # end point is shifted
            s_h = max(0, s_h - pad_t)
            s_w = max(0, s_w - pad_l)
            e_h = min(e_h - pad_t, imsize_hw[0])
            e_w = min(e_w - pad_l, imsize_hw[1])

            m_he = m_h + e_h - s_h
            m_we = m_w + e_w - s_w

            masks_sum[:, s_h:e_h, s_w:e_w] += mask[:, m_h:m_he, m_w:m_we]
            counts[s_h:e_h, s_w:e_w] += 1

        assert torch.all(counts > 0)
        
        # removing false pasitives
        masks_sum[1:, ::] *= labels[:, None, None]

        # removing the padding
        return (masks_sum / counts).numpy()

class PAMRWriter(ResultWriter):

    def save_batch(self, img_paths, imgs, all_masks, all_gt_masks):

        for b, img_path in enumerate(img_paths):
            
            img_name = os.path.basename(img_path).rstrip(".jpg")
            img_orig = imgs[b]
            gt_mask = all_gt_masks[b]

            # converting original image to [0, 255]
            img_orig255 = np.round(255. * img_orig).astype(np.uint8)
            img_orig255 = np.transpose(img_orig255, [1,2,0])
            img_orig255 = np.ascontiguousarray(img_orig255)

            mask_gt = torch.argmax(gt_mask, 0)

            # cancel ambiguous
            ambiguous = gt_mask.sum(0) == 0
            mask_gt[ambiguous] = 255
            mask_gt = mask_gt.numpy()

            # saving GT
            image_hwc = np.transpose(img_orig, [1,2,0])
            overlay_gt = self._mask_overlay(mask_gt.astype(np.uint8), image_hwc, alpha=0.5)

            filepath = os.path.join(self.root, img_name + '_gt.png')
            overlay255 = np.round(overlay_gt * 255.).astype(np.uint8)
            scipy.misc.imsave(filepath, overlay255)

            for it, mask_batch in enumerate(all_masks):

                mask = mask_batch[b]
                mask_idx = torch.argmax(mask, 0)

                # cancel ambiguous
                ambiguous = mask.sum(0) == 0
                mask_idx[ambiguous] = 255

                overlay = self._mask_overlay(mask_idx.numpy().astype(np.uint8), image_hwc, alpha=0.5)

                filepath = os.path.join(self.root, img_name + '_{:02d}.png'.format(it))
                overlay255 = np.round(overlay * 255.).astype(np.uint8)
                scipy.misc.imsave(filepath, overlay255)


def get_inference_io(method_name):

    if method_name == "multiscale":
        return MergeMultiScale, MultiscaleLoader
    elif method_name == "multicrop":
        return MergeCrops, CropLoader
    else:
        raise NotImplementedError("Method {} is unknown".format(method_name))
