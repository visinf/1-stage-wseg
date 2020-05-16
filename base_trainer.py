import os
import re
import torch
import math
import numpy as np

import torchvision.utils as vutils

from PIL import Image, ImageDraw, ImageFont
from tensorboardX import SummaryWriter
from utils.checkpoints import Checkpoint

from core.config import cfg, cfg_from_file, cfg_from_list

class BaseTrainer(object):

    def __del__(self):
        # commented out, because hangs on exit
        # (presumably some bug with threading in TensorboardX)
        """
        if not self.quiet:
            self.writer.close()
            self.writer_val.close()
        """
        pass

    def __init__(self, args, quiet=False):
        self.args = args
        self.quiet = quiet

        # config
        # Reading the config
        if type(args.cfg_file) is str \
                and os.path.isfile(args.cfg_file):

            cfg_from_file(args.cfg_file)
            if args.set_cfgs is not None:
                cfg_from_list(args.set_cfgs)

        self.start_epoch = 0
        self.best_score = -1e16
        self.checkpoint = Checkpoint(args.snapshot_dir, max_n = 5)

        if not quiet:
            #self.model_id = "%s" % args.run
            logdir = os.path.join(args.logdir, 'train')
            logdir_val = os.path.join(args.logdir, 'val')

            self.writer = SummaryWriter(logdir)
            self.writer_val = SummaryWriter(logdir_val)

    def _define_checkpoint(self, name, model, optim):
        self.checkpoint.add_model(name, model, optim)

    def _load_checkpoint(self, suffix):
        if self.checkpoint.load(suffix):
            # loading the epoch and the best score
            tmpl = re.compile("^e(\d+)Xs([\.\d+\-]+)$")
            match = tmpl.match(suffix)
            if not match:
                print("Warning: epoch and score could not be recovered")
                return
            else:
                epoch, score = match.groups()
                self.start_epoch = int(epoch) + 1
                self.best_score = float(score)

    def checkpoint_epoch(self, score, epoch):

        if score > self.best_score:
            self.best_score = score

        print(">>> Saving checkpoint with score {:3.2e}, epoch {}".format(score, epoch))
        suffix = "e{:03d}Xs{:4.3f}".format(epoch, score)
        self.checkpoint.checkpoint(suffix)

        return True

    def checkpoint_best(self, score, epoch):

        if score > self.best_score:
            print(">>> Saving checkpoint with score {:3.2e}, epoch {}".format(score, epoch))
            self.best_score= score

            suffix = "e{:03d}Xs{:4.3f}".format(epoch, score)
            self.checkpoint.checkpoint(suffix)

            return True

        return False

    @staticmethod
    def get_optim(params, cfg):

        if not hasattr(torch.optim, cfg.OPT):
            print("Optimiser {} not supported".format(cfg.OPT))
            raise NotImplementedError

        optim = getattr(torch.optim, cfg.OPT)

        if cfg.OPT == 'Adam':
            upd = torch.optim.Adam(params, lr=cfg.LR, \
                                   betas=(cfg.BETA1, 0.999), \
                                   weight_decay=cfg.WEIGHT_DECAY)
        elif cfg.OPT == 'SGD':
            print("Using SGD >>> learning rate = {:4.3e}, momentum = {:4.3e}, weight decay = {:4.3e}".format(cfg.LR, cfg.MOMENTUM, cfg.WEIGHT_DECAY))
            upd = torch.optim.SGD(params, lr=cfg.LR, \
                                  momentum=cfg.MOMENTUM, \
                                  weight_decay=cfg.WEIGHT_DECAY)

        else:
            upd = optim(params, lr=cfg.LR)

        upd.zero_grad()

        return upd
    
    @staticmethod
    def set_lr(optim, lr):
        for param_group in optim.param_groups:
            param_group['lr'] = lr


    def _visualise_grid(self, x_all, labels, t, ious=None, tag="visualisation", scores=None):

        # adding the labels to images
        bs, ch, h, w = x_all.size()
        x_all_new = torch.zeros(bs, ch, h + 16, w)
        _, y_labels_idx = torch.max(labels, -1)
        classNamesOffset = len(self.classNames) - labels.size(1)
        classNames = self.classNames[classNamesOffset:]
        for b in range(bs):
            label_idx = labels[b]
            label_names = [name for i,name in enumerate(classNames) if label_idx[i].item()]
            label_name = ", ".join(label_names)

            ndarr = x_all[b].mul(255).clamp(0, 255).byte().permute(1, 2, 0).cpu().numpy()
            arr = np.zeros((16, w, ch), dtype=ndarr.dtype)
            ndarr = np.concatenate((arr, ndarr), 0)
            im = Image.fromarray(ndarr)
            draw = ImageDraw.Draw(im)

            font = ImageFont.truetype("fonts/UbuntuMono-R.ttf", 12)

            # draw.text((x, y),"Sample Text",(r,g,b))
            draw.text((5, 1), label_name, (255,255,255), font=font)
            im_np = np.array(im).astype(np.float)
            x_all_new[b] = (torch.from_numpy(im_np)/255.0).permute(2,0,1)

        summary_grid = vutils.make_grid(x_all_new, nrow=1, padding=8, pad_value=0.9)
        self.writer.add_image(tag, summary_grid, t)

    def _apply_cmap(self, mask_idx, mask_conf):
        palette = self.trainloader.dataset.get_palette()

        masks = []
        col = Colorize()
        mask_conf = mask_conf.float() / 255.0
        for mask, conf in zip(mask_idx.split(1), mask_conf.split(1)):
            m = col(mask).float()
            m = m * conf
            masks.append(m[None, ...])

        return torch.cat(masks, 0)

    def _mask_rgb(self, masks, image_norm, alpha=0.3):
        # visualising masks
        masks_conf, masks_idx = torch.max(masks, 1)
        masks_conf = masks_conf - F.relu(masks_conf - 1, 0)

        masks_idx_rgb = self._apply_cmap(masks_idx.cpu(), masks_conf.cpu())
        return alpha * image_norm + (1 - alpha) * masks_idx_rgb

    def _init_norm(self):
        self.trainloader.dataset.set_norm(self.enc.normalize)
        self.valloader.dataset.set_norm(self.enc.normalize)
        self.trainloader_val.dataset.set_norm(self.enc.normalize)
