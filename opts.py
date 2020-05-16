from __future__ import print_function

import os
import torch
import argparse
from core.config import cfg

def add_global_arguments(parser):

    parser.add_argument("--dataset", type=str,
                        help="Determines dataloader to use (only Pascal VOC supported)")
    parser.add_argument("--exp", type=str, default="main",
                        help="ID of the experiment (multiple runs)")
    parser.add_argument("--resume", type=str, default=None,
                        help="Snapshot \"ID,iter\" to load")
    parser.add_argument("--run", type=str, help="ID of the run")
    parser.add_argument('--workers', type=int, default=8,
                        metavar='N', help='dataloader threads')
    parser.add_argument("--snapshot-dir", type=str, default='./snapshots',
                        help="Where to save snapshots of the model.")
    parser.add_argument("--logdir", type=str, default='./logs',
                        help="Where to save log files of the model.")

    # used at inference only
    parser.add_argument("--infer-list", type=str, default='data/val_augvoc.txt',
                        help="Path to a file list")
    parser.add_argument("--mask-output-dir", type=str, default='results/',
                        help="Path where to save masks")

    #
    # Configuration
    #
    parser.add_argument(
        '--cfg', dest='cfg_file', required=True,
        help='Config file for training (and optionally testing)')
    parser.add_argument(
        '--set', dest='set_cfgs',
        help='Set config keys. Key value sequence seperate by whitespace.'
             'e.g. [key] [value] [key] [value]',
        default=[], nargs='+')

    parser.add_argument("--random-seed", type=int, default=64, help="Random seed")


def maybe_create_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def check_global_arguments(args):

    torch.set_num_threads(args.workers)
    if args.workers != torch.get_num_threads():
        print("Warning: # of threads is only ", torch.get_num_threads())

    setattr(args, "fixed_batch_path", os.path.join(args.logdir, args.dataset, args.exp, "fixed_batch.pt"))
    args.logdir = os.path.join(args.logdir, args.dataset, args.exp, args.run)
    maybe_create_dir(args.logdir)
    #print("Saving events in: {}".format(args.logdir))

    #
    # Model directories
    #
    args.snapshot_dir = os.path.join(args.snapshot_dir, args.dataset, args.exp, args.run)
    maybe_create_dir(args.snapshot_dir)
    #print("Saving snapshots in: {}".format(args.snapshot_dir))

def get_arguments(args_in):
    """Parse all the arguments provided from the CLI.
    
    Returns:
      A list of parsed arguments.
    """
    parser = argparse.ArgumentParser(description="Model Evaluation")

    add_global_arguments(parser)
    args = parser.parse_args(args_in)
    check_global_arguments(args)

    return args
