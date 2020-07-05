from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals


import yaml
import six
import os
import os.path as osp
import copy
from ast import literal_eval

import numpy as np
from packaging import version

from utils.collections import AttrDict

__C = AttrDict()
# Consumers can get config by:
#   from fast_rcnn_config import cfg
cfg = __C

__C.NUM_GPUS = 1
# Random note: avoid using '.ON' as a config key since yaml converts it to True;
# prefer 'ENABLED' instead

# ---------------------------------------------------------------------------- #
# Training options
# ---------------------------------------------------------------------------- #
__C.TRAIN = AttrDict()
__C.TRAIN.BATCH_SIZE = 20
__C.TRAIN.NUM_EPOCHS = 15
__C.TRAIN.NUM_WORKERS = 4
__C.TRAIN.MASK_LOSS = 0.0
__C.TRAIN.PRETRAIN = 5

# ---------------------------------------------------------------------------- #
# Inference options
# ---------------------------------------------------------------------------- #
__C.TEST = AttrDict()
__C.TEST.METHOD = "multiscale" # multiscale | crop
__C.TEST.DATA_ROOT = "/data/your_directory"
__C.TEST.SCALES = [1, 0.5, 1.5, 2.0]
__C.TEST.FLIP = True
__C.TEST.PAD_SIZE = [1024, 1024]
__C.TEST.CROP_SIZE = [448, 448]
__C.TEST.CROP_GRID_SIZE = [2, 2]
__C.TEST.BATCH_SIZE = 8
__C.TEST.BG_POW = 3
__C.TEST.NUM_CLASSES = 21

# use ground-truth labels to remove
# false positive masks
__C.TEST.USE_GT_LABELS = False

# if class confidence does not exceed this threshold
# the mask is removed (count as false positive)
# used only if MASKS.USE_GT_LABELS is False
__C.TEST.FP_CUT_SCORE = 0.1

# ---------------------------------------------------------------------------- #
# Dataset options
# ---------------------------------------------------------------------------- #
__C.DATASET = AttrDict()

__C.DATASET.CROP_SIZE = 321
__C.DATASET.SCALE_FROM = 0.9
__C.DATASET.SCALE_TO = 1.0
__C.DATASET.PATH = "data/images"

# ---------------------------------------------------------------------------- #
# Network options
# ---------------------------------------------------------------------------- #
__C.NET  = AttrDict()
__C.NET.MODEL = 'vgg16'
__C.NET.BACKBONE = 'resnet50'
__C.NET.PRE_WEIGHTS_PATH = ""
__C.NET.OPT = 'SGD'
__C.NET.LR = 0.001
__C.NET.BETA1 = 0.5
__C.NET.MOMENTUM = 0.9
__C.NET.WEIGHT_DECAY = 1e-5
__C.NET.LOSS = 'SoftMargin'
__C.NET.MASK_LOSS_BCE = 1.0
__C.NET.BG_SCORE = 0.1 # background score (only for CAM)
__C.NET.FOCAL_P = 3
__C.NET.FOCAL_LAMBDA = 0.01
__C.NET.PAMR_KERNEL = [1, 2, 4, 8, 12, 24]
__C.NET.PAMR_ITER = 10
__C.NET.SG_PSI = 0.3

# Mask Inference
__C.MASKS = AttrDict()

# CRF options
__C.MASKS.CRF = AttrDict()
__C.MASKS.CRF.ALPHA_LOW = 4
__C.MASKS.CRF.ALPHA_HIGH = 32

# [Infered value]
__C.CUDA = False

__C.DEBUG = False

# [Infered value]
__C.PYTORCH_VERSION_LESS_THAN_040 = False

def assert_and_infer_cfg(make_immutable=True):
    """Call this function in your script after you have finished setting all cfg
    values that are necessary (e.g., merging a config from a file, merging
    command line config options, etc.). By default, this function will also
    mark the global cfg as immutable to prevent changing the global cfg settings
    during script execution (which can lead to hard to debug errors or code
    that's harder to understand than is necessary).
    """
    if make_immutable:
        cfg.immutable(True)


def merge_cfg_from_file(cfg_filename):
    """Load a yaml config file and merge it into the global config."""
    with open(cfg_filename, 'r') as f:
        if hasattr(yaml, "FullLoader"):
            yaml_cfg = AttrDict(yaml.load(f, Loader=yaml.FullLoader))
        else:
            yaml_cfg = AttrDict(yaml.load(f))

    _merge_a_into_b(yaml_cfg, __C)

cfg_from_file = merge_cfg_from_file


def merge_cfg_from_cfg(cfg_other):
    """Merge `cfg_other` into the global config."""
    _merge_a_into_b(cfg_other, __C)


def merge_cfg_from_list(cfg_list):
    """Merge config keys, values in a list (e.g., from command line) into the
    global config. For example, `cfg_list = ['TEST.NMS', 0.5]`.
    """
    assert len(cfg_list) % 2 == 0
    for full_key, v in zip(cfg_list[0::2], cfg_list[1::2]):
        key_list = full_key.split('.')
        d = __C
        for subkey in key_list[:-1]:
            assert subkey in d, 'Non-existent key: {}'.format(full_key)
            d = d[subkey]
        subkey = key_list[-1]
        assert subkey in d, 'Non-existent key: {}'.format(full_key)
        value = _decode_cfg_value(v)
        value = _check_and_coerce_cfg_value_type(
            value, d[subkey], subkey, full_key
        )
        d[subkey] = value

cfg_from_list = merge_cfg_from_list


def _merge_a_into_b(a, b, stack=None):
    """Merge config dictionary a into config dictionary b, clobbering the
    options in b whenever they are also specified in a.
    """
    assert isinstance(a, AttrDict), 'Argument `a` must be an AttrDict'
    assert isinstance(b, AttrDict), 'Argument `b` must be an AttrDict'

    for k, v_ in a.items():
        full_key = '.'.join(stack) + '.' + k if stack is not None else k
        # a must specify keys that are in b
        if k not in b:
            raise KeyError('Non-existent config key: {}'.format(full_key))

        v = copy.deepcopy(v_)
        v = _decode_cfg_value(v)
        v = _check_and_coerce_cfg_value_type(v, b[k], k, full_key)

        # Recursively merge dicts
        if isinstance(v, AttrDict):
            try:
                stack_push = [k] if stack is None else stack + [k]
                _merge_a_into_b(v, b[k], stack=stack_push)
            except BaseException:
                raise
        else:
            b[k] = v


def _decode_cfg_value(v):
    """Decodes a raw config value (e.g., from a yaml config files or command
    line argument) into a Python object.
    """
    # Configs parsed from raw yaml will contain dictionary keys that need to be
    # converted to AttrDict objects
    if isinstance(v, dict):
        return AttrDict(v)
    # All remaining processing is only applied to strings
    if not isinstance(v, six.string_types):
        return v
    # Try to interpret `v` as a:
    #   string, number, tuple, list, dict, boolean, or None
    try:
        v = literal_eval(v)
    # The following two excepts allow v to pass through when it represents a
    # string.
    #
    # Longer explanation:
    # The type of v is always a string (before calling literal_eval), but
    # sometimes it *represents* a string and other times a data structure, like
    # a list. In the case that v represents a string, what we got back from the
    # yaml parser is 'foo' *without quotes* (so, not '"foo"'). literal_eval is
    # ok with '"foo"', but will raise a ValueError if given 'foo'. In other
    # cases, like paths (v = 'foo/bar' and not v = '"foo/bar"'), literal_eval
    # will raise a SyntaxError.
    except ValueError:
        pass
    except SyntaxError:
        pass
    return v


def _check_and_coerce_cfg_value_type(value_a, value_b, key, full_key):
    """Checks that `value_a`, which is intended to replace `value_b` is of the
    right type. The type is correct if it matches exactly or is one of a few
    cases in which the type can be easily coerced.
    """
    # The types must match (with some exceptions)
    type_b = type(value_b)
    type_a = type(value_a)
    if type_a is type_b:
        return value_a

    # Exceptions: numpy arrays, strings, tuple<->list
    if isinstance(value_b, np.ndarray):
        value_a = np.array(value_a, dtype=value_b.dtype)
    elif isinstance(value_b, six.string_types):
        value_a = str(value_a)
    elif isinstance(value_a, tuple) and isinstance(value_b, list):
        value_a = list(value_a)
    elif isinstance(value_a, list) and isinstance(value_b, tuple):
        value_a = tuple(value_a)
    else:
        raise ValueError(
            'Type mismatch ({} vs. {}) with values ({} vs. {}) for config '
            'key: {}'.format(type_b, type_a, value_b, value_a, full_key)
        )
    return value_a
