from functools import partial
from .stage_net import network_factory

def get_model(cfg, *args, **kwargs):
    net = partial(network_factory(cfg), config=cfg, pre_weights=cfg.PRE_WEIGHTS_PATH)
    return net(*args, **kwargs)
