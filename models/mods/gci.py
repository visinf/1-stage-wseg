import torch
import torch.nn as nn
import torch.nn.functional as F


class GCI(nn.Module):
    """Global Cue Injection
    Takes shallow features with low receptive
    field and augments it with global info via
    adaptive instance normalisation"""

    def __init__(self, NormLayer=nn.BatchNorm2d):
        super(GCI, self).__init__()

        self.NormLayer = NormLayer
        self.from_scratch_layers = []

        self._init_params()

    def _conv2d(self, *args, **kwargs):
        conv = nn.Conv2d(*args, **kwargs)
        self.from_scratch_layers.append(conv)
        torch.nn.init.kaiming_normal_(conv.weight)
        return conv

    def _bnorm(self, *args, **kwargs):
        bn = self.NormLayer(*args, **kwargs)
        #self.bn_learn.append(bn)
        self.from_scratch_layers.append(bn)
        if not bn.weight is None:
            bn.weight.data.fill_(1)
            bn.bias.data.zero_()
        return bn

    def _init_params(self):

        self.fc_deep = nn.Sequential(self._conv2d(256, 512, 1, bias=False), \
                                     self._bnorm(512), nn.ReLU())

        self.fc_skip = nn.Sequential(self._conv2d(256, 256, 1, bias=False), \
                                     self._bnorm(256, affine=False))

        self.fc_cls = nn.Sequential(self._conv2d(256, 256, 1, bias=False), \
                                    self._bnorm(256), nn.ReLU()) 

    def forward(self, x, y):
        """Forward pass

        Args:
            x: shalow features
            y: deep features
        """

        # extract global attributes
        y = self.fc_deep(y)
        attrs, _ = y.view(y.size(0), y.size(1), -1).max(-1)

        # pre-process shallow features
        x = self.fc_skip(x)
        x = F.relu(self._adin_conv(x, attrs))

        return self.fc_cls(x)

    def _adin_conv(self, x, y):

        bs, num_c, _, _ = x.size()
        assert 2*num_c == y.size(1), "AdIN: dimension mismatch"

        y = y.view(bs, 2, num_c)
        gamma, beta = y[:, 0], y[:, 1]

        gamma = gamma.unsqueeze(-1).unsqueeze(-1)
        beta = beta.unsqueeze(-1).unsqueeze(-1)

        return x * (gamma + 1) + beta
