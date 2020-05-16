import torch
import torch.nn as nn
import torch.nn.functional as F


class StochasticGate(nn.Module):
    """Stochastically merges features from two levels 
    with varying size of the receptive field
    """

    def __init__(self):
        super(StochasticGate, self).__init__()
        self._mask_drop = None

    def forward(self, x1, x2, alpha_rate=0.3):
        """Stochastic Gate (SG)

        SG stochastically mixes deep and shallow features
        at training time and deterministically combines 
        them at test time with a hyperparam. alpha
        """

        if self.training: # training time
            # dropout: selecting either x1 or x2
            if self._mask_drop is None:
                bs, c, h, w = x1.size()
                assert c == x2.size(1), "Number of features is different"
                self._mask_drop = torch.ones_like(x1)

            # a mask of {0,1}
            mask_drop = (1 - alpha_rate) * F.dropout(self._mask_drop, alpha_rate)

            # shift and scale deep features
            # at train time: E[x] = x1
            x1 = (x1 - alpha_rate * x2) / max(1e-8, 1 - alpha_rate)

            # combine the features
            x = mask_drop * x1 + (1 - mask_drop) * x2
        else:
            # inference time: deterministic
            x = (1 - alpha_rate) * x1 + alpha_rate * x2

        return x

