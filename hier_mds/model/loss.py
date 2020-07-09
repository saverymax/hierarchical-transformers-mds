"""
Module for Label smoothing loss
OpenNMT https://github.com/OpenNMT/OpenNMT-py/blob/master/onmt/utils/loss.py
and Pyorch issue https://github.com/pytorch/pytorch/issues/7455
used as reference
"""
import logging

import torch
from torch import nn

class LabelSmoothingLoss(nn.Module):
    """
    Loss computation from OpenNMT
    With label smoothing,
    KL-divergence between q_{smoothed ground truth prob.}(w)
    and p_{prob. computed by model}(w) is minimized.
    """
    def __init__(self, label_smoothing, tgt_vocab_size, ignore_index=1):
        assert 0.0 < label_smoothing <= 1.0
        self.ignore_index = ignore_index
        super(LabelSmoothingLoss, self).__init__()
        self.vocab_size = tgt_vocab_size
        smoothing_value = label_smoothing / (tgt_vocab_size - 2)
        one_hot = torch.full((tgt_vocab_size,), smoothing_value)
        one_hot[self.ignore_index] = 0
        self.register_buffer('one_hot', one_hot.unsqueeze(0))

        self.confidence = 1.0 - label_smoothing

    def forward(self, output, target):
        """
        output (FloatTensor): batch_size x vocab_size x max_seq_len
        target (LongTensor): batch_size x max_seq_len
        """
        # Flatten output and target
        output = output.transpose(1, 2).contiguous().view(-1, self.vocab_size)
        target = target.contiguous().view(-1)
        assert output.size() == torch.Size([target.size()[0], self.vocab_size])
        model_prob = self.one_hot.repeat(target.size(0), 1)
        model_prob.scatter_(1, target.unsqueeze(1), self.confidence)
        model_prob.masked_fill_((target == self.ignore_index).unsqueeze(1), 0)

        return nn.functional.kl_div(output, model_prob, reduction='sum')


class LabelSmoothedCrossEntropyLoss(nn.Module):
    """this loss performs label smoothing to compute cross-entropy with soft labels, when smoothing=0.0, this
    is the same as torch.nn.CrossEntropyLoss"""

    def __init__(self, n_classes, smoothing=0.0, dim=-1):
        super().__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.n_classes = n_classes
        self.dim = dim

    def forward(self, pred, target):
        pred = pred.log_softmax(dim=self.dim)
        with torch.no_grad():
            true_dist = torch.zeros_like(pred)
            true_dist.fill_(self.smoothing / (self.n_classes - 1))
            true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)

        return torch.mean(torch.sum(-true_dist * pred, dim=self.dim))
