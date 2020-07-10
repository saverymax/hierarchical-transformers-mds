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
        super(LabelSmoothingLoss, self).__init__()
        self.ignore_index = ignore_index
        self.vocab_size = tgt_vocab_size
        smoothing_value = label_smoothing / (tgt_vocab_size)
        one_hot = torch.full((tgt_vocab_size,), smoothing_value)
        # Don't include any values in this index in the loss
        one_hot[self.ignore_index] = 0
        self.register_buffer('one_hot', one_hot.unsqueeze(0))
        self.confidence = 1.0 - label_smoothing

    def forward(self, output, target):
        """
        output (FloatTensor): batch_size x vocab_size x max_seq_len
        target (LongTensor): batch_size x max_seq_len
        """
        # Flatten output and target
        logging.info("smooooothn loss")
        output = output.transpose(1, 2).contiguous().view(-1, self.vocab_size)
        logging.info("initial incoming output %s", output.size())
        logging.info("output %s", output)
        logging.info("initial arget %s", target.size())
        logging.info("target %s", target)
        target = target.contiguous().view(-1)
        logging.info("target rehspad %s", target.size())
        logging.info("target %s", target)
        output = nn.functional.log_softmax(output, dim=-1)
        logging.info("output after soft %s", output)
        #output.masked_fill_((target == self.ignore_index).unsqueeze(1), 0)
        #logging.info("incoming output after mask %s", output.size())
        #logging.info("output %s", output)
        assert output.size() == torch.Size([target.size()[0], self.vocab_size])
        model_prob = self.one_hot.repeat(target.size(0), 1)
        logging.info("initial model prob %s", model_prob.size())
        logging.info("initial model prob %s", model_prob)
        model_prob.scatter_(1, target.unsqueeze(1), self.confidence)
        logging.info("scattered model prob %s", model_prob.size())
        logging.info("scattered model prob %s", model_prob)
        # Mask out based on the padding index provided in init
        model_prob.masked_fill_((target == self.ignore_index).unsqueeze(1), 0)
        logging.info("maskd model prob %s", model_prob.size())
        logging.info("maskd model prob %s", model_prob)
        # Returns sum(outputs * (log(outputs)-labels)
        loss = nn.functional.kl_div(output, model_prob, reduction='sum')
        logging.info("kl loss %s", loss.size())
        logging.info("kl loss %s", loss)
        return loss
