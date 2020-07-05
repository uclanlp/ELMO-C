import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import math

from log_uniform.log_uniform import LogUniformSampler
import sys
import util

class SampledSoftmax(nn.Module):
    def __init__(self, ntokens, nsampled, nhid, tied_weight, n_proj, sparse = True):
        super(SampledSoftmax, self).__init__()

        # Parameters
        self.ntokens = ntokens
        self.nsampled = nsampled

        self.sampler = LogUniformSampler(self.ntokens)

        self.n_proj = n_proj
        if n_proj != nhid:
            self.projection = nn.Linear(n_proj, nhid)
        else:
            self.projection = None

        if not sparse:
            self.params = nn.Linear(nhid, ntokens)
            if tied_weight is not None:
                self.params.weight = tied_weight
            else:
                util.initialize(self.params.weight)
        else:
            self.softmax_b = torch.nn.Embedding(ntokens, 1, sparse=True)
            self.softmax_w = torch.nn.Embedding(ntokens, nhid, sparse=True)
        self.sparse = sparse

        print("-- Used sampled softmax with " + str(self.nsampled) + " samples.")

    def forward(self, inputs, labels):
        
        assert(self.training)

        if self.projection:
            inputs = self.projection(inputs)

        if self.training:
            # sample ids according to word distribution - Unique
            sample_values = self.sampler.sample(self.nsampled, labels.data.cpu().numpy())
            return self.sampled(inputs, labels, sample_values, remove_accidental_match=True)
        else:
            return self.full(inputs, labels)

    def sampled(self, inputs, labels, sample_values, remove_accidental_match=False):
        assert(inputs.data.get_device() == labels.data.get_device())
        device_id = labels.data.get_device()

        batch_size, d = inputs.size()
        sample_ids, true_freq, sample_freq = sample_values

        sample_ids = Variable(torch.LongTensor(sample_ids)).cuda(device_id)
        true_freq = Variable(torch.FloatTensor(true_freq)).cuda(device_id)
        sample_freq = Variable(torch.FloatTensor(sample_freq)).cuda(device_id)

        # gather true labels - weights and frequencies

        if self.sparse:
            sample_weights = self.softmax_w(sample_ids)
            sample_bias = self.softmax_b(sample_ids).squeeze(1)

            true_weights = self.softmax_w(labels)
            true_bias = self.softmax_b(labels).squeeze(1)
        else:
            true_weights = self.params.weight[labels, :]
            true_bias = self.params.bias[labels]
            # gather sample ids - weights and frequencies
            sample_weights = self.params.weight[sample_ids, :]
            sample_bias = self.params.bias[sample_ids]

        # calculate logits
        true_logits = torch.sum(torch.mul(inputs, true_weights), dim=1) + true_bias
        sample_logits = torch.matmul(inputs, torch.t(sample_weights)) + sample_bias
        # remove true labels from sample set
        if remove_accidental_match:
            acc_hits = self.sampler.accidental_match(labels.data.cpu().numpy(), sample_ids.data.cpu().numpy())
            acc_hits = list(zip(*acc_hits))
            sample_logits[acc_hits] = -1e37

        # perform correction
        true_logits = true_logits.sub(torch.log(true_freq))
        sample_logits = sample_logits.sub(torch.log(sample_freq))

        # return logits and new_labels
        logits = torch.cat((torch.unsqueeze(true_logits, dim=1), sample_logits), dim=1)
        new_targets = Variable(torch.zeros(batch_size).long()).cuda(device_id)
        return logits, new_targets

    def full(self, inputs, labels):
        return self.params(inputs), labels
