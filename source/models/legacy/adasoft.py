import torch
from torch import nn
from torch.nn import functional as F
try:
    from torch.nn import LayerNorm
except:
    class LayerNorm(torch.nn.Module):
        def __init__(self, normalized_shape, eps=1e-5, elementwise_affine=True):
            super(LayerNorm, self).__init__()
            normalized_shape = (normalized_shape,)
            self.normalized_shape = torch.Size(normalized_shape)
            self.eps = eps
            self.elementwise_affine = elementwise_affine
            if self.elementwise_affine:
                self.weight = torch.nn.Parameter(torch.Tensor(*normalized_shape))
                self.bias = torch.nn.Parameter(torch.Tensor(*normalized_shape))
            else:
                self.register_parameter('weight', None)
                self.register_parameter('bias', None)
            self.reset_parameters()

        def reset_parameters(self):
            if self.elementwise_affine:
                self.weight.data.fill_(1)
                self.bias.data.zero_()

        def forward(self, x):
            if x.size(-1) == 1:
                return x
            mu = torch.mean(x, dim=-1)
            sigma = torch.std(x, dim=-1, unbiased=False)
            # HACK. PyTorch is changing behavior
            if mu.dim() == x.dim()-1:
                mu = mu.unsqueeze(mu.dim())
                sigma = sigma.unsqueeze(sigma.dim())
            output = (x - mu.expand_as(x)) / (sigma.expand_as(x) + self.eps)
            output = output.mul(self.weight.expand_as(output)) \
                + self.bias.expand_as(output)
            return output


class AdaptiveSoftmax(nn.Module):
    """Adaptive Softmax output layer

    Args:
        input_size: size of each input sample
        cutoff: indexes of words that splited into each bucket
        reduce_factor: dimension reduction factor of each tail bucket before projected
            to each words. Default: 4

    Shape:
        - input: (N, input_size)
        - target: (N)
        - output: [(N, cutoff[0] + len(cutoff) - 1), (N_1, cutoff[1] - cutoff[0]), ...]

    Attributes:
        head: the learnable weights of the module for head bucket
        tail: the learnable weights of the module for tail buckets

    Examples::

        >>> m = AdaptiveSoftmax(20, [2000, 10000])
        >>> input = torch.randn(128, 20)
        >>> target = torch.randint(low=0, high=10000, size=[128])
        >>> output = m(input, target)
        >>> log_prob = m.log_prob(input)
    """

    def __init__(self, input_size, cutoff, n_proj, reduce_factor=4, continue_softmax_from_emb = False):
        super().__init__()

        self.input_size = input_size
        self.cutoff = cutoff
        self.output_size = cutoff[0] + len(cutoff) - 1

        self.head = nn.Linear(input_size, self.output_size)
        self.tail = nn.ModuleList()
        self.continue_softmax_from_emb = continue_softmax_from_emb

        self.n_proj = n_proj
        if n_proj != input_size:
            self.projection = nn.Linear(n_proj, input_size)
            print("Using Projection {}".format(input_size))
        else:
            self.projection = None

        for i in range(len(cutoff) - 1):
            if reduce_factor == 1:
                seq = nn.Linear(input_size, cutoff[i + 1] - cutoff[i])

            else:
                seq = nn.Sequential(
                    nn.Linear(input_size, input_size // reduce_factor ** i, False),
                    nn.Linear(
                        input_size // reduce_factor ** i, cutoff[i + 1] - cutoff[i]
                    ),
                )

            self.tail.append(seq)
        if continue_softmax_from_emb:
            self.transform = nn.Linear(input_size, input_size)
            self.transform = LayerNorm(input_size)

    def set_target(self, target):
        self.id = []

        for i in range(len(self.cutoff) - 1):
            mask = target.ge(self.cutoff[i]).mul(target.lt(self.cutoff[i + 1]))

            if mask.any():
                self.id.append(mask.float().nonzero().squeeze(1))

            else:
                self.id.append(None)

    def forward(self, input, target=None):

        if self.projection:
            input = self.projection(input)

        if self.continue_softmax_from_emb:
            input = self.transform(input)
        output = [self.head(input)]

        if target is not None:
            self.set_target(target)

        for i in range(len(self.id)):
            if self.id[i] is not None:
                output.append(self.tail[i](input.index_select(0, self.id[i])))
            else:
                output.append(None)

        return output

    def log_prob(self, input):
        with torch.no_grad():
            head_out = self.head(input)

            batch_size = head_out.size(0)
            prob = torch.empty(batch_size, self.cutoff[-1], device=input.device)

            lsm_head = F.log_softmax(head_out, 1)
            prob[:, : self.cutoff[0]].copy_(lsm_head[:, : self.cutoff[0]])

            for i in range(len(self.tail)):
                split = lsm_head[:, self.cutoff[0] + i].unsqueeze(1)
                lsm_tail = F.log_softmax(self.tail[i](input), 1)
                prob[:, self.cutoff[i] : self.cutoff[i + 1]].copy_(lsm_tail).add_(split)

        return prob


class AdaptiveSoftmaxSparse(nn.Module):
    """Adaptive Softmax output layer

    Args:
        input_size: size of each input sample
        cutoff: indexes of words that splited into each bucket
        reduce_factor: dimension reduction factor of each tail bucket before projected
            to each words. Default: 4

    Shape:
        - input: (N, input_size)
        - target: (N)
        - output: [(N, cutoff[0] + len(cutoff) - 1), (N_1, cutoff[1] - cutoff[0]), ...]

    Attributes:
        head: the learnable weights of the module for head bucket
        tail: the learnable weights of the module for tail buckets

    Examples::

        >>> m = AdaptiveSoftmax(20, [2000, 10000])
        >>> input = torch.randn(128, 20)
        >>> target = torch.randint(low=0, high=10000, size=[128])
        >>> output = m(input, target)
        >>> log_prob = m.log_prob(input)
    """

    def __init__(self, input_size, cutoff, reduce_factor=4, continue_softmax_from_emb = False):
        super().__init__()

        self.input_size = input_size
        self.cutoff = cutoff
        self.output_size = cutoff[0] + len(cutoff) - 1

        self.head = nn.Linear(input_size, self.output_size)
        self.tail = nn.ModuleList()
        self.continue_softmax_from_emb = continue_softmax_from_emb

        for i in range(len(cutoff) - 1):
            if reduce_factor == 1:
                seq = nn.Linear(input_size, cutoff[i + 1] - cutoff[i])
            else:
                seq = nn.Sequential(
                    nn.Linear(input_size, input_size // reduce_factor ** i, False),
                    nn.Linear(
                        input_size // reduce_factor ** i, cutoff[i + 1] - cutoff[i]
                    ),
                )

            self.tail.append(seq)
        if continue_softmax_from_emb:
            self.transform = nn.Linear(input_size, input_size)
            self.transform = LayerNorm(input_size)

    def set_target(self, target):
        self.id = []

        for i in range(len(self.cutoff) - 1):
            mask = target.ge(self.cutoff[i]).mul(target.lt(self.cutoff[i + 1]))

            if mask.any():
                self.id.append(mask.float().nonzero().squeeze(1))

            else:
                self.id.append(None)

    def forward(self, input, target=None):

        if self.continue_softmax_from_emb:
            input = self.transform(input)
        output = [self.head(input)]

        if target is not None:
            self.set_target(target)

        for i in range(len(self.id)):
            if self.id[i] is not None:
                output.append(self.tail[i](input.index_select(0, self.id[i])))
            else:
                output.append(None)

        return output

    def log_prob(self, input):
        with torch.no_grad():
            head_out = self.head(input)

            batch_size = head_out.size(0)
            prob = torch.empty(batch_size, self.cutoff[-1], device=input.device)

            lsm_head = F.log_softmax(head_out, 1)
            prob[:, : self.cutoff[0]].copy_(lsm_head[:, : self.cutoff[0]])

            for i in range(len(self.tail)):
                split = lsm_head[:, self.cutoff[0] + i].unsqueeze(1)
                lsm_tail = F.log_softmax(self.tail[i](input), 1)
                prob[:, self.cutoff[i] : self.cutoff[i + 1]].copy_(lsm_tail).add_(split)

        return prob


class TiedAdaptiveSoftmax(nn.Module):
    """Adaptive Softmax that supports weight tying

    Args:
        weight: weight tensor for each words of shape [num_words, dim]
        cutoff: indexes of words that splited into each bucket

    Shape:
        - input: (N, input_size)
        - output: [(N, cutoff[0] + len(cutoff) - 1), (N_1, cutoff[1] - cutoff[0]), ...]

    Attributes:
        weight: the learnable weights of the module that tied with specified tensor
        biases: the learnable biases of the module

    Examples::

        >>> m = TiedAdaptiveSoftmax(20, [2000, 10000])
        >>> input = torch.randn(128, 20)
        >>> target = torch.randint(low=0, high=10000, size=[128])
        >>> output = m(input, target)
        >>> log_prob = m.log_prob(input)
    """

    def __init__(self, weight, cutoff):
        super().__init__()

        self.weight = weight
        self.biases = nn.ParameterList()
        self.biases.append(nn.Parameter(torch.zeros(cutoff[0])))
        for i in range(len(cutoff) - 1):
            self.biases.append(nn.Parameter(torch.zeros(cutoff[i + 1] - cutoff[i])))

        self.split = nn.Linear(weight.shape[1], len(cutoff) - 1)
        self.cutoff = cutoff

    def set_target(self, target):
        self.id = []

        for i in range(len(self.cutoff) - 1):
            mask = target.ge(self.cutoff[i]).mul(target.lt(self.cutoff[i + 1]))

            if mask.any():
                self.id.append(mask.float().nonzero().squeeze(1))

            else:
                self.id.append(None)

    def forward(self, input, target=None):
        head = F.linear(input, self.weight[: self.cutoff[0]], self.biases[0])
        split = self.split(input)
        output = [torch.cat([head, split], 1)]

        if target is not None:
            self.set_target(target)

        for i in range(len(self.id)):
            if self.id[i] is not None:
                output.append(
                    F.linear(
                        input.index_select(0, self.id[i]),
                        self.weight[self.cutoff[i] : self.cutoff[i + 1]],
                        self.biases[i + 1],
                    )
                )
            else:
                output.append(None)

        return output

    def log_prob(self, input):
        with torch.no_grad():
            linear_out = F.linear(
                input, self.weight, torch.cat([p for p in self.biases])
            )
            split = self.split(input)
            head = F.log_softmax(
                torch.cat([linear_out[:, : self.cutoff[0]], split], 1), 1
            )
            linear_out[:, : self.cutoff[0]].copy_(head[:, : -split.shape[1]])

            for i in range(len(self.cutoff) - 1):
                part = linear_out[:, self.cutoff[i] : self.cutoff[i + 1]]
                part.copy_(F.log_softmax(part, 1))
                part.add_(head[:, self.cutoff[0] + i].unsqueeze(1))

        return linear_out


class AdaptiveLoss(nn.Module):
    """Loss layer for Adaptive Softmax

    Args:
        cutoff: indexes of words that splited into each bucket

    Shape:
        - input: [(N, cutoff[0] + len(cutoff) - 1), (N_1, cutoff[1] - cutoff[0]), ...]
        - target: (N)

    Examples::

        >>> cutoff = [2000, 10000]
        >>> m = AdaptiveSoftmax(20, cutoff)
        >>> criterion = AdaptiveLoss(cutoff)
        >>> input = torch.randn(128, 20)
        >>> target = torch.randint(low=0, high=10000, size=[128])
        >>> output = m(input, target)
        >>> loss = criterion(output, target)
        >>> loss.backward()
    """

    def __init__(self, cutoff):
        super().__init__()

        self.cutoff = cutoff

    def remap_target(self, target):
        new_target = [target.clone()]

        for i in range(len(self.cutoff) - 1):
            mask = target.ge(self.cutoff[i]).mul(target.lt(self.cutoff[i + 1]))
            new_target[0][mask] = self.cutoff[0] + i

            if mask.any():
                new_target.append(target[mask].add(-self.cutoff[i]))

            else:
                new_target.append(None)

        return new_target

    def forward(self, input, target):
        batch_size = input[0].size(0)
        target = self.remap_target(target.data)

        output = 0.0

        for i in range(len(input)):
            if input[i] is not None:
                assert target[i].min() >= 0 and target[i].max() <= input[i].size(1)
                output = output + F.cross_entropy(
                    input[i], target[i], size_average=False
                )

        output /= batch_size

        return output