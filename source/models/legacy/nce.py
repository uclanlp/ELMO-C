"""A generic NCE wrapper which speedup the training and inferencing"""

import torch
import torch.nn as nn
from torch.autograd import Variable


class AliasMultinomial(torch.nn.Module):
    '''Alias sampling method to speedup multinomial sampling

    The alias method treats multinomial sampling as a combination of uniform sampling and
    bernoulli sampling. It achieves significant acceleration when repeatedly sampling from
    the save multinomial distribution.

    Attributes:
        - probs: the probability density of desired multinomial distribution

    Refs:
        - https://hips.seas.harvard.edu/blog/2013/03/03/the-alias-method-efficient-sampling-with-many-discrete-outcomes/
    '''
    def __init__(self, probs):
        super(AliasMultinomial, self).__init__()

        probs = probs / probs.sum()
        cpu_probs = probs.cpu()
        K = len(probs)

        # such a name helps to avoid the namespace check for nn.Module
        self_prob = [0] * K
        self_alias = [0] * K

        # Sort the data into the outcomes with probabilities
        # that are larger and smaller than 1/K.
        smaller = []
        larger = []
        for idx, prob in enumerate(cpu_probs):
            self_prob[idx] = K*prob
            if self_prob[idx] < 1.0:
                smaller.append(idx)
            else:
                larger.append(idx)

        # Loop though and create little binary mixtures that
        # appropriately allocate the larger outcomes over the
        # overall uniform mixture.
        while len(smaller) > 0 and len(larger) > 0:
            small = smaller.pop()
            large = larger.pop()

            self_alias[small] = large
            self_prob[large] = (self_prob[large] - 1.0) + self_prob[small]

            if self_prob[large] < 1.0:
                smaller.append(large)
            else:
                larger.append(large)

        for last_one in smaller+larger:
            self_prob[last_one] = 1

        self.register_buffer('prob', torch.Tensor(self_prob))
        self.register_buffer('alias', torch.LongTensor(self_alias))

    def draw(self, *size):
        """Draw N samples from multinomial

        Args:
            - size: the output size of samples
        """
        max_value = self.alias.size(0)

        kk = self.alias.new(*size).random_(0, max_value).long().view(-1)
        prob = self.prob[kk]
        alias = self.alias[kk]
        # b is whether a random number is greater than q
        b = torch.bernoulli(prob).long()
        oq = kk.mul(b)
        oj = alias.mul(1 - b)

        return (oq + oj).view(size)


class NCELoss(nn.Module):
    """Noise Contrastive Estimation

    NCE is to eliminate the computational cost of softmax
    normalization.

    There are two modes in this NCELoss module:
        - nce: enable the NCE approximtion
        - ce: use the original cross entropy as default loss
    They can be switched by calling function `enable_nce()` or
    `disable_nce()`, you can also switch on/off via `nce_mode(True/False)`

    Ref:
        X.Chen etal Recurrent neural network language
        model training with noise contrastive estimation
        for speech recognition
        https://core.ac.uk/download/pdf/42338485.pdf

    Attributes:
        noise: the distribution of noise
        noise_ratio: $\frac{#noises}{#real data samples}$ (k in paper)
        norm_term: the normalization term (lnZ in paper)
        size_average: average the loss by batch size
        reduce: returned the loss for each target_idx if True,
        this will ignore the value of `size_average`
        loss_type: loss type of this module, currently 'full', 'sampled', 'nce'
        are supported

    Shape:
        - noise: :math:`(V)` where `V = vocabulary size`
        - target: :math:`(B, N)`
        - loss: :math:`(B, N)` if `reduce=True`

    Input:
        target: the supervised training label.
        args&kwargs: extra arguments passed to underlying index module

    Return:
        loss: if `reduce=False` the scalar NCELoss Variable ready for backward,
        else the loss matrix for every individual targets.

    Shape:
    """

    def __init__(self,
                 noise,
                 noise_ratio=100,
                 norm_term=13,
                 size_average=True,
                 reduce=True,
                 per_word=False,
                 loss_type='nce',
                 ):
        super(NCELoss, self).__init__()

        self.register_buffer('noise', noise)
        self.alias = AliasMultinomial(noise)
        self.noise_ratio = noise_ratio
        self.norm_term = norm_term
        self.size_average = size_average
        self.reduce = reduce
        self.per_word = per_word
        self.bce = nn.BCELoss(reduce=False)
        self.ce = nn.CrossEntropyLoss(reduce=False)
        self.loss_type = loss_type

    def forward(self, target, *args, **kwargs):
        """compute the loss with output and the desired target

        The `forward` is the same among all NCELoss submodules, it
        takes care of generating noises and calculating the loss
        given target and noise scores.
        """
        batch = target.size(0)
        max_len = target.size(1)
        if self.loss_type != 'full':

            noise_samples = self.get_noise(batch, max_len)

            # B,N,Nr
            prob_noise = Variable(
                self.noise[noise_samples.data.view(-1)].view_as(noise_samples)
            ).cuda()
            prob_target_in_noise = Variable(
                self.noise[target.data.view(-1)].view_as(target)
            ).cuda()

            # (B,N), (B,N,Nr)
            prob_model, prob_noise_in_model = self._get_prob(target, noise_samples, *args, **kwargs)

            if self.loss_type == 'nce':
                #if self.training:
                loss = self.nce_loss(
                        prob_model, prob_noise_in_model,
                        prob_noise, prob_target_in_noise,
                    )
                #else:
                # directly output the approximated posterior
                #    loss = - prob_model.log()
            elif self.loss_type == 'sampled':
                loss = self.sampled_softmax_loss(
                    prob_model, prob_noise_in_model,
                    prob_noise, prob_target_in_noise,
                )
            elif self.loss_type == 'mix' and self.training:
                loss = 0.5 * self.nce_loss(
                    prob_model, prob_noise_in_model,
                    prob_noise, prob_target_in_noise,
                )
                loss += 0.5 * self.sampled_softmax_loss(
                    prob_model, prob_noise_in_model,
                    prob_noise, prob_target_in_noise,
                )
            elif self.loss_type == 'pmi':
                loss = self.PMI_loss(
                        prob_model, prob_noise_in_model,
                        prob_noise, prob_target_in_noise,
                    )

            else:
                current_stage = 'training' if self.training else 'inference'
                raise NotImplementedError('loss type {} not implemented at {}'.format(self.loss_type, current_stage))

        else:
            # Fallback into conventional cross entropy
            loss = self.ce_loss(target, *args, **kwargs)

        if self.reduce:
            if self.size_average:
                return loss.mean()
            else:
                return loss.sum()
        else:
            return loss

    def get_noise(self, batch_size, max_len):
        """Generate noise samples from noise distribution"""

        if self.per_word:
            noise_samples = self.alias.draw(
                batch_size,
                max_len,
                self.noise_ratio,
            )
        else:
            noise_samples = self.alias.draw(1, 1, self.noise_ratio).expand(batch_size, max_len, self.noise_ratio)

        noise_samples = Variable(noise_samples).contiguous()
        return noise_samples

    def _get_prob(self, target_idx, noise_idx, *args, **kwargs):
        """Get the NCE estimated probability for target and noise

        Shape:
            - Target_idx: :math:`(N)`
            - Noise_idx: :math:`(N, N_r)` where `N_r = noise ratio`
        """

        MIN_PROB = 1e-9  # a minimal probability for numerical stability
        target_score, noise_score = self.get_score(target_idx, noise_idx, *args, **kwargs)

        target_prob = target_score.sub(self.norm_term).exp()
        target_prob.data.clamp_(MIN_PROB, 1)
        noise_prob = noise_score.sub(self.norm_term).exp()
        noise_prob.data.clamp_(MIN_PROB, 1)
        return target_prob, noise_prob

    def get_score(self, target_idx, noise_idx, *args, **kwargs):
        """Get the target and noise scores given input

        This method should be override by inherit classes

        Returns:
            - target_score: real valued score for each target index
            - noise_score: real valued score for each noise index
        """
        raise NotImplementedError()

    def ce_loss(self, target_idx, *args, **kwargs):
        """Get the conventional CrossEntropyLoss

        The returned loss should be of the same size of `target`

        Args:
            - target_idx: batched target index
            - args, kwargs: any arbitrary input if needed by sub-class

        Returns:
            - loss: the estimated loss for each target
        """
        raise NotImplementedError()

    def nce_loss(self, prob_model, prob_noise_in_model, prob_noise, prob_target_in_noise):
        """Compute the classification loss given all four probabilities

        Args:
            - prob_model: probability of target words given by the model (RNN)
            - prob_noise_in_model: probability of noise words given by the model
            - prob_noise: probability of noise words given by the noise distribution
            - prob_target_in_noise: probability of target words given by the noise distribution

        Returns:
            - loss: a mis-classification loss for every single case
        """

        p_model = torch.cat([prob_model.unsqueeze(2), prob_noise_in_model], dim=2)
        p_noise = torch.cat([prob_target_in_noise.unsqueeze(2), prob_noise], dim=2)

        # predicted probability of the word comes from true data distribution
        p_true = p_model / (p_model + self.noise_ratio * p_noise)
        label = torch.cat(
            [torch.ones_like(prob_model).unsqueeze(2),
             torch.zeros_like(prob_noise)], dim=2
        )

        loss = self.bce(p_true, label).sum(dim=2)

        return loss

    def PMI_loss(self, prob_model, prob_noise_in_model, prob_noise, prob_target_in_noise):
        """Compute the classification loss given all four probabilities

        Args:
            - prob_model: probability of target words given by the model (RNN)
            - prob_noise_in_model: probability of noise words given by the model
            - prob_noise: probability of noise words given by the noise distribution
            - prob_target_in_noise: probability of target words given by the noise distribution

        Returns:
            - loss: a mis-classification loss for every single case
        """

        prob_model = prob_model.sigmoid().log()
        prob_noise_in_model = -prob_noise_in_model
        prob_noise_in_model = prob_noise_in_model.sigmoid().log().sum(-1)

        return -(prob_model + prob_noise_in_model)


    def sampled_softmax_loss(self, prob_model, prob_noise_in_model, prob_noise, prob_target_in_noise):
        """Compute the sampled softmax loss based on the tensorflow's impl"""
        logits = torch.cat([prob_model.unsqueeze(2), prob_noise_in_model], dim=2).log()
        q_logits = torch.cat([prob_target_in_noise.unsqueeze(2), prob_noise], dim=2).log()
        # subtract Q for correction of biased sampling
        logits = logits - q_logits
        labels = torch.zeros_like(logits.narrow(2, 0, 1)).squeeze(2).long()
        loss = self.ce(
            logits.view(-1, logits.size(-1)),
            labels.view(-1),
        ).view_as(labels)

        return loss

class NCEWrapper(NCELoss):
    def __init__(self,
                 word_vector,
                  *args, **kwargs):
        super(NCEWrapper, self).__init__(*args, **kwargs)
        self.word_vectors = word_vector

    def get_score(self, target_idx, noise_idx, logits, mask, dummy = 1):
        """Get the target and noise scores given input

        This method should be override by inherit classes

        Returns:
            - target_score: real valued score for each target index
            - noise_score: real valued score for each noise index
        """
        selected_word_embeddings = torch.index_select(self.word_vectors, 0, target_idx.view(-1)).cuda()
        logits = logits.view(-1, logits.size(-1))
        scores_target = torch.bmm(logits.unsqueeze(1), selected_word_embeddings.unsqueeze(-1)).squeeze(-1).squeeze(-1)

        if not self.per_word:
            noise_idx = noise_idx.view(-1, noise_idx.size(-1))
            noise_idx = noise_idx[0] # The noisy index is the same for all words

        batch_num = target_idx.size(0)
        token_len = target_idx.size(1)

        token_num = selected_word_embeddings.size(0)
        noisy_num = noise_idx.size(-1)
        embedding_dim = self.word_vectors.size(-1)
        selected_word_embeddings = torch.index_select(self.word_vectors, 0, noise_idx.view(-1)).cuda()

        # expand logits to token_num x 1 x 300, word_embedding to token_num x 300 x nr
        scores_noisy = torch.bmm(logits.unsqueeze(1), selected_word_embeddings.transpose(0, 1).unsqueeze(0).expand(token_num, embedding_dim ,noisy_num) ).squeeze(1)

        # scores_noisy: token_num x nr
        # target_score: token_num
        return scores_target.view(batch_num, token_len), scores_noisy.view(batch_num, token_len, noisy_num)
