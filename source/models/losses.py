import torch
from torch.nn import functional
from torch.autograd import Variable
import numpy
import numpy as np
import torch.nn as nn
from sklearn.neighbors import NearestNeighbors
import math
import torch
import numpy as np
import scipy.special
from numbers import Number

from ..helpers import *
class IveFunction(torch.autograd.Function):
    @staticmethod
    def forward(self, v, z):
        
        assert isinstance(v, Number), 'v must be a scalar'
        
        self.save_for_backward(z)
        self.v = v
        z_cpu = z.data.cpu().numpy()

        if np.isclose(v, 0):
            output = scipy.special.i0e(z_cpu, dtype=z_cpu.dtype)
        elif np.isclose(v, 1):
            output = scipy.special.i1e(z_cpu, dtype=z_cpu.dtype)
        else: #  v > 0
            output = scipy.special.ive(v, z_cpu, dtype=np.float64)
        return torch.from_numpy(np.array(output)).to(z.device)

    @staticmethod
    def backward(self, grad_output):
        z = self.saved_tensors[-1]
        #print(grad_output.double() * (ive(self.v - 1, z) - ive(self.v, z) * (self.v + z).double() / z.double()))
        return None, (grad_output.double() * (   ive(self.v - 1, z) - ive(self.v, z) * (self.v + z).double() / z.double())).float()

class IvFunction(torch.autograd.Function):

    @staticmethod
    def forward(self, v, z):
        
        assert isinstance(v, Number), 'v must be a scalar'
        
        self.save_for_backward(z)
        self.v = v
        z_cpu = z.data.cpu().numpy()
        output = scipy.special.iv(v, z_cpu, dtype=np.float64)
        return torch.from_numpy(np.array(output)).to(z.device)

    @staticmethod
    def backward(self, grad_output):
        z = self.saved_tensors[-1]
        #print(grad_output.double() * (ive(self.v - 1, z) - ive(self.v, z) * (self.v + z).double() / z.double()))
        return None, ( grad_output.cpu().double() * scipy.special.ivp(self.v, z.double()) ).float().cuda()

class LogIvFunctionWithUpperBoundGradient(torch.autograd.Function):
    @staticmethod
    def forward(self, v, z):
        assert isinstance(v, Number), 'v must be a scalar'
        self.save_for_backward(z)
        self.v = v
        z_cpu = z.data.cpu().numpy()
        output = scipy.special.iv(v, z_cpu, dtype=np.float64)

        # Automatically switches from iv to ive if overflow
        output = np.log(output)
        #print(output)
        for i in range(output.shape[0]):
            if np.isinf(output[i]):
                output[i] = np.log(scipy.special.ive(v, z_cpu[i], dtype=np.float64)) + z_cpu[i]
        return torch.from_numpy(np.array(output)).to(z.device)

        # Prevent overflow and underflow automatically
        '''if np.isinf(output.sum()):
            output = scipy.special.iv(v, z_cpu, dtype=np.float64)
            return torch.log(torch.from_numpy(np.array(output)).to(z.device))
        else:    
            return torch.log(torch.from_numpy(np.array(output)).to(z.device)) + z.double()'''

    @staticmethod
    def backward(self, grad_output):
        z = self.saved_tensors[-1]
        return None, ( grad_output.float() * (self.v / z + z / (self.v + torch.sqrt( (self.v+2) *(self.v+2) + z * z )) ) ).cuda()

class LogIvFunctionWithUpperBoundGradientConstant(torch.autograd.Function):
    @staticmethod
    def forward(self, v, z):
        assert isinstance(v, Number), 'v must be a scalar'
        self.save_for_backward(z)
        self.v = v  
        return z - z

    @staticmethod
    def backward(self, grad_output):
        z = self.saved_tensors[-1]
        return None, ( grad_output.float() * (self.v / z + z / (self.v + torch.sqrt( (self.v+2) *(self.v+2) + z * z )) ) ).cuda()


ive = IveFunction.apply
iv = IvFunction.apply
log_iv_approximate_gradient = LogIvFunctionWithUpperBoundGradient.apply
log_iv_approximate_gradient_return_constant = LogIvFunctionWithUpperBoundGradientConstant.apply


def masked_cross_entropy(logits, target, length):
    """
    Args:
        logits: A Variable containing a FloatTensor of size
            (batch, max_len, num_classes) which contains the
            unnormalized probability for each class.
        target: A Variable containing a LongTensor of size
            (batch, max_len) which contains the index of the true
            class for each corresponding step.
        length: A Variable containing a LongTensor of size (batch,)
            which contains the length of each data in a batch.

    Returns:
        loss: An average loss value masked by the length.
    """
    length = Variable(torch.LongTensor(length))
    # logits_flat: (batch * max_len, num_classes)
    logits_flat = logits.view(-1, logits.size(-1))
    # log_probs_flat: (batch * max_len, num_classes)
    log_probs_flat = functional.log_softmax(logits_flat)
    # target_flat: (batch * max_len, 1)
    target_flat = target.view(-1, 1)
    # losses_flat: (batch * max_len, 1)
    # cuda fix
    losses_flat = -torch.gather(log_probs_flat, dim=1, index=target_flat)
    # losses: (batch, max_len)
    losses = losses_flat.view(*target.size())
    # mask: (batch, max_len)
    mask = sequence_mask(sequence_length=length, max_len=target.size(1))
    if USE_CUDA:
        mask = mask.cuda()
        length = length.cuda()
    losses = losses * mask.float()
    loss = losses.sum() / length.float().sum()
    return loss

def masked_cross_entropy_mask(logits, target, mask):
    """
    Args:
        logits: A Variable containing a FloatTensor of size
            (batch, max_len, num_classes) which contains the
            unnormalized probability for each class.
        target: A Variable containing a LongTensor of size
            (batch, max_len) which contains the index of the true
            class for each corresponding step.
        length: A Variable containing a LongTensor of size (batch,)
            which contains the length of each data in a batch.

    Returns:
        loss: An average loss value masked by the length.
    """
    #print("Sperate:")
    #print(target)
    # logits_flat: (batch * max_len, num_classes)
    logits_flat = logits.view(-1, logits.size(-1))
    # log_probs_flat: (batch * max_len, num_classes)
    log_probs_flat = functional.log_softmax(logits_flat)
    # target_flat: (batch * max_len, 1)
    target_flat = target.view(-1, 1)
    # losses_flat: (batch * max_len, 1)
    losses_flat = -torch.gather(log_probs_flat, dim=1, index=target_flat)
    # losses: (batch, max_len)
    losses = losses_flat.view(*target.size())
    # mask: (batch, max_len)
    #print(losses)
    losses = losses * mask.float()
    #print(losses)

    loss = losses.sum() / mask.float().sum()
    return loss

def L1_mask(logits, target, mask, word_vectors, loss):
    """
    Args:
        logits: A Variable containing a FloatTensor of size
            (batch, max_len, num_classes) which contains the
            unnormalized probability for each class.
        target: A Variable containing a LongTensor of size
            (batch, max_len) which contains the index of the true
            class for each corresponding step.
        length: A Variable containing a LongTensor of size (batch,)
            which contains the length of each data in a batch.

    Returns:
        loss: An average loss value masked by the length.
    """
    selected_word_embeddings = torch.index_select(word_vectors, 0, target.view(-1))
    return_loss = loss(logits.view(-1, selected_word_embeddings.size(1)), selected_word_embeddings)
    return_loss = return_loss * mask.contiguous().view(-1).unsqueeze(1).expand_as(return_loss).float()
    return return_loss.sum() / mask.float().sum()

def masked_cross_entropy_flat(logits, target, length):
    """
    Args:
        logits: A Variable containing a FloatTensor of size
            (batch, max_len, num_classes) which contains the
            unnormalized probability for each class.
        target: A Variable containing a LongTensor of size
            (batch, max_len) which contains the index of the true
            class for each corresponding step.
        length: A Variable containing a LongTensor of size (batch,)
            which contains the length of each data in a batch.

    Returns:
        loss: An average loss value masked by the length.
    """
    length = Variable(torch.LongTensor(length))
    mask = sequence_mask(sequence_length=length, max_len=target.size(0))
    if USE_CUDA:
        mask = mask.cuda()
        length = length.cuda()
    ## mask: (batch, max_len)
    all_loss = 0
    #logits is a list, every single elements is batch x num_classes
    for index, single in enumerate(logits):
        target_single = target[index] #target is batch, 1
        log_probs_flat = functional.log_softmax(single)

        losses_flat = -torch.gather(log_probs_flat, dim=1, index=target_single.unsqueeze(1))
        losses_flat = losses_flat * mask[:, index].unsqueeze(1).float()
        all_loss += losses_flat.sum()
    return all_loss / length.float().sum()

# for the classes below, we always assume batch first

# logits = batch x len x dim
# target = batch x len
# mask = batch x len

class StandardCrossEntrophy(torch.nn.Module):
    def __init__(self):
        super(StandardCrossEntrophy, self).__init__()
        self.loss = nn.CrossEntropyLoss(reduce = False)

    def forward(self, logits, target, mask, dummy = True):
        mask = mask.float().view(-1)
        loss =  self.loss(logits.contiguous().view(-1, logits.size(-1)),
                    target.contiguous().view(-1)) * mask
        return loss.sum() / mask.sum()

class CustomCrossEntrophy(torch.nn.Module):
    '''Currently no mask'''
    def __init__(self):
        super(CustomCrossEntrophy, self).__init__()

    def forward(self, logits, target, mask, dummy = True):
        logits_flat = logits.contiguous().view(-1, logits.size(-1))
        # log_probs_flat: (batch * max_len, num_classes)
        log_probs_flat = functional.log_softmax(logits_flat)
        # target_flat: (batch * max_len, 1)
        target_flat = target.view(-1, 1)
        # losses_flat: (batch * max_len, 1)
        losses_flat = -torch.gather(log_probs_flat, dim=1, index=target_flat)
        # losses: (batch, max_len)
        try:
            losses = losses_flat.view(*mask.size())
            # mask: (batch, max_len)
            losses = losses * mask.float()
            loss = losses.sum() / mask.float().sum()
        except:
            loss = losses_flat.sum() / losses_flat.size(0)
        return loss

class ROCLoss(torch.nn.Module):
    def __init__(self):
        super(ROCLoss, self).__init__()

    def forward(self, logits, target, mask, dummy = True):

        # these code are very nasty. This is because SRU does not provide support for stacke sequence.

        # Improve later

        M = mask.cpu().numpy()

        adjusted_vector = []

        length = []

        logits = logits.transpose(1,2)

        for i in range(M.shape[0]):
            for j in range(M.shape[1]):
                counter = -1
                for k in range(M.shape[2]):
                    if M[i][j][k] != 1:
                        break
                    counter += 1
                assert(counter >= 0)
                adjusted_vector.append(logits[i][j][counter])

        adjusted_vector = torch.stack(adjusted_vector, dim = 0).view(M.shape[0], M.shape[1])

        # log_probs_flat: (batch * max_len, num_classes)
        log_probs_flat = functional.log_softmax(adjusted_vector)
        # target_flat: (batch * max_len, 1)
        target_flat = target.view(-1, 1)
        # losses_flat: (batch * max_len, 1)
        losses_flat = -torch.gather(log_probs_flat, dim=1, index=target_flat)
        # losses: (batch, max_len)
        try:
            losses = losses_flat.view(*mask.size())
            # mask: (batch, max_len)
            losses = losses * mask.float()
            loss = losses.sum() / mask.float().sum()
        except:
            loss = losses_flat.sum() / losses_flat.size(0)
        return loss

class SampledSoftmaxWrapper(torch.nn.Module):
    def __init__(self, sampled_softmax, loss):
        super(SampledSoftmaxWrapper, self).__init__()
        self.sampled_softmax = sampled_softmax
        self.loss = loss

    def forward(self, logits, target, mask, dummy = True):
        # I am not so sure about this
        target = target.cuda()
        mask = mask.cuda()
        '''if dummy:
            logits, new_targets = self.sampled_softmax(logits.contiguous().view(-1, logits.size(-1)), target.view(-1))
            return self.loss(logits, new_targets, mask)
        else:'''
        logits, new_targets = self.sampled_softmax(logits.contiguous().view(-1, logits.size(-1)), target.view(-1))
        return self.loss(logits, new_targets, mask)

class AdaptiveSoftmaxWrapper(torch.nn.Module):
    def __init__(self, sampled_softmax_forward, sampled_softmax_backward, loss):
        super(AdaptiveSoftmaxWrapper, self).__init__()
        self.sampled_softmax_forward = sampled_softmax_forward
        self.sampled_softmax_backward = sampled_softmax_backward
        self.loss = loss

    def forward(self, logits, target, mask, dummy = True):
        # I am not so sure about this
        target = target.cuda()
        mask = mask.cuda()
        if dummy:
            logits = self.sampled_softmax_forward(logits.contiguous().view(-1, logits.size(-1)), target.view(-1))
            return self.loss(logits, target.view(-1))
        else:
            logits= self.sampled_softmax_backward(logits.contiguous().view(-1, logits.size(-1)), target.view(-1))
            return self.loss(logits, target.view(-1))

class AdaptiveSoftmaxAccWrapper(torch.nn.Module):
    def __init__(self, sampled_softmax_forward, sampled_softmax_backward, loss, accuracy_slack_size):
        super(AdaptiveSoftmaxAccWrapper, self).__init__()
        self.sampled_softmax_forward = sampled_softmax_forward
        self.sampled_softmax_backward = sampled_softmax_backward
        self.loss = loss
        self.accuracy_slack_size = accuracy_slack_size

    def forward(self, logits, target, mask, indicator = True):
        target = target.cuda()
        mask = mask.cuda()
        if indicator:
            logits = self.sampled_softmax_forward.log_prob(logits.contiguous().view(-1, logits.size(-1)))
        else:
            logits= self.sampled_softmax_backward.log_prob(logits.contiguous().view(-1, logits.size(-1)))

        distance, words = torch.topk(logits, k = self.accuracy_slack_size, dim = 1)
        counter = 0.0
        all_counter = 0.0
        target = target.contiguous().view(-1).cpu().numpy()
        mask = mask.contiguous().view(-1).cpu().numpy()
        for index, i in enumerate(words):
            if mask[index] == 1:
                all_counter += 1.0
                if target[index] in i:
                    counter += 1.0
        return torch.Tensor([counter / all_counter])

    def return_probs(self, logits, target, mask, indicator = True):
        target = target.cuda()
        mask = mask.cuda()
        if indicator:
            logits = self.sampled_softmax_forward.log_prob(logits.contiguous().view(-1, logits.size(-1)))
        else:
            logits= self.sampled_softmax_backward.log_prob(logits.contiguous().view(-1, logits.size(-1)))

        distance, words = torch.topk(logits, k = self.accuracy_slack_size, dim = 1)

        self.words = words

        counter = 0.0
        all_counter = 0.0
        target = target.contiguous().view(-1).cpu().numpy()
        mask = mask.contiguous().view(-1).cpu().numpy()
        for index, i in enumerate(words):
            if mask[index] == 1:
                all_counter += 1.0
                if target[index] in i:
                    counter += 1.0
        return torch.Tensor([counter / all_counter])

class MSEWrapper(torch.nn.Module):
    def __init__(self, word_vectors, reduce_loss = False):
        super(MSEWrapper, self).__init__()
        self.loss = torch.nn.MSELoss(reduce=False)
        self.word_vectors = torch.nn.Parameter(word_vectors, requires_grad = False)
        self.reduce_loss = reduce_loss
        print("Use contiguous output: MSE")
    def forward(self, logits, target, mask, dummy = True):
            
        selected_word_embeddings = torch.index_select(self.word_vectors, 0, target.view(-1)).cuda()
        return_loss = self.loss(logits.contiguous().view(-1, selected_word_embeddings.size(1)), selected_word_embeddings)
        if self.reduce_loss:
            mask = mask.contiguous().view(-1).unsqueeze(1).expand_as(return_loss).float()
            return_loss = return_loss * mask
            return return_loss.sum() / mask.sum()
        else:
            return_loss = return_loss * mask.contiguous().view(-1).unsqueeze(1).expand_as(return_loss).float()
            return return_loss.sum() / mask.float().sum()

class CosineWrapper(torch.nn.Module):
    def __init__(self, word_vectors, ignore_index = None):
        super(CosineWrapper, self).__init__()
        self.loss = torch.nn.CosineSimilarity(dim = 1)
        self.word_vectors = word_vectors # This is very important
        self.ignore_index = ignore_index
        
    def forward(self, logits, target, mask = None, dummy = True):
        assert(self.word_vectors.requires_grad == False)
        target = target.cpu()

        if self.ignore_index:
            induced_mask = target == self.ignore_index
            target[induced_mask] = 0
            mask = induced_mask != 0
        device_id = logits.data.get_device()
        selected_word_embeddings = torch.index_select(self.word_vectors, 0, target.view(-1)).cuda(device_id)
        return_loss = -self.loss(logits.contiguous().view(-1, selected_word_embeddings.size(1)), selected_word_embeddings)
        mask = mask.contiguous().view(-1).float()
        return_loss = return_loss * mask
        #if inspect_loss:
        #    print([str(index) + " " + str(i)  for index, i in enumerate(return_loss.cpu().numpy().tolist())])
        return return_loss.sum() / mask.sum()

    def square(self, input):
        return torch.norm(input, p = 2, dim = 1)

class ModifiedCosineWrapper(torch.nn.Module):
    def __init__(self, word_vectors):
        super(ModifiedCosineWrapper, self).__init__()
        self.word_vectors = word_vectors # This is very important
        print("Use contiguous output: Cosine Loss")
    def forward(self, logits, target, mask, dummy = True):
        assert(self.word_vectors.requires_grad == False)
        selected_word_embeddings = torch.index_select(self.word_vectors, 0, target.view(-1)).cuda()
        logits = logits.view(-1, logits.size(-1))
        logits = logits / torch.norm(logits, p = 2, dim = 1).unsqueeze(-1).expand_as(logits)

        return_loss = -torch.bmm(logits.unsqueeze(1), selected_word_embeddings.unsqueeze(-1)).squeeze(-1).squeeze(-1)
        mask = mask.contiguous().view(-1).float()
        return_loss = return_loss * mask

        return return_loss.sum() / mask.sum()

    def square(self, input):
        return torch.norm(input, p = 2, dim = 1)


class ModifiedCosineL2RegWrapper(torch.nn.Module):
    def __init__(self, word_vectors, lamda_one, lamda_two):
        super(ModifiedCosineL2RegWrapper, self).__init__()
        self.word_vectors = word_vectors
        self.lamda_one = lamda_one
        self.lamda_two = lamda_two
        self.cosine_loss = torch.nn.CosineSimilarity(dim = 1)

        self.scale_record = 0.0
        self._timer = 0

    def forward(self, logits, target, mask, dummy = True):
        assert(self.word_vectors.requires_grad == False)
        selected_word_embeddings = torch.index_select(self.word_vectors, 0, target.view(-1)).cuda()
        logits = logits.contiguous().view(-1, selected_word_embeddings.size(1))
        all_loss, scale = self.get_vMF_loss(logits, selected_word_embeddings)

        mask = mask.contiguous().view(-1).float()
        self._timer += 1
        with torch.no_grad():
            if 0 <= self._timer % check_cosine_interval < monitor_interval: 
               logits = logits / torch.norm(logits, p = 2, dim = 1).unsqueeze(-1).expand_as(logits)
               self.scale_record += ((torch.bmm(logits.unsqueeze(1), selected_word_embeddings.unsqueeze(-1)).squeeze(-1).squeeze(-1) * mask).sum() / mask.sum()).detach()

        if self._timer % check_cosine_interval == monitor_interval - 1:
            print("### Training Cosine Loss: {} {}".format(self._timer, self.scale_record.cpu().item() / monitor_interval))
            self.scale_record = 0

        all_loss = all_loss * mask
        return all_loss.sum() / mask.sum()

    def get_vMF_loss(self, input_vector, target_vector):
        loc = input_vector
        scale = self.square(input_vector)

        __m = loc.size(-1)
        normalization_term = (__m / 2 - 1) * torch.log(scale) - (__m / 2) * math.log(2 * math.pi) - ( log_iv_approximate_gradient_return_constant(__m / 2 - 1, scale).float())
        _log_unnormalized_prob = (loc * target_vector).sum(-1)

        returned_loss =  -self.lamda_one * _log_unnormalized_prob - normalization_term + self.lamda_two * scale
        
        target_scale = self.square(target_vector)
        print(target_scale[:1000])
        print(target_scale.mean(-1))

        return returned_loss, scale
 
    def square(self, input):
        return torch.norm(input, p = 2, dim = 1)

class AccuracyEmbeddingWrapper(torch.nn.Module):
    def __init__(self, word_vectors, metric, accuracy_slack_size, loss = None):
        super(AccuracyEmbeddingWrapper, self).__init__()
        self.nearest_neighbors = NearestNeighbors(
            n_neighbors = accuracy_slack_size, 
            n_jobs = 1, 
            metric = metric).fit(word_vectors.cpu().numpy())
        self.word_vectors = word_vectors
        self.loss = loss
        self.accuracy_slack_size = accuracy_slack_size

    def forward(self, logits, target, mask, dummy = True):
        #assert(self.word_vectors.size(1) == 300)
        distance, words = self.nearest_neighbors.kneighbors(logits.contiguous().view(-1, self.word_vectors.size(1)).cpu().numpy())
        counter = 0.0
        all_counter = 0.0
        target = target.contiguous().view(-1).cpu().numpy()
        mask = mask.contiguous().view(-1).cpu().numpy()
        for index, i in enumerate(words):
            if mask[index] == 1:
                all_counter += 1.0
                if target[index] in i:
                    counter += 1.0
        return torch.Tensor([counter / all_counter])

class AccuracyEmbeddingWrapperCuda(torch.nn.Module):
    def __init__(self, word_vectors, metric, accuracy_slack_size, loss = None):
        super(AccuracyEmbeddingWrapperCuda, self).__init__()
        self.word_vectors = word_vectors.cuda()
        self.loss = loss
        self.accuracy_slack_size = accuracy_slack_size
        self.normed_vectors = torch.norm(self.word_vectors, p = 2, dim = 1).unsqueeze(0).detach().cuda()
        self.cross_loss = nn.CrossEntropyLoss(reduce = False)

    def forward(self, logits, target, mask, dummy = True):
        #assert(self.word_vectors.size(1) == 300)

        '''logits = logits.view(-1, logits.size(-1))
        cosine_sim = torch.matmul(logits, self.word_vectors.transpose(0,1)) /  torch.matmul(torch.norm(logits, p = 2, dim = 1).unsqueeze(-1), self.normed_vectors)

        distance, words = torch.topk(cosine_sim, k = self.accuracy_slack_size, dim = 1)
        counter = 0.0
        all_counter = 0.0
        target = target.contiguous().view(-1).cpu().numpy()
        mask = mask.contiguous().view(-1).cpu().numpy()
        for index, i in enumerate(words):
            if mask[index] == 1:
                all_counter += 1.0
                if target[index] in i:
                    counter += 1.0
        return torch.Tensor([counter / all_counter])'''
        mask = mask.cuda().float().view(-1)
        logits = logits.view(-1, logits.size(-1))
        cosine_sim = torch.matmul(logits, (self.word_vectors / self.normed_vectors.transpose(0,1).expand_as(self.word_vectors)).transpose(0,1))
        loss =  self.cross_loss(cosine_sim.contiguous().view(-1, cosine_sim.size(-1)),
                    target.cuda().contiguous().view(-1)) * mask

        return loss.sum() / mask.sum()

    def return_probs(self, logits, target, mask, dummy = True):
        logits = logits.view(-1, logits.size(-1))
        cosine_sim = torch.matmul(logits, self.word_vectors.transpose(0,1)) /  torch.matmul(torch.norm(logits, p = 2, dim = 1).unsqueeze(-1), self.normed_vectors)

        distance, words = torch.topk(cosine_sim, k = self.accuracy_slack_size, dim = 1)
        counter = 0.0
        all_counter = 0.0
        target = target.contiguous().view(-1).cpu().numpy()
        mask = mask.contiguous().view(-1).cpu().numpy()

        self.saved_words = words
        self.saved_logits = logits.cpu().detach()

        for index, i in enumerate(words):
            if mask[index] == 1:
                all_counter += 1.0
                if target[index] in i:
                    counter += 1.0
        return torch.Tensor([counter / all_counter])

class AccuracyNLLWrapper(torch.nn.Module):
    def __init__(self, accuracy_slack_size):
        super(AccuracyNLLWrapper, self).__init__()
        self.accuracy_slack_size = accuracy_slack_size

    def forward(self, logits, target, mask, dummy = True):
        logits, words = logits.contiguous().view(-1, logits.size(-1)).topk(self.accuracy_slack_size, dim = 1)
        # now words = batch*len x self.accuracy_slack_size
        words = unwrap_tensor(words)
        target = unwrap_tensor(target.contiguous().view(-1))
        mask = unwrap_tensor(mask.contiguous().view(-1)) 
        counter = 0.0
        all_counter = 0.0
        for index, i in enumerate(words):
            if mask[index] == 1:
                all_counter += 1.0
                if target[index] in i:
                    counter += 1.0
        return torch.Tensor([counter / all_counter])

class AccuracyROCWrapper(torch.nn.Module):
    def __init__(self, accuracy_slack_size):
        super(AccuracyROCWrapper, self).__init__()
        self.accuracy_slack_size = accuracy_slack_size

    def forward(self, logits, target, mask, dummy = True):

        # these code are very nasty. This is because SRU does not provide support for stacked sequence.

        # Improve later

        M = mask.cpu().numpy()

        adjusted_vector = []

        length = []

        logits = logits.transpose(1,2)

        for i in range(M.shape[0]):
            for j in range(M.shape[1]):
                counter = -1
                for k in range(M.shape[2]):
                    if M[i][j][k] != 1:
                        break
                    counter += 1
                assert(counter >= 0)
                adjusted_vector.append(logits[i][j][counter])

        adjusted_vector = torch.stack(adjusted_vector, dim = 0).view(M.shape[0], M.shape[1])

        logits, words = adjusted_vector.contiguous().view(-1, adjusted_vector.size(-1)).topk(self.accuracy_slack_size, dim = 1)
        # now words = batch*len x self.accuracy_slack_size
        words = unwrap_tensor(words)
        target = unwrap_tensor(target.contiguous().view(-1))
        mask = unwrap_tensor(mask.contiguous().view(-1)) 
        counter = 0.0
        all_counter = 0.0
        for index, i in enumerate(words):
            if mask[index] == 1:
                all_counter += 1.0
                if target[index] in i:
                    counter += 1.0
        return torch.Tensor([counter / all_counter])

class vMFLossWrapper(torch.nn.Module):
    def __init__(self, word_vectors, lamda_one, lamda_two):
        super(vMFLossWrapper, self).__init__()
        self.word_vectors = word_vectors
        self.lamda_one = lamda_one
        self.lamda_two = lamda_two
        self.cosine_loss = torch.nn.CosineSimilarity(dim = 1)

        self.scale_record = 0.0
        self._timer = 0

    def forward(self, logits, target, mask, dummy = True):
        selected_word_embeddings = torch.index_select(self.word_vectors, 0, target.cpu().view(-1)).cuda()
        logits = logits.contiguous().view(-1, selected_word_embeddings.size(1))
        all_loss, scale = self.get_vMF_loss(logits, selected_word_embeddings)
        mask = mask.contiguous().view(-1).float()

        self._timer += 1
        with torch.no_grad():
            if 0 <= self._timer % check_cosine_interval < monitor_interval:
               self.scale_record += ((self.cosine_loss(logits.contiguous().view(-1, selected_word_embeddings.size(1)), selected_word_embeddings) * mask).sum() / mask.sum()).detach()

        if self._timer % check_cosine_interval == monitor_interval - 1:
            print("### Training Cosine Loss: {} {}".format(self._timer, self.scale_record.cpu().item() / monitor_interval))
            self.scale_record = 0

        all_loss = all_loss * mask
        return all_loss.sum() / mask.sum()

    def get_vMF_loss(self, input_vector, target_vector):
        loc = input_vector
        scale = self.square(input_vector)

        target_vector = target_vector / self.square(target_vector).unsqueeze(-1).expand_as(target_vector)

        __m = loc.size(-1)

        normalization_term = (__m / 2 - 1) * torch.log(scale) - (__m / 2) * math.log(2 * math.pi) - ( log_iv_approximate_gradient_return_constant(__m / 2 - 1, scale).float())

        _log_unnormalized_prob = (loc * target_vector).sum(-1)

        returned_loss =  - self.lamda_one * _log_unnormalized_prob - normalization_term + self.lamda_two * scale

        return returned_loss, scale
 
    def square(self, input):
        return torch.norm(input, p = 2, dim = 1)