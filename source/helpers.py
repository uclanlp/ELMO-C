import numpy as np
import numpy
import torch

from timeit import default_timer as timer
from allennlp.modules.elmo import batch_to_ids
import os
import string
# For profilling
try:
    from line_profiler import LineProfiler
    if not config["other_stuff"].get("profiling", False):
        assert(0)
    def do_profile(follow=[]):
        def inner(func):
            def profiled_func(*args, **kwargs):
                try:
                    profiler = LineProfiler()
                    profiler.add_function(func)
                    for f in follow:
                        profiler.add_function(f)
                    profiler.enable_by_count()
                    return func(*args, **kwargs)
                finally:
                    profiler.print_stats()
            return profiled_func
        return inner
except:
    def do_profile(follow=[]):
        "Helpful if you accidentally leave in production!"
        def inner(func):
            def nothing(*args, **kwargs):
                return func(*args, **kwargs)
            return nothing
        return inner
import json
class Vocab:
    def __init__(self):
        self.idx = []
        self.w2idx = {}

    def save(self, path):
        with open(path, "w") as f:
            json.dump([self.idx, self.w2idx], f)

    def load(self, path):
        with open(path) as f:
            [self.idx, self.w2idx] = json.load(f)
        print("Dictionary loaded. There are {:d} words.".format(len(self.idx)))

    def add_word(self, word):
        result = self.w2idx.get(word, -1)
        if result == -1:
            self.idx.append(word)
            self.w2idx[word] = len(self.idx) - 1

    def add_word_light(self, word): # only for mimick
        self.idx.append(word)

    def get_and_add_word(self, word):
        result = self.w2idx.get(word, -1)
        if result == -1:
            self.idx.append(word)
            result = len(self.idx) - 1
            self.w2idx[word] = result
        return result

    def size(self):
        return len(self.idx)

    def get(self, key, default):
        return self.w2idx.get(key, default)

    def create_frequence(self, default = 1):
        self.freq_list = []
        for i in self.idx:
            self.freq_list.append(default)

    def create_from_file(self, file_name, trunacate = None):
        with open(file_name) as f:
            lines = f.read().split("\n")
        if trunacate:
            lines = lines[:trunacate]
        for index, i in enumerate(lines):
            if index % 10000 == 0:
                print(index)
            self.add_word(i.split(" ")[0])

    def __getitem__(self, token):
        return self.w2idx[token]

from .bpe import create_bpe
class BPE:
    def __init__(self, model_path, vocab_path):
        self.vocab = Vocab()
        self.vocab.load(vocab_path)
        self.w2idx = self.vocab.w2idx
        self.idx = self.vocab.idx
        self.model = create_bpe(model_path, set(self.idx))
    def process_line(self, line):
        return self.model.process_line(line)
    def segment_tokens(self, tokens):
        return self.model.segment_tokens(tokens)

    def convert_inputs(self, inputs, word_vocab = None, special_indicator = False, ditch_unk = False):

        unk_id = len(self.idx)

        if word_vocab is None:
            # The inputs is already word list
            assert(isinstance(inputs, list))
            word_inputs = inputs
        else:
            inputs = inputs.data.cpu().numpy().tolist()
            word_inputs = [[word_vocab.idx[j] for j in i if j != 0] for i in inputs] # Strip the paddings

        #print(word_inputs)

        lengths = [len(i) for i in word_inputs]

        if not special_indicator:
            word_inputs = [self.segment_tokens(i) for i in word_inputs]

        #print(word_inputs)

        record_postions = []
        flag = 0

        for index_i, i in enumerate(word_inputs):
            tmp = []
            for index, j in enumerate(i):
                if not special_indicator:
                    if len(j) > 2 and j[-2:] == "@@":
                        if flag == 1:
                            continue
                        else:
                            tmp.append(index)
                            flag = 1
                    else:
                        if flag == 1: # We have seen "@@" before and recorded that word
                            flag = 0
                        else:
                            tmp.append(index)
                else:
                    tmp.append(index)
            assert(len(tmp) == lengths[index_i])
            record_postions.append(tmp)
        #print(word_inputs[:2])
        #print(record_postions[:2])

        #print(" ".join(word_inputs[1]))

        out = []
        for i in word_inputs:
            tmp = []
            for j in i:
                tmp_j = self.w2idx.get(j, -1)
                if tmp_j == -1:
                    print(tmp_j)
                    print(j)
                    #assert(0)
                    #if ditch_unk:
                    #    continue
                    #print(j)
                    tmp_j = unk_id
                tmp.append(tmp_j)
            out.append(tmp)
        word_inputs = out

        max_len = max([len(i) for i in word_inputs])
        word_inputs = [i + [ 0 for j in range(max_len - len(i))] for i in word_inputs]

        max_len_original = max([len(i) for i in record_postions])

        record_postions = [i + [ 0 for j in range(max_len_original - len(i))] for i in record_postions]
        record_postions = torch.LongTensor(record_postions)
        '''for i in word_inputs:
            for j in i:
                assert(j < len(self.bpe.idx))'''
        inputs = torch.LongTensor(word_inputs)# Convert it back
        return inputs, record_postions

def average_model(model, models):
    models_parameters = [dict(i.named_parameters()) for i in models]
    for name, param in model.named_parameters():
        tmp = [j[name].data for j in models_parameters]
        tmp = torch.stack(tmp, dim = 0)
        param.data.copy_(torch.sum(tmp, dim = 0) / len(tmp))


def average_model_memory_efficient(models_list, config, input_word_vectors):
    from complete_elmo import GeneralRNN
    if os.path.exists("/local/harold/data/"):
        on_nlp7 = 0
    else:
        on_nlp7 = 1

    if config["other_stuff"].get("models_path", None):
        models_path = config["other_stuff"]["models_path"]
    else:
        if on_nlp7:
            models_path = "/home/guojy/harold/main/elmo/"
        else:
            models_path = "/local/harold/"

        print("Did not specify model path. Automatic resolve: {}".format(models_path))

    number = len(models_list)
    print("\n\nLoading from {}".format(models_list[0]))
    config["other_stuff"]["continue_model_name"] = models_list[0]
    elmo_0 = GeneralRNN(config, input_word_vectors)
    load_state_dict_from_file(elmo_0, models_path + models_list[0])

    for name, param in elmo_0.named_parameters():
        param.data = param.data / number

    for i in models_list[1:]:
        print("\n\nLoading from {}".format(i))
        config["other_stuff"]["continue_model_name"] = i
        elmo_i = GeneralRNN(config, input_word_vectors)
        load_state_dict_from_file(elmo_i, models_path + i)

        elmo_i = dict(elmo_i.named_parameters())
        for name, param in elmo_0.named_parameters():
            tmp = elmo_i[name].data
            param.data = param.data + tmp / number
    return elmo_0



def load_state_dict_from_file(model, state_dict):
    try:
        model.load_state_dict(state_dict)
    except:
        print("Full loading failed!! Try partial loading from {}!!".format(state_dict))

    own_state = model.state_dict()
    try:
        state_dict = torch.load(state_dict)
    except:
        print("############### Open State File Failed!")
        return
    try:
        state_dict = state_dict["model_state"]
    except:
        print("Loading a legacy model!")
        pass

    for name, param in state_dict.items():
        if name not in own_state:
            print("Skipped:" + name)
            continue
        if isinstance(param, torch.nn.Parameter):
            # backwards compatibility for serialized parameters
            param = param.data
        try:
            own_state[name].copy_(param)
            print("Successfully loaded: "+name)
        except:
            print("Part load failed: " + name)

def parameters_count(model):
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    return params

def unfreeze_network(model):
    if isinstance(model, list):
        for i in model:
            unfreeze_network(i)
        return
    for parm in model.parameters():
        parm.requires_grad = True

def freeze_network(model):
    if isinstance(model, list):
        for i in model:
            freeze_network(i)
        return
    for parm in model.parameters():
        parm.requires_grad = False

def change_to_train(model_list):
    assert(isinstance(model_list, list))
    for i in model_list:
        i.train()

def change_to_eval(model_list):
    if isinstance(model_list, list):
        for i in model_list:
            change_to_eval(i)
        return
    model_list.eval()

def get_parameters_for_optimizer(net):
    return filter(lambda p: p.requires_grad, net.parameters())

def unwrap_tensor(tensor):
    return tensor.cpu().numpy()
def wrap_float_tensor(vector):
    return torch.Tensor(vector)


def cuda_sync():
    torch.cuda.synchronize()

def print_GPU_memory_usage(device = None, message = ""):
    print(message)
    print('Currently used GPU: ' + str( float(torch.cuda.memory_allocated(device)) / 1000000 ))
    print('Currently max used GPU: ' + str(float(torch.cuda.max_memory_allocated(device)) / 1000000))

def clear_GPU_cache():
    torch.cuda.empty_cache()

def detach_hidden(hidden):
    if isinstance(hidden, tuple) or isinstance(hidden, list):
        assert(len(hidden) == 2)
        return (hidden[0].detach(), hidden[1].detach())
    else:
        return hidden.detach()


def check_and_reset_hidden(hidden, input, pass_indicator = True):
    if not pass_indicator or hidden is None:
        return None
    if isinstance(hidden, tuple) or isinstance(hidden, list):
        assert(len(hidden) == 2)
        if hidden[0].size(1) != input.size(0):
            return None
        else:
            return hidden
    else:
        if hidden.size(1) != input.size(0):
            return None
        else:
            return hidden

def init_weights(model, dimention):
    val_range = (3.0/dimention)**0.5
    for p in model.parameters():
        if p.requires_grad == False:
            continue
        if p.dim() > 1:  # matrix
            p.data.uniform_(-val_range, val_range)
        else:
            p.data.zero_()

def build_unigram_noise(freq):
    """build the unigram noise from a list of frequency
    Parameters:
        freq: a tensor of #occurrences of the corresponding index
    Return:
        unigram_noise: a torch.Tensor with size ntokens,
        elements indicate the probability distribution
    """
    total = freq.sum()
    noise = freq / total
    assert abs(noise.sum() - 1) < 0.001
    return noise

def is_sparse(tensor):
    return tensor.is_sparse

def sparse_clip_norm(parameters, max_norm, norm_type=2):
    """Clips gradient norm of an iterable of parameters.
    The norm is computed over all gradients together, as if they were
    concatenated into a single vector. Gradients are modified in-place.
    Supports sparse gradients.
    Parameters
    ----------
    parameters : ``(Iterable[torch.Tensor])``
        An iterable of Tensors that will have gradients normalized.
    max_norm : ``float``
        The max norm of the gradients.
    norm_type : ``float``
        The type of the used p-norm. Can be ``'inf'`` for infinity norm.
    Returns
    -------
    Total norm of the parameters (viewed as a single vector).
    """
    # pylint: disable=invalid-name,protected-access
    parameters = list(filter(lambda p: p.grad is not None, parameters))
    max_norm = float(max_norm)
    norm_type = float(norm_type)
    if norm_type == float('inf'):
        total_norm = max(p.grad.data.abs().max() for p in parameters)
    else:
        total_norm = 0
        for p in parameters:
            if is_sparse(p.grad):
                # need to coalesce the repeated indices before finding norm
                grad = p.grad.data.coalesce()
                param_norm = grad._values().norm(norm_type)
            else:
                param_norm = p.grad.data.norm(norm_type)
            total_norm += param_norm ** norm_type
        total_norm = total_norm ** (1. / norm_type)
    clip_coef = max_norm / (total_norm + 1e-6)
    if clip_coef < 1:
        for p in parameters:
            if is_sparse(p.grad):
                p.grad.data._values().mul_(clip_coef)
            else:
                p.grad.data.mul_(clip_coef)
    return total_norm

def load_my_state_dict(model, state_dict):
    own_state = model.state_dict()
    for name, param in state_dict.items():
        if name not in own_state:
            print("Skipped:" + name)
            continue
        if isinstance(param, torch.nn.Parameter):
            # backwards compatibility for serialized parameters
            param = param.data
        try:
            own_state[name].copy_(param)
            print("Successfully loaded: "+name)
        except:
            print("Part load failed: " + name)


class TextScreener():
    def __init__(self, vocab):
        self.vocab = vocab
    def count_full(self, text):
        counter = 0.0
        for i in text:
            if i not in self.vocab.w2idx or i in string.punctuation:
                counter += 1
            else:
                try:
                    float(i)
                    counter += 1
                except:
                    pass
        return counter

    def count_with_vocabulary(self, text):
        counter = 0.0
        for i in text:
                if i in string.punctuation:
                    counter += 1
        return counter


