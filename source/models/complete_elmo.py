import json
import logging
from typing import Union, List, Dict, Any
import warnings
import torch
from torch.nn.modules import Dropout
from torch import nn
from timeit import default_timer as timer
import numpy

from allennlp.modules.scalar_mix import ScalarMix

from .elmo_own import _ElmoCharacterEncoder, SRUWrapper

from .legacy.transformer import Transformer

from ..helpers import *
import sys

import copy
# SoftMaxes...
from .losses import CosineWrapper, vMFLossWrapper, AdaptiveSoftmaxWrapper, SampledSoftmaxWrapper, StandardCrossEntrophy

from elmo_c.source.models.legacy.adasoft import AdaptiveLoss, AdaptiveSoftmax

class InitializeWrapper(torch.nn.Module):
    # Wraps the following compenents: embedding, rnn, loss/last_layer.
    def __init__(self, config, input_word_vectors = None, output_word_vectors = None):
        super(InitializeWrapper, self).__init__()
        self.config = config
        ##################### INPUT LAYER
        input_layer_config = self.config["input_layer"]
        if "cnn" in input_layer_config["name"]:  # Support both "cnn" and "acc_cnn"
            self.embedding = _ElmoCharacterEncoder(input_layer_config["options"], input_layer_config["weight_file"], requires_grad = not input_layer_config["freeze"], dropout = input_layer_config["dropout"])
            # right now CNN load its weight in its __init__ function, because the CNN need the load_weight fucntion to construct some network structures
        elif input_layer_config["name"] == "embedding":
            if input_layer_config["freeze"]:
                try: # This is to be compatible with the PyTorch 0.3 and PyTorch 0.4
                    self.embedding = nn.Embedding.from_pretrained(input_word_vectors, freeze = True)
                except:
                    self.embedding = nn.Embedding(input_word_vectors.size(0), input_word_vectors.size(1))
                    self.embedding.weight = torch.nn.Parameter(input_word_vectors, requires_grad = False)
            else:
                self.embedding = nn.Embedding(config["output_layer"]["vocab_size"], embedding_dim = input_layer_config["input_size"])
                print("###### Creating Non Freezed Input Embedding of size {}".format(config["output_layer"]["vocab_size"]))
        elif input_layer_config["name"] == "none":
            self.embedding = None

        ##################### RNN
        rnn_layer_config = self.config["rnn_layer"]
        #try:
        # Just to be compatible with SRL evaluation.

        if torch.__version__[:3] == "0.3":
            from .old_allennlp.elmo_lstm_old import ElmoLstm, ElmoLstmUni
        else:
            from torch.nn import LayerNorm
            if rnn_layer_config.get("custom_elmo", False):
                from .custom_elmo_lstm import ElmoLstmUni
            else:
                from .elmo_lstm import ElmoLstm, ElmoLstmUni

        if rnn_layer_config["name"] == "sru":
            rnn = SRUWrapper(
                input_size = rnn_layer_config["input_size"],
                projection_size = rnn_layer_config["projection_size"], 
                hidden_size = rnn_layer_config["hidden_size"],
                input_proj = rnn_layer_config["input_proj"],
                num_layers = rnn_layer_config["num_layers"],
                dropout = rnn_layer_config["dropout"],           # dropout applied between RNN layers
                rnn_dropout = rnn_layer_config["rnn_dropout"],       # variational dropout applied on linear transformation
                is_input_normalized = rnn_layer_config.get("is_input_normalized", False),
                use_tanh = True,     # use tanh?
                use_relu = 0,            # use ReLU?
                use_selu = 0,            # use SeLU?
                bidirectional = False,   # Always False
                weight_norm = False,     # apply weight normalization on parameters
                layer_norm = rnn_layer_config["layer_norm"],      # apply layer normalization on the output of each layer
                highway_bias = rnn_layer_config["highway_bias"],         # initial bias of highway gate (<= 0)
                batch_first = True,
                common_crawl_style = rnn_layer_config["common_crawl_style"],
                use_new_sru = rnn_layer_config["use_new_sru"],
                rescale = rnn_layer_config["rescale"],
                reset_hidden_every_time = rnn_layer_config["reset_hidden_every_time"],
                v1 = rnn_layer_config["v1"],
                clip_value = rnn_layer_config.get("clip_value", -1),
                use_lstm = rnn_layer_config.get("use_lstm", False)
                )
        elif rnn_layer_config["name"] == "qrnn":
            rnn = QRNNWrapper(
                    input_size = rnn_layer_config["input_size"],
                    projection_size = rnn_layer_config["projection_size"], 
                    hidden_size = rnn_layer_config["hidden_size"],
                    input_proj = rnn_layer_config["input_proj"],
                    num_layers = rnn_layer_config["num_layers"],
                    dropout = rnn_layer_config["dropout"],           # dropout applied between RNN layers
                    layer_norm = rnn_layer_config["layer_norm"],      # apply layer normalization on the output of each layer
                    batch_first = True,
                    common_crawl_style = rnn_layer_config["common_crawl_style"],
                    reset_hidden_every_time = rnn_layer_config["reset_hidden_every_time"],
                    clip_value = rnn_layer_config.get("clip_value", -1),
                    save_prev_x=rnn_layer_config["save_prev_x"], 
                    zoneout=rnn_layer_config["zoneout"], 
                    window=rnn_layer_config["window"]
                    )
        elif rnn_layer_config["name"] == "transformer":
            rnn = Transformer(
                   input_size = rnn_layer_config["input_size"],
                   N=rnn_layer_config["num_layers"],
                   d_model=rnn_layer_config["projection_size"],
                   d_ff=rnn_layer_config["hidden_size"],
                   h=rnn_layer_config["h"],
                   dropout=rnn_layer_config["rnn_dropout"],
                   reverse = False)
        elif rnn_layer_config['name'] == "elmo":
            rnn = ElmoLstmUni(
                input_size = rnn_layer_config["input_size"],
                hidden_size = rnn_layer_config["projection_size"],
                cell_size = rnn_layer_config["hidden_size"],
                num_layers = rnn_layer_config["num_layers"],
                memory_cell_clip_value = 3,
                state_projection_clip_value = 3, # fixed now, change later
                requires_grad=True,
                recurrent_dropout_probability = rnn_layer_config["dropout"],
                layer_norm = rnn_layer_config["layer_norm"],
                ln_before_act = rnn_layer_config["ln_before_act"],
                add_embedding_layer = rnn_layer_config["add_embedding_layer"],
                correct_layer_norm = rnn_layer_config.get("correct_layer_norm", False),
                reset_hidden_state = rnn_layer_config.get("reset_hidden_state", False),
                output_pre_norm = rnn_layer_config.get("output_pre_norm", False))
        elif rnn_layer_config["name"] == "original_elmo":
            rnn = ElmoLstm(input_size=rnn_layer_config["input_size"],
                           hidden_size=rnn_layer_config["projection_size"],
                           cell_size=rnn_layer_config["hidden_size"],
                           num_layers=rnn_layer_config["num_layers"],
                           memory_cell_clip_value=3,
                           state_projection_clip_value=3,
                           requires_grad=False,
                           reset_hidden_state = rnn_layer_config.get("reset_hidden_state", False))
            print("############## Loading ELMo LSTM weights...")
            rnn.load_weights(rnn_layer_config["weight_file"])
            print("############## Loaded!")
        elif rnn_layer_config["name"] == "none":
            rnn = None
        elif rnn_layer_config["name"] == "bert":
            rnn = BertModelWithCalculatedWordEmbedding(bert_config)
        else:
            assert(0) # Unsupported RNN type

        if rnn_layer_config["bidirectional"]:
            rnn_backward = copy.deepcopy(rnn)
            rnn = BiDirectionalRNN(rnn_fowrad = rnn, rnn_backward = rnn_backward)

        self.rnn = rnn

        ##################### LAST LAYER
        last_layer_config = config["output_layer"]
        last_layer_insize = last_layer_config["input_size"]
        last_layer_outsize = last_layer_config.get("output_size", -1)
        _last_layer_backward = None # a place holder

        if last_layer_config["name"] == "semfit":
            if rnn_layer_config["bidirectional"]:
                additional_linear_layer = nn.Linear(last_layer_insize * 2, last_layer_outsize * 2)
            else:
                additional_linear_layer = nn.Linear(last_layer_insize, last_layer_outsize)
        else:
            additional_linear_layer = None
        self.additional_linear_layer = additional_linear_layer
        
        if last_layer_config["name"] == "adaptive_softmax":
            if config['dataset'].get("hard_restrict_vocab", False):
                last_layer_config["vocab_size"] = config['dataset'].get("hard_restrict_vocab", False) + 10
                # We only use restricted vocabulary

            while last_layer_config["cutoff"][-1] > last_layer_config["vocab_size"]:
                last_layer_config["cutoff"] = last_layer_config["cutoff"][:-1]

            new_cutoff = last_layer_config["cutoff"]

            if last_layer_config["cutoff"][-1] < last_layer_config["vocab_size"]:
                new_cutoff = new_cutoff + [last_layer_config["vocab_size"]]

            if rnn_layer_config.get("projection_size", 0):
                n_proj = rnn_layer_config.get("projection_size", 0)
            else:
                n_proj = rnn_layer_config["hidden_size"]

            _last_layer_forward = AdaptiveSoftmax(
                last_layer_insize, 
                new_cutoff,
                n_proj = n_proj,
                continue_softmax_from_emb = config["other_stuff"].get("continue_softmax_from_emb", False)
                )
            _last_layer_backward = _last_layer_forward
        elif last_layer_config["name"] == "softmax":
            _last_layer_forward = nn.Linear(last_layer_insize, last_layer_config["vocab_size"])
            # very tricky
            if last_layer_config.get("tie_input_output_embedding", False):
                _last_layer_forward.weight = self.embedding.weight
            _last_layer_backward = _last_layer_forward
        else:
            _last_layer_forward = None
            _last_layer_backward = None
        if not rnn_layer_config["bidirectional"]:
            _last_layer_backward = None

        if config["loss"]["name"] == "mse":
            loss_cri = MSEWrapper(output_word_vectors, config["loss"]["reduce_loss"])
        elif config["loss"]["name"] == "cosine":
            loss_cri = CosineWrapper(output_word_vectors)
        elif config["loss"]["name"] == "modified_cosine":
            loss_cri = ModifiedCosineWrapper(output_word_vectors)
        elif config["loss"]["name"] == "modified_cosine_l2":
            loss_cri = ModifiedCosineL2RegWrapper(output_word_vectors, 
                lamda_one = config["loss"]["lamda_one"], 
                lamda_two = config["loss"]["lamda_two"])
        elif config["loss"]["name"] == "vmf":
            loss_cri = vMFLossWrapper(
                output_word_vectors, 
                lamda_one = config["loss"]["lamda_one"], 
                lamda_two = config["loss"]["lamda_two"])
        elif config["loss"]["name"] == "nll":
            loss_cri = StandardCrossEntrophy()
        elif config["loss"]["name"] == "sampled_softmax":
            loss_cri = StandardCrossEntrophy()
            from .sampled_softmax import SampledSoftmax

            if rnn_layer_config.get("projection_size", 0):
                n_proj = rnn_layer_config.get("projection_size", 0)
            else:
                n_proj = rnn_layer_config["hidden_size"]
            loss_cri = SampledSoftmaxWrapper(SampledSoftmax(ntokens = last_layer_config["vocab_size"], nsampled = last_layer_config["nsampled"], nhid = last_layer_insize, tied_weight = None, n_proj = n_proj), loss_cri)
        elif config["loss"]["name"] == "adaptive_softmax":
            last_layer_config = config['output_layer']
            loss_cri = AdaptiveLoss(last_layer_config["cutoff"] + [last_layer_config["vocab_size"]])
            loss_cri = AdaptiveSoftmaxWrapper(_last_layer_forward, _last_layer_backward, loss_cri)
        elif config["loss"]["name"] == "acc_embedding":
            loss_cri = AccuracyEmbeddingWrapperCuda(output_word_vectors, metric = "cosine",accuracy_slack_size = 10)
        elif config["loss"]["name"] == "acc_adaptive":
            last_layer_config = config['output_layer']
            loss_cri = AdaptiveLoss(last_layer_config["cutoff"] + [last_layer_config["vocab_size"]])
            loss_cri = AdaptiveSoftmaxAccWrapper(elmo.rnn_last_layer_wrapper._last_layer_forward, elmo.rnn_last_layer_wrapper._last_layer_backward, loss_cri, accuracy_slack_size = 10)
        elif config["loss"]["name"] == "nce":
            global_vocab.create_frequence()
            tmp_dict = {}
            with open(data_folder + config["dataset"]["unigram_file"]) as f:
                for line in f:
                    if len(line) > 2:
                        words = line.split(" ")
                        index = global_vocab.w2idx.get(words[0], -1)
                        if index != -1:
                            global_vocab.freq_list[index]= int(words[1])
            noise = build_unigram_noise(torch.Tensor(global_vocab.freq_list))
            loss_cri = NCEWrapper(word_vector = output_word_vectors, 
                         noise = noise.pow(config['loss']["sample_pow"]), 
                         noise_ratio=config['loss']["noise_ratio"],
                         norm_term=config['loss']["norm_term"],
                         loss_type=config['loss']["loss_type"])
        else:
            assert(0) # Unsupported Loss Type

        self.loss = loss_cri

    def save_full_training_state(self, epoch, train_iter, optimizer_state, optimizer_step, save_path):
        save_dict = {
            "model_state": self.state_dict(), 
            "epoch": epoch,
            "train_iter": train_iter, 
            "optimizer_state" : optimizer_state, 
            "optimizer_step": optimizer_step}
        torch.save(save_dict, save_path)

    def load_state_dict_from_file(self, models_path, config, weight = None):
        if config["other_stuff"].get("load_legacy_model", False):
            from elmo_c.source.models.legacy.legacy_complete_elmo import GeneralRNN
            temporary_model = GeneralRNN(config, weight)
            temporary_model.load_state_dict_from_file(models_path) # Directly call elmo!
            self.embedding = temporary_model.embedding
            if config["other_stuff"]["parallel_rnn_and_last_layer"]:
                self.rnn = temporary_model.rnn_last_layer_wrapper.module.rnn
            else:
                self.rnn = temporary_model.rnn_last_layer_wrapper.rnn
        else:     
            if not self.config["other_stuff"].get("secondary_model_name", False):
                load_state_dict_from_file(self, models_path + self.config["other_stuff"]["model_name"])
            else:
                print("Merging Models...")
                assert(not self.config["rnn_layer"]["bidirectional"])
                forward_model = copy.deepcopy(self)
                backward_model = self
                load_state_dict_from_file(forward_model, models_path + self.config["other_stuff"]["model_name"])
                load_state_dict_from_file(backward_model, models_path + self.config["other_stuff"]["secondary_model_name"])
                self.config["rnn_layer"]["bidirectional"] = True
                self.rnn = BiDirectionalRNN(rnn_fowrad = forward_model.rnn, rnn_backward = backward_model.rnn)

    def replace_embedding(self, word_vectors):
        parameter_wrap = torch.nn.Parameter(word_vectors, requires_grad = False)
        self.embedding.weight = parameter_wrap
        self.loss.word_vectors = parameter_wrap

class TrueRunningModel(torch.nn.Module):
    def __init__(self, config, input_layer, rnn, loss, additional_linear_layer, device = None):
        super(TrueRunningModel, self).__init__()
        # Doing three wrapper layers
        self.config = config
        input_layer = FlexibleInputLayer(config = config, embedding = input_layer)
        rnn = FlexibleRNNLayer(config = config, rnn = rnn, additional_linear_layer = additional_linear_layer)
        loss = FlexibleLoss(config = config, loss = loss)

        self.situation = config["other_stuff"]["situation"]
        if self.situation == 1:
            # Situation 1: Input layer is on CPU and not paralleled. RNN should be paralleled.
            # Loss should not be paralleled.
            rnn = rnn.cuda()
            self.model = CustomSequential(input_layer, nn.DataParallel(rnn), loss)
        elif self.situation == 2:
            # Situation 2: Input layer is fixed and not paralleled. RNN and the loss should be paralleled. (Adaptive SoftMax)
            rnn = rnn.cuda()
            loss = loss.cuda()
            self.model = CustomSequential(input_layer, nn.DataParallel(CustomSequential(rnn, loss)))

        elif self.situation == 3:
            # Situation 3: For subword models
            self.model = nn.DataParallel(CustomSequential(input_layer, rnn, loss))
            self.model = self.model.cuda()

        elif self.situation == 4:
            # Situation 4: During evaluation, we just pass it through ...
            rnn = rnn.cuda()
            loss = loss.cuda()
            self.model = CustomSequential(input_layer, rnn)
        elif self.situation == 5:
            # For CNN
            input_layer = input_layer.cuda()
            rnn = rnn.cuda()
            self.model = CustomSequential(input_layer, rnn)

        elif self.situation == 6:
            # Situation 6: This is for fixed weight sampled softmax speed test
            rnn = rnn.cuda()
            #loss = loss.cuda()
            try:
                loss.loss.sampled_softmax.projection = loss.loss.sampled_softmax.projection.cuda()
            except:
                pass
            self.model = CustomSequential(input_layer, nn.DataParallel(CustomSequential(rnn, loss)))

        self.bidirectional = config["rnn_layer"]["bidirectional"]
        self.reverse = config["rnn_layer"]["reverse"]

    def forward(self, input, forward_target, mask_forward, backward_target, mask_backward, hidden, return_hidden):
        # Wrapup mask
        if self.bidirectional:
            if self.config["other_stuff"].get("output_representations", False):
                mask = mask_forward
                assert(mask_backward is None)
            else:
                mask = (mask_forward, mask_backward)
        elif self.reverse:
            mask = mask_backward
        else:
            mask = mask_forward

        if not self.bidirectional: # Input contains both forward and backward sequence.
            if self.reverse: # Reverse the input/target/mask
                input = input[1]
                forward_target = backward_target
                mask_forward = mask_backward
            else:
                input = input[0]

        # Move necessary things to the GPU
        if self.situation == 1 or self.situation == 4:
            # Move only mask to the GPU
            mask_forward = mask_forward.cuda()
            try:
                mask_backward = mask_backward.cuda()
            except:
                pass

        return self.model(input = input, mask = mask, hidden = hidden, return_hidden = return_hidden, forward_target = forward_target, mask_forward = mask_forward, backward_target = backward_target, mask_backward = mask_backward)

class FlexibleInputLayer(torch.nn.Module):
    def __init__(self, config, embedding):
        super(FlexibleInputLayer, self).__init__()
        self.embedding = embedding
        self.config = config
        self.dropout_after_input = nn.Dropout(config["input_layer"]["dropout"])

    def forward(self, input, **kwrgs):
        if self.config["input_layer"]["name"] == "cnn":
            if isinstance(input, tuple):
                original_shape = input[0].size()
                if len(original_shape) > 3:
                    timesteps, num_characters = original_shape[-2:]
                    reshaped_inputs = input[0].view(-1, timesteps, num_characters)
                else:
                    reshaped_inputs = input[0]
                token_embedding = self.embedding(reshaped_inputs)
                embedded_0 = token_embedding['token_embedding']
                if self.config["rnn_layer"]["bidirectional"]:
                    original_shape = input[1].size()
                    if len(original_shape) > 3:
                        timesteps, num_characters = original_shape[-2:]
                        reshaped_inputs = input[1].view(-1, timesteps, num_characters)
                    else:
                        #assert(0)
                        reshaped_inputs = input[1]
                    token_embedding = self.embedding(reshaped_inputs)
                    embedded_1 = token_embedding['token_embedding']
                    embedded = (embedded_0, embedded_1)
                else:
                    embedded = embedded_0             
            else:
                original_shape = input.size()
                if len(original_shape) > 3:
                    timesteps, num_characters = original_shape[-2:]
                    reshaped_inputs = input.view(-1, timesteps, num_characters)
                else:
                    #assert(0)
                    reshaped_inputs = input
                token_embedding = self.embedding(reshaped_inputs)
                embedded = token_embedding['token_embedding']
                embedded = self.dropout_after_input(embedded)
        else:
            if isinstance(input, tuple):
                assert(len(input) == 2)
                embedded_forward = self.embedding(input[0])
                embedded_backward = self.embedding(input[1])
                embedded_forward =  self.dropout_after_input(embedded_forward)
                embedded_backward = self.dropout_after_input(embedded_backward)
                if self.config["rnn_layer"]["bidirectional"]:
                    embedded = (embedded_forward, embedded_backward)
                else:
                    embedded = embedded_forward
            else:
                embedded = self.dropout_after_input(self.embedding(input))

        if isinstance(embedded, tuple) or isinstance(embedded, list):
            kwrgs["embedded"] = (embedded[0].cuda(), embedded[1].cuda())
        else:
            kwrgs["embedded"] = embedded.cuda()
        return kwrgs

    def replace_embedding(self, replace):
        # We could have a unified fucntion to replace the embedding
        assert(0)

class FlexibleRNNLayer(torch.nn.Module):
    # This also takes care of the additional Linear Layer
    def __init__(self, config, rnn, additional_linear_layer = None):
        super(FlexibleRNNLayer, self).__init__()
        self.rnn = rnn
        self.config = config
        self.additional_linear_layer = additional_linear_layer
    def forward(self, embedded, mask, hidden = None, return_hidden = True, **kwrgs):
        if self.config["rnn_layer"]["name"] == "none":
            if isinstance(embedded, tuple):
                embedded = torch.cat(embedded, dim = -1)
            outputs = embedded
        else:
            if return_hidden:
                outputs, hidden = self.rnn(embedded, mask = mask, hidden = hidden, return_hidden = return_hidden)
            else:
                outputs = self.rnn(embedded, mask = mask)
            ##################### Dropout after RNN
            if not isinstance(outputs, list) and self.config["output_layer"]["dropout"] != 0:
                outputs = self.dropout_after_rnn_instance(outputs)

        backward_output = None
        output_representations = self.config["other_stuff"].get("output_representations", False)

        if output_representations:
            #kwrgs["forward_output"] = forward_output
            return outputs
        if self.additional_linear_layer:
            outputs = self.additional_linear_layer(outputs)

        if self.config["rnn_layer"]["bidirectional"]:
            forward_output = outputs[:, :, :int(outputs.size(-1) / 2)]
        else:
            forward_output = outputs
        if self.config["rnn_layer"]["bidirectional"]:
            backward_output = outputs[:, :, int(outputs.size(-1) / 2):]
        else:
            backward_output = None
    
        kwrgs["forward_output"] = forward_output
        kwrgs["backward_output"] = backward_output
        kwrgs["hidden"] = hidden
        return kwrgs

class FlexibleLoss(torch.nn.Module):
    def __init__(self, config, loss):
        super(FlexibleLoss, self).__init__()
        self.loss = loss
        self.config = config

    def forward(self, forward_output, backward_output, forward_target, backward_target, mask_forward, mask_backward, **kwrgs):
        forward_loss = self.loss(logits = forward_output, target = forward_target, mask = mask_forward).unsqueeze(0)

        if self.config["rnn_layer"]["bidirectional"]:
            backward_loss = self.loss(logits = backward_output, target = backward_target, mask = mask_backward).unsqueeze(0)
        else:
            backward_loss = None

        kwrgs["forward_loss"] = forward_loss
        kwrgs["backward_loss"] = backward_loss
        return kwrgs

class CustomSequential(torch.nn.Sequential):
    def __init__(self, *args):
        super(CustomSequential, self).__init__(*args)
    
    def forward(self, **input):
        for module in self._modules.values():
            input = module(**input)
        return input

class BiDirectionalRNN(nn.Module):
    '''
    Not to confuse with bi-directional RNN, this is just two RNN
    An abstract class, which is used to provide a clean inter-face for dealing with bi-directional RNN
    Let's hope PyTorch's Asynchronous Excution can provide reasonable speed
    '''
    def __init__(self, rnn_fowrad, rnn_backward):
        super(BiDirectionalRNN, self).__init__()
        self.rnn_fowrad = rnn_fowrad
        self.rnn_backward = rnn_backward

    def forward(self,
                embedded,
                mask = None,
                hidden = None,
                return_hidden = False):
        assert(isinstance(embedded, tuple))
        if not isinstance(mask, tuple):
            mask = (mask, mask) # In transfomer or my_elmo_token_embedder, there is only one mask

        if not return_hidden:
            outputs_forward = self.rnn_fowrad(embedded[0], mask = mask[0]) 
            outputs_backward = self.rnn_backward(embedded[1], mask = mask[1])
        else:
            assert(len(hidden) == 2)
            hidden_forward = hidden[0]
            hidden_backward = hidden[1]
            outputs_forward, hidden_forward = self.rnn_fowrad(embedded[0], mask = mask[0], hidden = hidden_forward, return_hidden = return_hidden) 
            outputs_backward, hidden_backward = self.rnn_backward(embedded[1], mask = mask[1], hidden = hidden_backward, return_hidden = return_hidden)
            hidden = (hidden_forward, hidden_backward)


        outputs = torch.cat((outputs_forward, outputs_backward), dim = -1) # cat along the last "hidden_size" dimention
        if not return_hidden:
            return outputs
        else:
            return outputs, hidden

    def get_all_layer_output(self, input, mask, no_carry = False):
        outputs_forward = self.rnn_fowrad.get_all_layer_output(input[0], mask = mask, no_carry = no_carry)
        outputs_backward = self.rnn_backward.get_all_layer_output(input[1], mask = mask, no_carry = no_carry)
        outputs = torch.cat((outputs_forward, outputs_backward), dim = -1) # cat along the last "hidden_size" dimention
        assert(outputs.dim() == 4)
        return outputs

class WeightedSumWrapper(torch.nn.Module): 
    def __init__(self, 
        rnn, 
        config, 
        num_output_representations = 1,
        ditch_boundry = False, 
        only_use_embedding = False, 
        include_embedding = False,
        dropout = 0.0,
        do_layer_norm = False,
        plain_representations = False):

        super(WeightedSumWrapper, self).__init__()

        layer = config["rnn_layer"]["num_layers"] 
        rnn_layer_type = config["rnn_layer"]["name"]
        self.config = config
        self.rnn = rnn
        self.rnn_layer_type = rnn_layer_type
        self.include_embedding = include_embedding
        self.only_use_embedding = only_use_embedding # This is to verify that only using FastText won't give that much good results
        self.layer = layer
        self.num_output_representations = num_output_representations
        self._scalar_mixes: Any = []

        if rnn_layer_type == "transformer" or rnn_layer_type == "elmo":
            layer += 1

        if self.config["other_stuff"].get("no_input_layer", False):
            layer = layer - 1

        if self.config["rnn_layer"].get("single_layer", False):
            layer = 1

        if self.config["other_stuff"].get("raw_embedding", False):
            layer = 1

        self.plain_representations = plain_representations
        if not plain_representations:
            for k in range(num_output_representations):
                if only_use_embedding:
                    scalar_mix = ScalarMix(1, do_layer_norm=do_layer_norm)
                elif self.include_embedding:
                    scalar_mix = ScalarMix(layer + 1, do_layer_norm=do_layer_norm)
                else:
                    scalar_mix = ScalarMix(layer, do_layer_norm=do_layer_norm)

                self.add_module('scalar_mix_{}'.format(k), scalar_mix)
                self._scalar_mixes.append(scalar_mix)

        self._dropout = nn.Dropout(dropout)

    def forward(self, embedded, mask = None):
        if self.only_use_embedding:
            outputs = embedded[0].unsqueeze(0)
        elif self.config["other_stuff"].get("raw_embedding", False):
            outputs = embedded.unsqueeze(0)
        else:
            ############## First get through RNN to get the raw outputs
            if self.rnn_layer_type == "sru" or self.rnn_layer_type == "transformer" or self.rnn_layer_type == "elmo" or self.rnn_layer_type == "qrnn":
                outputs = self.rnn.get_all_layer_output(embedded, mask = mask, no_carry = False)
                outputs = outputs * mask.float().unsqueeze(0).unsqueeze(-1).expand_as(outputs) # Apply Mask

                if self.rnn_layer_type != "transformer": # For transformer, do not inverse representations; For SRU and elmo, we need to inverse the representations
                    outputs[:, :, :, int(outputs.size(-1) / 2): ] = inverse_representations(outputs.transpose(0,1).transpose(1,2)[:, :, :, int(outputs.size(-1) / 2): ], mask).transpose(1,2).transpose(0,1)

                if self.config["rnn_layer"].get("unidirectional", False):
                    # Very special, we discard the backward representation
                    outputs =  outputs[:, :, :, :int(outputs.size(-1) / 2)]
                elif self.config["rnn_layer"].get("unidirectional_backward", False):
                    # Very special, we discard the backward representation
                    outputs =  outputs[:, :, :, int(outputs.size(-1) / 2):]

                assert(outputs.dim() == 4)

            elif self.rnn_layer_type == "original_elmo":
                assert(mask is not None)
                outputs = self.rnn(embedded, mask = mask)
            else:
                assert(0) # Unsupported LSTM type

            if self.include_embedding:
                if isinstance(embedded, tuple):
                    outputs = cat_to_outputs(embedded[0], embedded[1], outputs, True)
                else:
                    outputs = cat_to_outputs(embedded, embedded, outputs, True)
                assert(outputs.size(0) == self.layer + 1)

        if self.config["rnn_layer"].get("single_layer", False):
            outputs = outputs[self.config["rnn_layer"]["single_layer"]].unsqueeze(0)

        if self.config["rnn_layer"].get("exclude_layer", False):
            outputs = outputs[:self.config["rnn_layer"]["exclude_layer"]].unsqueeze(0)

        if self.config["other_stuff"].get("no_input_layer", False):
            outputs = outputs[1:]

        if self.plain_representations:
            return [outputs]
        else:
            representations = []
            for i in range(len(self._scalar_mixes)):
                scalar_mix_i = getattr(self, 'scalar_mix_{}'.format(i))
                new_outputs = scalar_mix_i(outputs, mask)
                new_outputs = new_outputs * mask.float().unsqueeze(-1).expand_as(new_outputs)
                assert(new_outputs.dim() == 3)
                representations.append(self._dropout(new_outputs))

            return representations

def cat_to_outputs(forward_one_layer, backward_one_layer, outputs, insert_first):
    embedded_0_list = []
    dim = 0
    while dim < outputs.size(-1) / 2:
        embedded_0_list.append(forward_one_layer)
        dim += forward_one_layer.size(-1)

    slice_dim = int(outputs.size(-1) / 2)

    embedded_0_list = torch.cat(embedded_0_list, dim = -1)[:, :, :slice_dim]

    embedded_1_list = []
    dim = 0
    while dim < outputs.size(-1) / 2:
        embedded_1_list.append(backward_one_layer)
        dim += backward_one_layer.size(-1)
    embedded_1_list = torch.cat(embedded_1_list, dim = -1)[:, :, :slice_dim]

    embedded = torch.cat((embedded_0_list, embedded_1_list), dim = -1)
    assert(embedded.size(-1) == outputs.size(-1))

    if insert_first:
        outputs = torch.cat((embedded.unsqueeze(0), outputs), dim = 0)
    else:
        outputs = torch.cat((outputs, embedded.unsqueeze(0)), dim = 0)
    return outputs

def inverse_input(inputs, mask):
    # we create another tensor
    try:
        inputs_numpy = inputs.cpu().numpy()
        mask = mask.cpu().numpy()
        using_py03 = 0
    except:
        inputs_numpy = inputs.data.cpu().numpy()
        mask = mask.data.cpu().numpy()
        using_py03 = 1

    new_input = numpy.zeros(inputs_numpy.shape, inputs_numpy.dtype)
    lengths = numpy.sum(mask, 1)
    max_length = lengths.max()
    for i in range(inputs_numpy.shape[0]):
        new_input[i][:lengths[i]] = np.flip(inputs_numpy[i][:lengths[i]], 0).copy() # do not flip the mask
    new_input = torch.from_numpy(new_input)
    if using_py03:
        new_input = torch.autograd.Variable(new_input)
    return new_input

def inverse_representations(inputs, mask):
    assert(inputs.requires_grad == False) # this is destructive operation for data with gradients
    # we create another tensor
    try:
        inputs_numpy = inputs.cpu().numpy()
        mask = mask.cpu().numpy()
        using_py03 = 0
    except:
        inputs_numpy = inputs.data.cpu().numpy()
        mask = mask.data.cpu().numpy()
        using_py03 = 1
    new_input = numpy.zeros(inputs_numpy.shape, inputs_numpy.dtype)
    lengths = numpy.sum(mask, 1)
    #print(lengths)
    for i in range(inputs_numpy.shape[0]):
        new_input[i][:lengths[i]] = np.flip(inputs_numpy[i][:lengths[i]], 0).copy() # do not flip the mask
    new_input = torch.from_numpy(new_input).cuda()
    if using_py03:
        new_input = torch.autograd.Variable(new_input)
    return new_input

