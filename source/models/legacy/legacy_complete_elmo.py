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

from elmo_c.source.models.elmo_own import _ElmoCharacterEncoder

from elmo_c.source.helpers import *

import sys

import copy
# SoftMaxes...

class GeneralRNN(torch.nn.Module):
    def __init__(self,
                config,
                input_word_vectors = None
                ) -> None:
        super(GeneralRNN, self).__init__()
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
        self.dropout_after_input = nn.Dropout(input_layer_config["dropout"])

        ##################### RNN
        rnn_layer_config = self.config["rnn_layer"]
        try:
            from torch.nn import LayerNorm
            if rnn_layer_config.get("custom_elmo", False):
                from elmo_c.source.models.custom_elmo_lstm import ElmoLstmUni
            else:
                from elmo_c.source.models.elmo_lstm import ElmoLstm, ElmoLstmUni
        except:
            from elmo_c.source.models.elmo_lstm_old import ElmoLstm, ElmoLstmUni

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
        else:
            assert(0) # Unsupported RNN type

        if rnn_layer_config["bidirectional"]:
            if rnn_layer_config["name"] == "sru":
                rnn_backward = SRUWrapper(
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
                rnn_backward = QRNNWrapper(
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
                rnn_backward = Transformer(
                   input_size = rnn_layer_config["input_size"],
                   N=rnn_layer_config["num_layers"],
                   d_model=rnn_layer_config["projection_size"],
                   d_ff=rnn_layer_config["hidden_size"],
                   h=rnn_layer_config["h"],
                   dropout=rnn_layer_config["rnn_dropout"],
                   reverse = True)
            elif rnn_layer_config['name'] == "elmo":
                rnn_backward =  ElmoLstmUni(
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
            rnn = BiDirectionalRNN(rnn_fowrad = rnn, rnn_backward = rnn_backward)

        try:
            print("There are " + str(parameters_count(rnn)) + " parameters in RNN")
        except:
            pass
        ##################### LAST LAYER
        last_layer_config = config["output_layer"]
        last_layer_insize = last_layer_config["input_size"]
        last_layer_outsize = last_layer_config.get("output_size", -1)
        _last_layer_backward = None # a place holder

        if last_layer_config["name"] == "semfit":
            _last_layer_forward = nn.Linear(last_layer_insize, last_layer_outsize)
            if rnn_layer_config["bidirectional"]:
                _last_layer_backward = nn.Linear(last_layer_insize, last_layer_outsize)

        # We always share the last layer
        elif last_layer_config["name"] == "sampled_softmax":
            _last_layer_forward = sampled_softmax.SampledSoftmax(ntokens = last_layer_config["vocab_size"], nsampled = last_layer_config["nsampled"], nhid = last_layer_insize, tied_weight = None, n_proj = rnn_layer_config["hidden_size"])

            _last_layer_backward = _last_layer_forward

        elif last_layer_config["name"] == "adaptive_softmax":
            if config['dataset'].get("hard_restrict_vocab", False):
                last_layer_config["vocab_size"] = config['dataset'].get("hard_restrict_vocab", False) + 10
                # We only use restricted vocabulary

            while last_layer_config["cutoff"][-1] > last_layer_config["vocab_size"]:
                last_layer_config["cutoff"] = last_layer_config["cutoff"][:-1]

            new_cutoff = last_layer_config["cutoff"]

            if last_layer_config["cutoff"][-1] < last_layer_config["vocab_size"]:
                new_cutoff = new_cutoff + [last_layer_config["vocab_size"]]
            _last_layer_forward = AdaptiveSoftmax(
                last_layer_insize, 
                new_cutoff,
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

        ##################### WRAP RNN AND LAST LAYER TO PARALLEL THE MODEL
        self.rnn_last_layer_wrapper = RNNLastLayerWrapper(
                rnn = rnn,
                _last_layer_forward = _last_layer_forward, 
                _last_layer_backward = _last_layer_backward, 
                config = config)

        ##################### INITIALIZE PARAMETER
        if config["other_stuff"]["initialize"]:
            try:
                self.init_weights(self._last_layer_backward)
                self.init_weights(self._last_layer_forward)
            except:
                print("Initialization Failed!")
                pass

        ##################### PARALLEL OPTIMIZATION
        if config["other_stuff"]["parallel_rnn_and_last_layer"]: # This is the best optimized version
            self.rnn_last_layer_wrapper = torch.nn.DataParallel(self.rnn_last_layer_wrapper)

        '''if config["other_stuff"]["whole_parallel"]:
            assert(config["other_stuff"]["not_parallel_embedding"] == False)
            assert(config["other_stuff"]["parallel_rnn_and_last_layer"] == False)
            self.parallel = torch.nn.DataParallel(self)'''

        self.embedding_parallel = config["input_layer"].get("parallel", False)
        self.special_parallel_strategy_for_adaptive_softmax = config["other_stuff"].get("special_parallel_strategy_for_adaptive_softmax", False)

    def forward(self,
                word_inputs,
                mask = None,
                hidden = None,
                return_hidden = False): # Only do this during training

        if self.config["input_layer"]["freeze"]:
            try:
                self.embedding.eval() # Very important as we added Batch Normalization to Pre-trained CNN
            except:
                pass
        ##################### input
        if self.embedding_parallel: # Special
            #return self.rnn_last_layer_wrapper(word_inputs, mask = mask, hidden = hidden)
            embedded, input_batches_words, input_word_backward = word_inputs
            embedded = embedded[0]
            embedded_0 = torch.nn.functional.embedding(input_batches_words, embedded)
            embedded_1 = torch.nn.functional.embedding(input_word_backward, embedded)
            embedded = (embedded_0, embedded_1)
            return self.rnn_last_layer_wrapper(embedded, mask = mask, hidden = hidden)

        if self.special_parallel_strategy_for_adaptive_softmax:
            return self.rnn_last_layer_wrapper(word_inputs, mask = mask, hidden = hidden, return_hidden = return_hidden)

        if self.config["input_layer"]["name"] == "cnn":
            if isinstance(word_inputs, tuple):
                original_shape = word_inputs[0].size()
                if len(original_shape) > 3:
                    timesteps, num_characters = original_shape[-2:]
                    reshaped_inputs = word_inputs[0].view(-1, timesteps, num_characters)
                else:
                    reshaped_inputs = word_inputs[0]
                token_embedding = self.embedding(reshaped_inputs)
                embedded_0 = token_embedding['token_embedding']
                if self.config["rnn_layer"]["bidirectional"]:
                    original_shape = word_inputs[1].size()
                    if len(original_shape) > 3:
                        timesteps, num_characters = original_shape[-2:]
                        reshaped_inputs = word_inputs[1].view(-1, timesteps, num_characters)
                    else:
                        #assert(0)
                        reshaped_inputs = word_inputs[1]
                    token_embedding = self.embedding(reshaped_inputs)
                    embedded_1 = token_embedding['token_embedding']
                    embedded = (embedded_0, embedded_1)
                else:
                    embedded = embedded_0             
            else:
                original_shape = word_inputs.size()
                if len(original_shape) > 3:
                    timesteps, num_characters = original_shape[-2:]
                    reshaped_inputs = word_inputs.view(-1, timesteps, num_characters)
                else:
                    #assert(0)
                    reshaped_inputs = word_inputs
                token_embedding = self.embedding(reshaped_inputs)
                embedded = token_embedding['token_embedding']
                embedded = self.dropout_after_input(embedded)
        elif self.config["input_layer"]["name"] == "acc_cnn": # This only works for bidirectional
            input_batches_words, input_word_backward, character_level_inputs, lenth_records = word_inputs
            assert(character_level_inputs.size(0) == 1)
            character_level_inputs = character_level_inputs[:, :lenth_records.item(), :]
            token_embedding = self.embedding(character_level_inputs)
            embedded = token_embedding['token_embedding'][0]
            embedded_0 = torch.nn.functional.embedding(input_batches_words, embedded)
            embedded_1 = torch.nn.functional.embedding(input_word_backward, embedded)
            embedded = (embedded_0, embedded_1)
        else:
            if isinstance(word_inputs, tuple):
                assert(len(word_inputs) == 2)
                embedded_forward = self.embedding(word_inputs[0])
                embedded_backward = self.embedding(word_inputs[1])

                if word_inputs[0].is_cuda:
                    embedded_forward = embedded_forward.cuda()                    
                    embedded_backward = embedded_backward.cuda()

                embedded_forward =  self.dropout_after_input(embedded_forward)
                embedded_backward = self.dropout_after_input(embedded_backward)
                if self.config["rnn_layer"]["bidirectional"]:
                    embedded = (embedded_forward, embedded_backward)
                else:
                    embedded = embedded_forward
            else:
                embedded = self.dropout_after_input(self.embedding(word_inputs))
        return self.rnn_last_layer_wrapper(embedded, mask = mask, hidden = hidden, return_hidden = return_hidden)
            
    def load_state_dict_from_file(self, models_path):
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
            self.rnn_last_layer_wrapper.module.rnn = BiDirectionalRNN(rnn_fowrad = forward_model.rnn_last_layer_wrapper.module.rnn, rnn_backward = backward_model.rnn_last_layer_wrapper.module.rnn)

    def init_weights(self, model):
        if model is None:
            return
        for p in model.parameters():
            if p.requires_grad == False:
                continue
            if p.dim() > 1:  # matrix
                nn.init.xavier_uniform(p)

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

# Not surpported if there is no RNN (CNN-embedding Model)
class RNNLastLayerWrapper(nn.Module):
    def __init__(self, rnn, _last_layer_forward, _last_layer_backward, config):
        super(RNNLastLayerWrapper, self).__init__()
        self.rnn = rnn
        self._last_layer_forward = _last_layer_forward
        self._last_layer_backward = _last_layer_backward
        self.config = config

        self.dropout_after_rnn_instance = nn.Dropout(self.config["output_layer"]["dropout"])

        self.last_layer_size = self.config["output_layer"]["input_size"]

    def forward(self, embedded, mask, hidden = None, return_hidden = False):

        if self.config["rnn_layer"]["name"] == "none":
            if isinstance(embedded, tuple):
                embedded = torch.cat(embedded, dim = -1)
            outputs = embedded
        else:
            if not return_hidden:
                outputs = self.rnn(embedded, mask = mask)
            else:
                outputs, hidden = self.rnn(embedded, mask = mask, hidden = hidden, return_hidden = return_hidden)

            ##################### Dropout after RNN
            if not isinstance(outputs, list) and self.config["output_layer"]["dropout"] != 0:
                outputs = self.dropout_after_rnn_instance(outputs)

        backward_output = None
        output_representations = self.config["other_stuff"].get("output_representations", False)
        if output_representations:
            return outputs # replace some of the function of ...
        ##################### LastLayer
        if self.config["output_layer"]["name"] != None:
            if self.config["rnn_layer"]["bidirectional"]:
                forward_output = outputs[:, :, :int(outputs.size(-1) / 2)]
            else:
                forward_output = outputs
            if not (self.config["output_layer"]["name"] == "adaptive_softmax" or self.config["output_layer"]["name"] == "sampled_softmax" or self.config["output_layer"]["name"] == "none"):
                forward_output = self._last_layer_forward(forward_output)

            if self.config["rnn_layer"]["bidirectional"]:
                backward_output = outputs[:, :, int(outputs.size(-1) / 2):]
                if not (self.config["output_layer"]["name"] == "adaptive_softmax" or self.config["output_layer"]["name"] == "sampled_softmax" or self.config["output_layer"]["name"] == "none"):
                    backward_output = self._last_layer_backward(backward_output)
            if return_hidden:
                return forward_output, backward_output, hidden
            else:
                return forward_output, backward_output
        else:
            return embedded, None

class ModelLoss(torch.nn.Module):

    '''Should be DataParallel friendly'''
    def __init__(self, model, loss, bidirectional, config):
        super(ModelLoss, self).__init__()

        self.model = model
        self.loss = loss
        self.bidirectional = bidirectional
        self.config = config

        self.reverse = config["rnn_layer"].get("reverse", False) # Hacky optimization!
        if self.reverse:
            assert(not config["rnn_layer"]["bidirectional"])

    def forward(self, input, forward_target, mask_forward, backward_target, mask_backward, hidden = None, return_hidden = False):

        # Wrapup mask
        if self.bidirectional:
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
        if not return_hidden:
            forward_output, backward_output = self.model(input, mask = mask, return_hidden = False)
        else:
            forward_output, backward_output, hidden = self.model(input, mask = mask, hidden = hidden, return_hidden = True)
        # might be buggy, we always assume batch first

        forward_loss = self.loss(logits = forward_output, target = forward_target, mask = mask_forward, dummy = 1).unsqueeze(0)
        if self.bidirectional:
            backward_loss = self.loss(logits = backward_output, target = backward_target, mask = mask_backward, dummy = 0).unsqueeze(0)
        else:
            backward_loss = None

        if not return_hidden:
            return forward_loss, backward_loss
        else:
            return forward_loss, backward_loss, hidden

class WeightedSumWrapper(torch.nn.Module): 
    def __init__(self, 
        rnn, 
        config,  

        num_output_representations = 1,
        include_embedding = False, 
        ditch_boundry = False, 
        include_last_layer_emb_model = False,
        only_use_embedding = False, 
        elmo_last_layer_forward = None, 
        elmo_last_layer_backward = None, 
        elmo_bidirectional = None,
        do_layer_norm = False,
        plain_representations = False):

        super(WeightedSumWrapper, self).__init__()

        layer = config["rnn_layer"]["num_layers"] 
        rnn_layer_type = config["rnn_layer"]["name"]
        dropout = config["rnn_layer"]["dropout"]
        self.config = config
        self.rnn = rnn
        self.rnn_layer_type = rnn_layer_type

        self.include_embedding = include_embedding
        self.only_use_embedding = only_use_embedding # This is to verify that only using FastText won't give that much good results
        self.include_last_layer_emb_model = include_last_layer_emb_model

        self.elmo_last_layer_forward = elmo_last_layer_forward
        self.elmo_last_layer_backward = elmo_last_layer_backward
        self.elmo_bidirectional = elmo_bidirectional

        if self.include_last_layer_emb_model:
            assert(elmo_last_layer_forward is not None)

        self.layer = layer
        self.dropout_rate = dropout
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
                else:
                    if self.include_embedding and self.include_last_layer_emb_model:
                        scalar_mix = ScalarMix(layer + 2, do_layer_norm=do_layer_norm)
                    elif self.include_embedding or self.include_last_layer_emb_model:
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

            if self.include_last_layer_emb_model:
                last_layer = outputs[-1, :, :, :]
                assert(outputs.size(-1) % 2 == 0)
                tmp_size = int(outputs.size(-1)/2)
                last_layer_output_forward = self.elmo_last_layer_forward(last_layer[:, :, :tmp_size])
                last_layer_output_backward = self.elmo_last_layer_backward(last_layer[:, :, tmp_size:])
                outputs = cat_to_outputs(last_layer_output_forward, last_layer_output_backward, outputs, False)

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

