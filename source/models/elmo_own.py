import json
import logging
from typing import Union, List, Dict, Any
import warnings

import torch
from torch.nn.modules import Dropout

import numpy
with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=FutureWarning)
    import h5py
from overrides import overrides

from allennlp.common.file_utils import cached_path
from allennlp.common.checks import ConfigurationError
from allennlp.common import Params
from allennlp.common.util import lazy_groups_of
from allennlp.modules.elmo_lstm import ElmoLstm
from allennlp.modules.highway import Highway
from allennlp.modules.scalar_mix import ScalarMix
from allennlp.nn.util import remove_sentence_boundaries, add_sentence_boundary_token_ids, get_device_of
from allennlp.data.token_indexers.elmo_indexer import ELMoCharacterMapper, ELMoTokenCharactersIndexer
from allennlp.data.dataset import Batch
from allennlp.data import Token, Vocabulary, Instance
from allennlp.data.fields import TextField

from elmo_c.source.helpers import *

def batch_to_ids(batch: List[List[str]]) -> torch.Tensor:
    """
    Converts a batch of tokenized sentences to a tensor representing the sentences with encoded characters
    (len(batch), max sentence length, max word length).

    Parameters
    ----------
    batch : ``List[List[str]]``, required
        A list of tokenized sentences.

    Returns
    -------
        A tensor of padded character ids.
    """
    instances = []
    indexer = ELMoTokenCharactersIndexer()
    for sentence in batch:
        tokens = [Token(token) for token in sentence]
        field = TextField(tokens,
                          {'character_ids': indexer})
        instance = Instance({"elmo": field})
        instances.append(instance)

    dataset = Batch(instances)
    vocab = Vocabulary()
    dataset.index_instances(vocab)
    return dataset.as_tensor_dict()['elmo']['character_ids']

class SRUWrapper(torch.nn.Module):
    def __init__(self, 
        input_size, 
        projection_size,
        hidden_size,
        num_layers,          # number of stacking RNN layers
        dropout,           # dropout applied between RNN layers
        rnn_dropout,       # variational dropout applied on linear transformation
        
        bidirectional,   # bidirectional RNN ?
        weight_norm,     # apply weight normalization on parameters
        layer_norm,      # apply layer normalization on the output of each layer
        highway_bias,         # initial bias of highway gate (<= 0)
        rescale,
        common_crawl_style,
        use_new_sru,
        v1,
        input_proj,
        is_input_normalized,
        clip_value,
        batch_first = True,
        reset_hidden_every_time = True,
        use_tanh = 1,            # use tanh?
        use_relu = 0,            # use ReLU?
        use_selu = 0,            # use SeLU?,


        use_lstm = False
        ):
        super(SRUWrapper, self).__init__()
        self.batch_first = batch_first
        self.bidirectional = bidirectional
        self.use_new_sru = use_new_sru
        self.reset_hidden_every_time = reset_hidden_every_time

        self.use_lstm = use_lstm
        if use_lstm:
            self.sru = torch.nn.LSTM(
                input_size = input_size, 
                hidden_size = hidden_size,
                num_layers = num_layers,  
                batch_first = False
                )
        else:
            if use_new_sru:
                from .legacy.new_sru import SRU
                self.sru = SRU(
                input_size = input_size, 
                n_proj = projection_size,
                hidden_size = hidden_size,
                is_input_normalized = is_input_normalized,
                input_proj = input_proj,
                num_layers = num_layers,          # number of stacking RNN layers
                dropout = dropout,           # dropout applied between RNN layers
                rnn_dropout = rnn_dropout,       # variational dropout applied on linear transformation
                use_tanh = use_tanh,            # use tanh?
                use_relu = use_relu,            # use ReLU?
                use_selu = use_selu,            # use SeLU?
                bidirectional = False,   # bidirectional RNN ?
                weight_norm = weight_norm,     # apply weight normalization on parameters
                layer_norm = layer_norm,      # apply layer normalization on the output of each layer
                highway_bias = highway_bias,
                rescale = rescale,
                v1 = v1,
                clip_value = clip_value
                )
            else:
                from .legacy.sru import SRU, SRUCell, SRUWithProjection
                if projection_size != 0:
                    self.sru = SRUWithProjection(
                    input_size = input_size, 
                    projection_dim = projection_size,
                    hidden_size = hidden_size,
                    num_layers = num_layers,          # number of stacking RNN layers
                    dropout = dropout,           # dropout applied between RNN layers
                    rnn_dropout = rnn_dropout,       # variational dropout applied on linear transformation
                    use_tanh = use_tanh,            # use tanh?
                    use_relu = use_relu,            # use ReLU?
                    use_selu = use_selu,            # use SeLU?
                    bidirectional = False,   # bidirectional RNN ?
                    weight_norm = weight_norm,     # apply weight normalization on parameters
                    layer_norm = layer_norm,      # apply layer normalization on the output of each layer
                    highway_bias = highway_bias,
                    rescale = rescale
                    )
                    print("############# Caution!!! Using SRU with projection!!")
                else:
                    self.sru = SRU(
                        input_size, 
                        hidden_size,
                        num_layers = num_layers,          # number of stacking RNN layers
                        dropout = dropout,           # dropout applied between RNN layers
                        rnn_dropout = rnn_dropout,       # variational dropout applied on linear transformation
                        use_tanh = use_tanh,            # use tanh?
                        use_relu = use_relu,            # use ReLU?
                        use_selu = use_selu,            # use SeLU?
                        bidirectional = False,   # bidirectional RNN ?
                        weight_norm = weight_norm,     # apply weight normalization on parameters
                        layer_norm = layer_norm,      # apply layer normalization on the output of each layer
                        highway_bias = highway_bias,
                        rescale = rescale
                    )
        
        self.hidden = None
        self.saved_indicator = None
        self.common_crawl_style = common_crawl_style
        '''self.hidden_pass_type = hidden_pass_type 
        if self.hidden_pass_type == 3:
            print("#### Caution!! SRU with precise hidden pass, may be extremely slow!!")'''

    def forward(self, input, mask = None, hidden = None):

        if self.use_lstm:
            if self.batch_first:
                input = input.transpose(0, 1)
            output, hidden = self.sru(input)
            return output

        if hidden is None:
            self.adjust_hidden(input = input) # attention, sru now automatically do hidden adjust This also contains detach. Adjust before the transpose

            if self.batch_first:
                input = input.transpose(0, 1)
            if self.hidden is not None:
                assert(self.hidden.size(1) == input.size(1)) # as input has been transfered into seq_len x batch x dim
                
            # We actually don't do any mask as it won't affect the results for TRANING
            output, hidden = self.sru(input, self.hidden)

            if self.batch_first:
                output = output.transpose(0,1)

            self.hidden = detach_hidden(hidden) # update self.hidden

            if self.common_crawl_style:
                self.update_indicator(mask) # input now is not batch first!!!

            # we do nothing about hidden because that's expected
            return output
        else:
            if self.batch_first:
                input = input.transpose(0, 1)
                hidden = hidden.transpose(0,1)
                assert(hidden.size(1) == input.size(1)) # as input has been transfered into seq_len x batch x dim
            output, hidden = self.sru(input, hidden)
            if self.batch_first:
                output = output.transpose(0,1)
                hidden = hidden.transpose(0,1) # Transpose back
            return output, hidden

    # This is a Type Three (Correct) version
    def get_all_layer_output(self, input, mask, no_carry = False):
        if not self.reset_hidden_every_time: # correct
            layers_hidden_results = [] # we keep it as a list right now and we will stack it later
            layers_output_results = [] # notice that because the SRU keeps a residual connection, so output != hidden

            self.adjust_hidden(input = input)
            # Could this be faster?
            for i in range(input.size(1)):
                output, hidden = self.return_hidden_all_layer(input[:, i].unsqueeze(1), adjust = False)
                output = output.squeeze(2)
                layers_output_results.append(output)
                layers_hidden_results.append(hidden)

            layers_hidden_results = self.stack_hidden(layers_hidden_results)
            layers_output_results = self.stack_hidden(layers_output_results)

            # now we need to select the correct hidden
            # length need to minus one because index starts from zero
            full_length = torch.sum(mask, dim = 1) - 1

            full_length = full_length.unsqueeze(0).unsqueeze(-1).expand(layers_hidden_results.size(0), layers_hidden_results.size(1), layers_hidden_results.size(-1))
            full_length = full_length.unsqueeze(2) # insert a dimension at the seq_len dimension
            self.hidden = torch.gather(input = layers_hidden_results, dim=2, index = full_length).squeeze(2)
            assert(self.hidden.dim() == 3)
            assert(self.hidden.size(0) == layers_hidden_results.size(0))

            # we return the outputs, not the memories!!
            return layers_output_results

        elif self.reset_hidden_every_time: # (Reset Hidden All the Time)
            self.adjust_hidden(input = input)
            if self.batch_first:
                input = input.transpose(0, 1)

            if self.hidden is not None:
                assert(self.hidden.size(1) == input.size(1))

            layers_output_results, self.hidden = self.sru.get_all_layer_output(input, self.hidden)
            layers_output_results = layers_output_results.transpose(1,2)
            #layers_output_results = layer x batch x len x dim
            assert(layers_output_results.size(2) == input.size(0))
            self.hidden = None

            return layers_output_results

    def return_hidden_all_layer(self, input, adjust = True):
        # This function assumes batch x sequence_len

        if adjust:
            self.adjust_hidden(input = input) # attention, sru now automatically do hidden adjust This also contains detach
        if self.batch_first:
            input = input.transpose(0, 1)
        if self.hidden is not None:
            assert(self.hidden.size(1) == input.size(1)) # as input has been transfered into seq_len x batch x dim
        output, hidden = self.sru.get_all_layer_output(input, self.hidden)
        if self.batch_first:
            output = output.transpose(1,2) # this is the difference
        self.hidden = detach_hidden(hidden) # update self.hidden

        # we do nothing about hidden because that's expected
        return output, hidden

    def stack_hidden(self, hidden_list):
        if isinstance(hidden_list[0], tuple) or isinstance(hidden_list[0], list):
            hidden_1 = [i[0] for i in hidden_list]
            hidden_2 = [i[1] for i in hidden_list]
            assert(len(hidden_list[0]) == 2)
            output = (torch.stack(hidden_1, 2), torch.stack(hidden_2, 2))
        else:
            output = torch.stack(hidden_list, 2)
        # output = layer x batch x seq_len x hidden_size
        return output


    def adjust_hidden(self, input):
        # Input is assumed to be batch first
        if self.hidden is None:
            return
        if self.saved_indicator is not None:
            self.hidden = self.hidden * self.saved_indicator
        if self.hidden.size(1) > input.size(0): # batch first
            self.hidden = self.hidden[:, :input.size(0), :]
        elif self.hidden.size(1) < input.size(0):
            self.hidden = torch.cat((self.hidden, torch.zeros(self.hidden.size(0), input.size(0) - self.hidden.size(1), self.hidden.size(2)).cuda()), dim = 1)
            #self.hidden = self.hidden[:, 0, :].unsqueeze(1).expand(self.hidden.size(0), input.size(0), self.hidden.size(2))
        self.hidden = self.hidden.detach()

    def update_indicator(self, mask):
        assert(mask is not None)
        self.saved_indicator = mask[:, -1].unsqueeze(0).unsqueeze(-1).float() # we get the last index

class QRNNWrapper(torch.nn.Module):
    def __init__(self, 
        input_size, 
        projection_size,
        hidden_size,
        num_layers,          # number of stacking RNN layers
        dropout,           # dropout applied between RNN layers
        layer_norm,
        common_crawl_style,
        input_proj,
        clip_value,
        batch_first = True,
        reset_hidden_every_time = True,
        save_prev_x=False, 
        zoneout=0, 
        window=1,
        ):
        super(QRNNWrapper, self).__init__()
        self.batch_first = batch_first
        self.reset_hidden_every_time = reset_hidden_every_time
        
        self.sru = QRNNProjection(
                input_size = input_size, 
                projection_size = projection_size,
                hidden_size = hidden_size,
                num_layers = num_layers,          # number of stacking RNN layers
                dropout = dropout,           # dropout applied between RNN layers
                layer_norm = layer_norm,
                bidirectional = False,   # bidirectional RNN ?
                input_proj = input_proj,
                clip_value = clip_value,
                save_prev_x=False, zoneout=0, window=1
                )
        self.hidden = None
        self.saved_indicator = None
        self.common_crawl_style = common_crawl_style

    def forward(self, input, mask = None, hidden = None):

        if hidden is None:
            self.adjust_hidden(input = input) # attention, sru now automatically do hidden adjust This also contains detach. Adjust before the transpose

            if self.batch_first:
                input = input.transpose(0, 1)
            if self.hidden is not None:
                assert(self.hidden.size(1) == input.size(1)) # as input has been transfered into seq_len x batch x dim
                
            # We actually don't do any mask as it won't affect the results for TRANING
            output, hidden = self.sru(input, self.hidden)

            if self.batch_first:
                output = output.transpose(0,1)

            self.hidden = detach_hidden(hidden) # update self.hidden

            if self.common_crawl_style:
                self.update_indicator(mask) # input now is not batch first!!!

            # we do nothing about hidden because that's expected
            return output
        else:
            if self.batch_first:
                input = input.transpose(0, 1)
                hidden = hidden.transpose(0,1)
                assert(hidden.size(1) == input.size(1)) # as input has been transfered into seq_len x batch x dim
            output, hidden = self.sru(input, hidden)
            if self.batch_first:
                output = output.transpose(0,1)
                hidden = hidden.transpose(0,1) # Transpose back
            return output, hidden

    def get_all_layer_output(self, input, mask, no_carry = False):
        if not self.reset_hidden_every_time: # correct
            layers_hidden_results = [] # we keep it as a list right now and we will stack it later
            layers_output_results = [] # notice that because the SRU keeps a residual connection, so output != hidden

            self.adjust_hidden(input = input)
            # Could this be faster?
            for i in range(input.size(1)):
                output, hidden = self.return_hidden_all_layer(input[:, i].unsqueeze(1), adjust = False)
                output = output.squeeze(2)
                layers_output_results.append(output)
                layers_hidden_results.append(hidden)

            layers_hidden_results = self.stack_hidden(layers_hidden_results)
            layers_output_results = self.stack_hidden(layers_output_results)

            # now we need to select the correct hidden
            # length need to minus one because index starts from zero
            full_length = torch.sum(mask, dim = 1) - 1

            full_length = full_length.unsqueeze(0).unsqueeze(-1).expand(layers_hidden_results.size(0), layers_hidden_results.size(1), layers_hidden_results.size(-1))
            full_length = full_length.unsqueeze(2) # insert a dimension at the seq_len dimension
            self.hidden = torch.gather(input = layers_hidden_results, dim=2, index = full_length).squeeze(2)
            assert(self.hidden.dim() == 3)
            assert(self.hidden.size(0) == layers_hidden_results.size(0))

            # we return the outputs, not the memories!!
            return layers_output_results

        elif self.reset_hidden_every_time: # (Reset Hidden All the Time)
            self.adjust_hidden(input = input)
            if self.batch_first:
                input = input.transpose(0, 1)

            if self.hidden is not None:
                assert(self.hidden.size(1) == input.size(1))

            layers_output_results, self.hidden = self.sru.get_all_layer_output(input, self.hidden)
            layers_output_results = layers_output_results.transpose(1,2)
            #layers_output_results = layer x batch x len x dim
            assert(layers_output_results.size(2) == input.size(0))
            self.hidden = None

            return layers_output_results

    def return_hidden_all_layer(self, input, adjust = True):
        # This function assumes batch x sequence_len

        if adjust:
            self.adjust_hidden(input = input) # attention, sru now automatically do hidden adjust This also contains detach
        if self.batch_first:
            input = input.transpose(0, 1)
        if self.hidden is not None:
            assert(self.hidden.size(1) == input.size(1)) # as input has been transfered into seq_len x batch x dim
        output, hidden = self.sru.get_all_layer_output(input, self.hidden)
        if self.batch_first:
            output = output.transpose(1,2) # this is the difference
        self.hidden = detach_hidden(hidden) # update self.hidden

        # we do nothing about hidden because that's expected
        return output, hidden

    def stack_hidden(self, hidden_list):
        if isinstance(hidden_list[0], tuple) or isinstance(hidden_list[0], list):
            hidden_1 = [i[0] for i in hidden_list]
            hidden_2 = [i[1] for i in hidden_list]
            assert(len(hidden_list[0]) == 2)
            output = (torch.stack(hidden_1, 2), torch.stack(hidden_2, 2))
        else:
            output = torch.stack(hidden_list, 2)
        # output = layer x batch x seq_len x hidden_size
        return output


    def adjust_hidden(self, input):
        # Input is assumed to be batch first
        if self.hidden is None:
            return
        if self.saved_indicator is not None:
            self.hidden = self.hidden * self.saved_indicator
        if self.hidden.size(1) > input.size(0): # batch first
            self.hidden = self.hidden[:, :input.size(0), :]
        elif self.hidden.size(1) < input.size(0):
            self.hidden = torch.cat((self.hidden, torch.zeros(self.hidden.size(0), input.size(0) - self.hidden.size(1), self.hidden.size(2)).cuda()), dim = 1)
            #self.hidden = self.hidden[:, 0, :].unsqueeze(1).expand(self.hidden.size(0), input.size(0), self.hidden.size(2))
        self.hidden = self.hidden.detach()

    def update_indicator(self, mask):
        assert(mask is not None)
        self.saved_indicator = mask[:, -1].unsqueeze(0).unsqueeze(-1).float() # we get the last index

class _ElmoCharacterEncoder(torch.nn.Module):

    def __init__(self,
                 options_file: str,
                 weight_file: str,
                 requires_grad: bool = False,
                 dropout: float = 0.0) -> None:
        super(_ElmoCharacterEncoder, self).__init__()

        '''with open(options_file, 'r') as fin:
            self._options = json.load(fin)
        '''

        # Instead of passing a file, we directly pass options!!
        self._options = options_file

        self._weight_file = weight_file

        self.output_dim = self._options['lstm']['projection_dim']
        self.requires_grad = requires_grad

        # Caution, this function also creates some layers, so we cann't simply ignore it
        self._load_weights()

        # Cache the arrays for use in forward -- +1 due to masking.

        self._beginning_of_sentence_characters = torch.from_numpy(
                numpy.array(ELMoCharacterMapper.beginning_of_sentence_characters) + 1
        )
        self._end_of_sentence_characters = torch.from_numpy(
                numpy.array(ELMoCharacterMapper.end_of_sentence_characters) + 1
        )
        self.dropout = torch.nn.Dropout(dropout)

        cnn_options = self._options['char_cnn']
        filters = cnn_options['filters']
        n_filters = sum(f[1] for f in filters)

        if cnn_options.get("batch_norm_after_feature", False):
            self.bn_feature = torch.nn.BatchNorm1d(num_features=n_filters)
        if cnn_options.get("batch_norm_output", False):
            self.bn_output = torch.nn.BatchNorm1d(num_features=self.output_dim)

    def get_output_dim(self):
        return self.output_dim

    @overrides
    def forward(self, inputs: torch.Tensor) -> Dict[str, torch.Tensor]:  # pylint: disable=arguments-differ
        """
        Compute context insensitive token embeddings for ELMo representations.

        Parameters
        ----------
        inputs: ``torch.Tensor``
            Shape ``(batch_size, sequence_length, 50)`` of character ids representing the
            current batch.

        Returns
        -------
        Dict with keys:
        ``'token_embedding'``: ``torch.Tensor``
            Shape ``(batch_size, sequence_length + 2, embedding_dim)`` tensor with context
            insensitive token representations.
        ``'mask'``:  ``torch.Tensor``
            Shape ``(batch_size, sequence_length + 2)`` long tensor with sequence mask.
        """
        # WE DO NOT Add BOS/EOS !!!!
        mask = ((inputs > 0).long().sum(dim=-1) > 0).long()

        character_ids_with_bos_eos = inputs
        mask_with_bos_eos = mask

        ''', mask_with_bos_eos = add_sentence_boundary_token_ids(
                inputs,
                mask,
                self._beginning_of_sentence_characters,
                self._end_of_sentence_characters
        )'''

        # the character id embedding
        max_chars_per_token = self._options['char_cnn']['max_characters_per_token']
        # (batch_size * sequence_length, max_chars_per_token, embed_dim)

        character_embedding = torch.nn.functional.embedding(
                character_ids_with_bos_eos.view(-1, max_chars_per_token),
                self._char_embedding_weights
        )

        # run convolutions
        cnn_options = self._options['char_cnn']
        if cnn_options['activation'] == 'tanh':
            activation = torch.nn.functional.tanh
        elif cnn_options['activation'] == 'relu':
            activation = torch.nn.functional.relu
        else:
            raise ConfigurationError("Unknown activation")

        # (batch_size * sequence_length, embed_dim, max_chars_per_token)
        character_embedding = torch.transpose(character_embedding, 1, 2)

        character_embedding = self.dropout(character_embedding)

        convs = []
        for i in range(len(self._convolutions)):
            conv = getattr(self, 'char_conv_{}'.format(i))
            convolved = conv(character_embedding)
            # (batch_size * sequence_length, n_filters for this width)
            convolved, _ = torch.max(convolved, dim=-1)
            convolved = activation(convolved)
            convs.append(convolved)

        # (batch_size * sequence_length, n_filters)
        token_embedding = torch.cat(convs, dim=-1)
        if hasattr(self, "bn_feature"):
            token_embedding = self.bn_feature(token_embedding)

        # apply the highway layers (batch_size * sequence_length, n_filters)
        token_embedding = self._highways(token_embedding)

        # final projection  (batch_size * sequence_length, embedding_dim)
        token_embedding = self._projection(token_embedding)

        if hasattr(self, "bn_output"):
            token_embedding = self.bn_output(token_embedding)

        # reshape to (batch_size, sequence_length, embedding_dim)
        batch_size, sequence_length, _ = character_ids_with_bos_eos.size()

        return {
                'mask': mask_with_bos_eos,
                'token_embedding': token_embedding.view(batch_size, sequence_length, -1)
        }

    def _load_weights(self):
        self._load_char_embedding()
        self._load_cnn_weights()
        self._load_highway()
        self._load_projection()
        print("Loaded CNN weights!")

    def _load_char_embedding(self):
        try:
            with h5py.File(self._weight_file, 'r') as fin:
                char_embed_weights = fin['char_embed'][...]

            weights = numpy.zeros(
                (char_embed_weights.shape[0] + 1, char_embed_weights.shape[1]),
                dtype='float32'
            )
            weights[1:, :] = char_embed_weights
        except:
            print("### Char_embedding load failed. Skipped loading. Using Magic number")
            weights = numpy.random.rand(
                261 + 1, 16).astype("float32") #this is magic number, should check later
        
        self._char_embedding_weights = torch.nn.Parameter(
                torch.FloatTensor(weights), requires_grad=self.requires_grad
        )

    def _load_cnn_weights(self):
        cnn_options = self._options['char_cnn']
        filters = cnn_options['filters']
        char_embed_dim = cnn_options['embedding']['dim']

        convolutions = []
        for i, (width, num) in enumerate(filters):
            conv = torch.nn.Conv1d(
                    in_channels=char_embed_dim,
                    out_channels=num,
                    kernel_size=width,
                    bias=True
            )
            try:
                # load the weights
                with h5py.File(self._weight_file, 'r') as fin:
                    weight = fin['CNN']['W_cnn_{}'.format(i)][...]
                    bias = fin['CNN']['b_cnn_{}'.format(i)][...]
    
                w_reshaped = numpy.transpose(weight.squeeze(axis=0), axes=(2, 1, 0))
                if w_reshaped.shape != tuple(conv.weight.data.shape):
                    raise ValueError("Invalid weight file")
                conv.weight.data.copy_(torch.FloatTensor(w_reshaped))
                conv.bias.data.copy_(torch.FloatTensor(bias))
            except:
                print("## Warning: Loading CNN weights failed. Skipped")

            conv.weight.requires_grad = self.requires_grad
            conv.bias.requires_grad = self.requires_grad

            convolutions.append(conv)
            self.add_module('char_conv_{}'.format(i), conv)

        self._convolutions = convolutions

    def _load_highway(self):
        # pylint: disable=protected-access
        # the highway layers have same dimensionality as the number of cnn filters
        cnn_options = self._options['char_cnn']
        filters = cnn_options['filters']
        n_filters = sum(f[1] for f in filters)
        n_highway = cnn_options['n_highway']

        # create the layers, and load the weights
        self._highways = Highway(n_filters, n_highway, activation=torch.nn.functional.relu)
        for k in range(n_highway):
            # The AllenNLP highway is one matrix multplication with concatenation of
            # transform and carry weights.
            try:
                with h5py.File(self._weight_file, 'r') as fin:
                    # The weights are transposed due to multiplication order assumptions in tf
                    # vs pytorch (tf.matmul(X, W) vs pytorch.matmul(W, X))
                    w_transform = numpy.transpose(fin['CNN_high_{}'.format(k)]['W_transform'][...])
                    # -1.0 since AllenNLP is g * x + (1 - g) * f(x) but tf is (1 - g) * x + g * f(x)
                    w_carry = -1.0 * numpy.transpose(fin['CNN_high_{}'.format(k)]['W_carry'][...])
                    weight = numpy.concatenate([w_transform, w_carry], axis=0)
                    self._highways._layers[k].weight.data.copy_(torch.FloatTensor(weight))
                

                    b_transform = fin['CNN_high_{}'.format(k)]['b_transform'][...]
                    b_carry = -1.0 * fin['CNN_high_{}'.format(k)]['b_carry'][...]
                    bias = numpy.concatenate([b_transform, b_carry], axis=0)
                    self._highways._layers[k].bias.data.copy_(torch.FloatTensor(bias))
            except:
                print("### Warning: Loading highway network weights failed. Skipped")

            #this has to be done whether or not we load the weights
            self._highways._layers[k].weight.requires_grad = self.requires_grad
            self._highways._layers[k].bias.requires_grad = self.requires_grad

    def _load_projection(self):
        cnn_options = self._options['char_cnn']
        filters = cnn_options['filters']
        n_filters = sum(f[1] for f in filters)

        self._projection = torch.nn.Linear(n_filters, self.output_dim, bias=True)

        try:
            with h5py.File(self._weight_file, 'r') as fin:
                weight = fin['CNN_proj']['W_proj'][...]
                bias = fin['CNN_proj']['b_proj'][...]
                self._projection.weight.data.copy_(torch.FloatTensor(numpy.transpose(weight)))
                self._projection.bias.data.copy_(torch.FloatTensor(bias))
        except:
            print("### Loading projection weights failed. Skipped")

        self._projection.weight.requires_grad = self.requires_grad
        self._projection.bias.requires_grad = self.requires_grad

