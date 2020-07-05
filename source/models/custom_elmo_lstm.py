from typing import Tuple, Union, Optional, Callable
import torch
from torch.nn.utils.rnn import pack_padded_sequence, PackedSequence

from allennlp.nn.util import get_lengths_from_binary_sequence_mask, sort_batch_by_length

from typing import Optional, Tuple, List
import warnings

import torch
from torch.nn.utils.rnn import PackedSequence, pad_packed_sequence
with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=FutureWarning)
    import h5py
import numpy

from allennlp.common.checks import ConfigurationError
from allennlp.common.file_utils import cached_path
from typing import Optional, Tuple, List

import torch

from allennlp.nn.util import get_dropout_mask
from allennlp.nn.initializers import block_orthogonal


from .lstm_with_projection import LstmCellWithProjection

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

RnnState = Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]  # pylint: disable=invalid-name
RnnStateStorage = Tuple[torch.Tensor, ...]  # pylint: disable=invalid-name

# Classes in this file is only meat to be used in training time with return_hidden set to True!
class _EncoderBase(torch.nn.Module):
    # pylint: disable=abstract-method
    """
    This abstract class serves as a base for the 3 ``Encoder`` abstractions in AllenNLP.
    - :class:`~allennlp.modules.seq2seq_encoders.Seq2SeqEncoders`
    - :class:`~allennlp.modules.seq2vec_encoders.Seq2VecEncoders`

    Additionally, this class provides functionality for sorting sequences by length
    so they can be consumed by Pytorch RNN classes, which require their inputs to be
    sorted by length. Finally, it also provides optional statefulness to all of it's
    subclasses by allowing the caching and retrieving of the hidden states of RNNs.
    """
    def __init__(self, stateful: bool = False) -> None:
        super(_EncoderBase, self).__init__()
        self.stateful = stateful

    def sort_and_run_forward(self,
                             module: Callable[[PackedSequence, Optional[RnnState]],
                                              Tuple[Union[PackedSequence, torch.Tensor], RnnState]],
                             inputs: torch.Tensor,
                             mask: torch.Tensor,
                             hidden_states: Optional[RnnState] = None,
                             reset_hidden_state = False):
        # First count how many sequences are empty.
        batch_size = mask.size(0)
        num_valid = torch.sum(mask[:, 0]).int().item()

        sequence_lengths = get_lengths_from_binary_sequence_mask(mask)
        sorted_inputs, sorted_sequence_lengths, restoration_indices, sorting_indices =\
            sort_batch_by_length(inputs, sequence_lengths)

        # Now create a PackedSequence with only the non-empty, sorted sequences.
        packed_sequence_input = pack_padded_sequence(sorted_inputs[:num_valid, :, :],
                                                     sorted_sequence_lengths[:num_valid].data.tolist(),
                                                     batch_first=True)
        # Prepare the initial states.
        initial_states, hidden_states = self._get_initial_states(batch_size, num_valid, sorting_indices, hidden_states)

        if reset_hidden_state:
            initial_states = None

        # Actually call the module on the sorted PackedSequence.
        module_output, final_states = module(packed_sequence_input, initial_states)

        return module_output, final_states, restoration_indices, hidden_states

    def _get_initial_states(self,
                            batch_size: int,
                            num_valid: int,
                            sorting_indices: torch.LongTensor,
                            hidden_states) -> Optional[RnnState]:
        # We don't know the state sizes the first time calling forward,
        # so we let the module define what it's initial hidden state looks like.
        if hidden_states is None:
            return None, None

        # Otherwise, we have some previous states.
        if batch_size > hidden_states[0].size(1):
            # This batch is larger than the all previous states.
            # If so, resize the states.
            num_states_to_concat = batch_size - hidden_states[0].size(1)
            resized_states = []
            # state has shape (num_layers, batch_size, hidden_size)
            for state in hidden_states:
                # This _must_ be inside the loop because some
                # RNNs have states with different last dimension sizes.
                zeros = state.new_zeros(state.size(0),
                                        num_states_to_concat,
                                        state.size(2))
                resized_states.append(torch.cat([state, zeros], 1))
            hidden_states = tuple(resized_states)
            correctly_shaped_states = hidden_states

        elif batch_size < hidden_states[0].size(1):
            # This batch is smaller than the previous one.
            correctly_shaped_states = tuple(state[:, :batch_size, :] for state in hidden_states)
        else:
            correctly_shaped_states = hidden_states

        # At this point, our states are of shape (num_layers, batch_size, hidden_size).
        # However, the encoder uses sorted sequences and additionally removes elements
        # of the batch which are fully padded. We need the states to match up to these
        # sorted and filtered sequences, so we do that in the next two blocks before
        # returning the state/s.
        if len(hidden_states) == 1:
            # GRUs only have a single state. This `unpacks` it from the
            # tuple and returns the tensor directly.
            correctly_shaped_state = correctly_shaped_states[0]
            sorted_state = correctly_shaped_state.index_select(1, sorting_indices)
            return sorted_state[:, :num_valid, :], hidden_states
        else:
            # LSTMs have a state tuple of (state, memory).
            sorted_states = [state.index_select(1, sorting_indices)
                             for state in correctly_shaped_states]
            return tuple(state[:, :num_valid, :] for state in sorted_states), hidden_states

    def _update_states(self, final_states: RnnStateStorage, restoration_indices: torch.LongTensor, hidden_states) -> None:
        new_unsorted_states = [state.index_select(1, restoration_indices)
                               for state in final_states]
        if hidden_states is None:
            # We don't already have states, so just set the
            # ones we receive to be the current state.
            hidden_states = tuple(state.data for state in new_unsorted_states)
        else:
            # Now we've sorted the states back so that they correspond to the original
            # indices, we need to figure out what states we need to update, because if we
            # didn't use a state for a particular row, we want to preserve its state.
            # Thankfully, the rows which are all zero in the state correspond exactly
            # to those which aren't used, so we create masks of shape (new_batch_size,),
            # denoting which states were used in the RNN computation.
            current_state_batch_size = hidden_states[0].size(1)
            new_state_batch_size = final_states[0].size(1)
            # Masks for the unused states of shape (1, new_batch_size, 1)
            used_new_rows_mask = [(state[0, :, :].sum(-1)
                                   != 0.0).float().view(1, new_state_batch_size, 1)
                                  for state in new_unsorted_states]
            new_states = []
            if current_state_batch_size > new_state_batch_size:
                # The new state is smaller than the old one,
                # so just update the indices which we used.
                for old_state, new_state, used_mask in zip(hidden_states,
                                                           new_unsorted_states,
                                                           used_new_rows_mask):
                    # zero out all rows in the previous state
                    # which _were_ used in the current state.
                    masked_old_state = old_state[:, :new_state_batch_size, :] * (1 - used_mask)
                    # The old state is larger, so update the relevant parts of it.
                    old_state[:, :new_state_batch_size, :] = new_state + masked_old_state
                    new_states.append(old_state.detach())
            else:
                # The states are the same size, so we just have to
                # deal with the possibility that some rows weren't used.
                new_states = []
                for old_state, new_state, used_mask in zip(hidden_states,
                                                           new_unsorted_states,
                                                           used_new_rows_mask):
                    # zero out all rows which _were_ used in the current state.
                    masked_old_state = old_state * (1 - used_mask)
                    # The old state is larger, so update the relevant parts of it.
                    new_state += masked_old_state
                    new_states.append(new_state.detach())

            # It looks like there should be another case handled here - when
            # the current_state_batch_size < new_state_batch_size. However,
            # this never happens, because the states themeselves are mutated
            # by appending zeros when calling _get_inital_states, meaning that
            # the new states are either of equal size, or smaller, in the case
            # that there are some unused elements (zero-length) for the RNN computation.
            hidden_states = tuple(new_states)
        return hidden_states

class ElmoLstmUni(_EncoderBase):
    def __init__(self,
                 input_size: int,
                 hidden_size: int,
                 cell_size: int,
                 num_layers: int,
                 requires_grad: bool = False,
                 recurrent_dropout_probability: float = 0.0,
                 memory_cell_clip_value: Optional[float] = None,
                 state_projection_clip_value: Optional[float] = None,
                 layer_norm = False,
                 ln_before_act = False,
                 correct_layer_norm= False,
                 add_embedding_layer = False,
                 reset_hidden_state = False,
                 output_pre_norm= False) -> None:
        super(ElmoLstmUni, self).__init__(stateful=True)

        self.reset_hidden_state = reset_hidden_state
        if reset_hidden_state:
            print("########## Warning!! You are setting reset hidden to true. Only support this in evaluation! #######")
        # Required to be wrapped with a :class:`PytorchSeq2SeqWrapper`.
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.cell_size = cell_size
        self.requires_grad = requires_grad

        self.add_embedding_layer = add_embedding_layer
        self.layer_norm = layer_norm
        self.ln_before_act = ln_before_act

        if hidden_size != input_size:
            self.input_projection = torch.nn.Linear(input_size, hidden_size)
        else:
            self.input_projection = None

        forward_layers = []
        backward_layers = []

        lstm_input_size = hidden_size

        go_forward = True
        for layer_index in range(num_layers):
            forward_layer = LstmCellWithProjection(lstm_input_size,
                                                   hidden_size,
                                                   cell_size,
                                                   go_forward,
                                                   recurrent_dropout_probability,
                                                   memory_cell_clip_value,
                                                   state_projection_clip_value,
                                                   layer_norm = layer_norm,
                                                   correct_layer_norm = correct_layer_norm,
                                                   output_pre_norm = output_pre_norm)
            lstm_input_size = hidden_size

            self.add_module('forward_layer_{}'.format(layer_index), forward_layer)
            forward_layers.append(forward_layer)
        self.forward_layers = forward_layers

        self.correct_layer_norm = correct_layer_norm

        if ln_before_act:
            self.ln_i2c = LayerNorm(hidden_size) # LN for input->cell
            #self.ln_h2c = torch.nn.LayerNorm(4 * cell_size) # LN for hidden->cell

    def forward(self,  # pylint: disable=arguments-differ
                inputs: torch.Tensor,
                mask: torch.LongTensor,
                hidden = None,
                return_hidden = True) -> torch.Tensor:

        assert(return_hidden)

        batch_size, total_sequence_length = mask.size()

        if self.input_projection:
            inputs = self.input_projection(inputs)
        original_inputs = inputs
        if self.layer_norm:
            if self.ln_before_act:
                inputs = self.ln_i2c(inputs)

        stacked_sequence_output, final_states, restoration_indices, hidden = \
            self.sort_and_run_forward(self._lstm_forward, inputs, mask, reset_hidden_state = self.reset_hidden_state, hidden_states = hidden)
        num_layers, num_valid, returned_timesteps, encoder_dim = stacked_sequence_output.size()
        # Add back invalid rows which were removed in the call to sort_and_run_forward.
        # Rows with all padding was removed
        if num_valid < batch_size:
            zeros = stacked_sequence_output.new_zeros(num_layers,
                                                      batch_size - num_valid,
                                                      returned_timesteps,
                                                      encoder_dim)
            stacked_sequence_output = torch.cat([stacked_sequence_output, zeros], 1)

            # The states also need to have invalid rows added back.
            new_states = []
            for state in final_states:
                state_dim = state.size(-1)
                zeros = state.new_zeros(num_layers, batch_size - num_valid, state_dim)
                new_states.append(torch.cat([state, zeros], 1))
            final_states = new_states

        # It's possible to need to pass sequences which are padded to longer than the
        # max length of the sequence to a Seq2StackEncoder. However, packing and unpacking
        # the sequences mean that the returned tensor won't include these dimensions, because
        # the RNN did not need to process them. We add them back on in the form of zeros here.
        sequence_length_difference = total_sequence_length - returned_timesteps
        if sequence_length_difference > 0:
            zeros = stacked_sequence_output.new_zeros(num_layers,
                                                      batch_size,
                                                      sequence_length_difference,
                                                      stacked_sequence_output[0].size(-1))
            stacked_sequence_output = torch.cat([stacked_sequence_output, zeros], 2)

        hidden = self._update_states(final_states, restoration_indices, hidden)

        # Restore the original indices and return the sequence.
        # Has shape (num_layers, batch_size, sequence_length, hidden_size)
        returned = stacked_sequence_output.index_select(1, restoration_indices)[-1]
        return returned, hidden

    def _lstm_forward(self,
                      inputs: PackedSequence,
                      initial_state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None) -> \
            Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        if initial_state is None:
            hidden_states: List[Optional[Tuple[torch.Tensor,
                                               torch.Tensor]]] = [None] * len(self.forward_layers)
        elif initial_state[0].size()[0] != len(self.forward_layers):
            raise ConfigurationError("Initial states were passed to forward() but the number of "
                                     "initial states does not match the number of layers.")
        else:
            hidden_states = list(zip(initial_state[0].split(1, 0), initial_state[1].split(1, 0)))

        inputs, batch_lengths = pad_packed_sequence(inputs, batch_first=True)
        forward_output_sequence = inputs

        final_states = []
        sequence_outputs = []
        for layer_index, state in enumerate(hidden_states):
            forward_layer = getattr(self, 'forward_layer_{}'.format(layer_index))

            forward_cache = forward_output_sequence

            forward_state = state

            forward_output_sequence, forward_state = forward_layer(forward_output_sequence,
                                                                   batch_lengths,
                                                                   forward_state)
            if layer_index != 0:
                forward_output_sequence += forward_cache # This is the residual connection!

            sequence_outputs.append(forward_output_sequence)
            # Append the state tuples in a list, so that we can return
            # the final states for all the layers.
            final_states.append(forward_state)

        stacked_sequence_outputs: torch.FloatTensor = torch.stack(sequence_outputs)
        # Stack the hidden state and memory for each layer into 2 tensors of shape
        # (num_layers, batch_size, hidden_size) and (num_layers, batch_size, cell_size)
        # respectively.
        final_hidden_states, final_memory_states = zip(*final_states)
        final_state_tuple: Tuple[torch.FloatTensor,
                                 torch.FloatTensor] = (torch.cat(final_hidden_states, 0),
                                                       torch.cat(final_memory_states, 0))
        return stacked_sequence_outputs, final_state_tuple
