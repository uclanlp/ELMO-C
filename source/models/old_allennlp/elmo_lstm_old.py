"""
A stacked bidirectional LSTM with skip connections between layers.
"""
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

from .encoder_base_old import _EncoderBase
from .lstm_with_projection_old import LstmCellWithProjection

"""
An LSTM with Recurrent Dropout, a hidden_state which is projected and
clipping on both the hidden state and the memory state of the LSTM.
"""

from typing import Optional, Tuple, List

import torch

from allennlp.nn.util import get_dropout_mask
from allennlp.nn.initializers import block_orthogonal

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

class ElmoLstm(_EncoderBase):
    """
    A stacked, bidirectional LSTM which uses
    :class:`~allennlp.modules.lstm_cell_with_projection.LstmCellWithProjection`'s
    with highway layers between the inputs to layers.
    The inputs to the forward and backward directions are independent - forward and backward
    states are not concatenated between layers.

    Additionally, this LSTM maintains its `own` state, which is updated every time
    ``forward`` is called. It is dynamically resized for different batch sizes and is
    designed for use with non-continuous inputs (i.e inputs which aren't formatted as a stream,
    such as text used for a language modelling task, which is how stateful RNNs are typically used).
    This is non-standard, but can be thought of as having an "end of sentence" state, which is
    carried across different sentences.

    Parameters
    ----------
    input_size : ``int``, required
        The dimension of the inputs to the LSTM.
    hidden_size : ``int``, required
        The dimension of the outputs of the LSTM.
    cell_size : ``int``, required.
        The dimension of the memory cell of the
        :class:`~allennlp.modules.lstm_cell_with_projection.LstmCellWithProjection`.
    num_layers : ``int``, required
        The number of bidirectional LSTMs to use.
    requires_grad: ``bool``, optional
        If True, compute gradient of ELMo parameters for fine tuning.
    recurrent_dropout_probability: ``float``, optional (default = 0.0)
        The dropout probability to be used in a dropout scheme as stated in
        `A Theoretically Grounded Application of Dropout in Recurrent Neural Networks
        <https://arxiv.org/abs/1512.05287>`_ .
    state_projection_clip_value: ``float``, optional, (default = None)
        The magnitude with which to clip the hidden_state after projecting it.
    memory_cell_clip_value: ``float``, optional, (default = None)
        The magnitude with which to clip the memory cell.
    """
    def __init__(self,
                 input_size: int,
                 hidden_size: int,
                 cell_size: int,
                 num_layers: int,
                 requires_grad: bool = False,
                 recurrent_dropout_probability: float = 0.0,
                 memory_cell_clip_value: Optional[float] = None,
                 state_projection_clip_value: Optional[float] = None,
                 reset_hidden_state = False) -> None:
        super(ElmoLstm, self).__init__(stateful=True)

        # Required to be wrapped with a :class:`PytorchSeq2SeqWrapper`.
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.cell_size = cell_size
        self.requires_grad = requires_grad

        forward_layers = []
        backward_layers = []

        lstm_input_size = input_size
        go_forward = True
        for layer_index in range(num_layers):
            forward_layer = LstmCellWithProjection(lstm_input_size,
                                                   hidden_size,
                                                   cell_size,
                                                   go_forward,
                                                   recurrent_dropout_probability,
                                                   memory_cell_clip_value,
                                                   state_projection_clip_value)
            backward_layer = LstmCellWithProjection(lstm_input_size,
                                                    hidden_size,
                                                    cell_size,
                                                    not go_forward,
                                                    recurrent_dropout_probability,
                                                    memory_cell_clip_value,
                                                    state_projection_clip_value)
            lstm_input_size = hidden_size

            self.add_module('forward_layer_{}'.format(layer_index), forward_layer)
            self.add_module('backward_layer_{}'.format(layer_index), backward_layer)
            forward_layers.append(forward_layer)
            backward_layers.append(backward_layer)
        self.forward_layers = forward_layers
        self.backward_layers = backward_layers
        self.reset_hidden_state = reset_hidden_state


    def forward(self,  # pylint: disable=arguments-differ
                inputs: torch.Tensor,
                mask: torch.LongTensor) -> torch.Tensor:
        """
        Parameters
        ----------
        inputs : ``torch.Tensor``, required.
            A Tensor of shape ``(batch_size, sequence_length, hidden_size)``.
        mask : ``torch.LongTensor``, required.
            A binary mask of shape ``(batch_size, sequence_length)`` representing the
            non-padded elements in each sequence in the batch.

        Returns
        -------
        A ``torch.Tensor`` of shape (num_layers, batch_size, sequence_length, hidden_size),
        where the num_layers dimension represents the LSTM output from that layer.
        """
        batch_size, total_sequence_length = mask.size()

        stacked_sequence_output, final_states, restoration_indices = \
            self.sort_and_run_forward(self._lstm_forward, inputs, mask, reset_hidden_state = self.reset_hidden_state)

        num_layers, num_valid, returned_timesteps, encoder_dim = stacked_sequence_output.size()
        # Add back invalid rows which were removed in the call to sort_and_run_forward.
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

        self._update_states(final_states, restoration_indices)

        # Restore the original indices and return the sequence.
        # Has shape (num_layers, batch_size, sequence_length, hidden_size)
        returned = stacked_sequence_output.index_select(1, restoration_indices)
        inputs = torch.cat((inputs, inputs), dim = -1)
        returned = torch.cat((inputs.unsqueeze(0), returned), dim = 0)
        return returned

    def _lstm_forward(self,
                      inputs: PackedSequence,
                      initial_state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None) -> \
            Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Parameters
        ----------
        inputs : ``PackedSequence``, required.
            A batch first ``PackedSequence`` to run the stacked LSTM over.
        initial_state : ``Tuple[torch.Tensor, torch.Tensor]``, optional, (default = None)
            A tuple (state, memory) representing the initial hidden state and memory
            of the LSTM, with shape (num_layers, batch_size, 2 * hidden_size) and
            (num_layers, batch_size, 2 * cell_size) respectively.

        Returns
        -------
        output_sequence : ``torch.FloatTensor``
            The encoded sequence of shape (num_layers, batch_size, sequence_length, hidden_size)
        final_states: ``Tuple[torch.FloatTensor, torch.FloatTensor]``
            The per-layer final (state, memory) states of the LSTM, with shape
            (num_layers, batch_size, 2 * hidden_size) and  (num_layers, batch_size, 2 * cell_size)
            respectively. The last dimension is duplicated because it contains the state/memory
            for both the forward and backward layers.
        """
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
        backward_output_sequence = inputs

        final_states = []
        sequence_outputs = []
        for layer_index, state in enumerate(hidden_states):
            forward_layer = getattr(self, 'forward_layer_{}'.format(layer_index))
            backward_layer = getattr(self, 'backward_layer_{}'.format(layer_index))

            forward_cache = forward_output_sequence
            backward_cache = backward_output_sequence

            if state is not None:
                forward_hidden_state, backward_hidden_state = state[0].split(self.hidden_size, 2)
                forward_memory_state, backward_memory_state = state[1].split(self.cell_size, 2)
                forward_state = (forward_hidden_state, forward_memory_state)
                backward_state = (backward_hidden_state, backward_memory_state)
            else:
                forward_state = None
                backward_state = None

            forward_output_sequence, forward_state = forward_layer(forward_output_sequence,
                                                                   batch_lengths,
                                                                   forward_state)
            backward_output_sequence, backward_state = backward_layer(backward_output_sequence,
                                                                      batch_lengths,
                                                                      backward_state)
            # Skip connections, just adding the input to the output.
            if layer_index != 0:
                forward_output_sequence += forward_cache
                backward_output_sequence += backward_cache

            sequence_outputs.append(torch.cat([forward_output_sequence,
                                               backward_output_sequence], -1))
            # Append the state tuples in a list, so that we can return
            # the final states for all the layers.
            final_states.append((torch.cat([forward_state[0], backward_state[0]], -1),
                                 torch.cat([forward_state[1], backward_state[1]], -1)))

        stacked_sequence_outputs: torch.FloatTensor = torch.stack(sequence_outputs)
        # Stack the hidden state and memory for each layer into 2 tensors of shape
        # (num_layers, batch_size, hidden_size) and (num_layers, batch_size, cell_size)
        # respectively.
        final_hidden_states, final_memory_states = zip(*final_states)
        final_state_tuple: Tuple[torch.FloatTensor,
                                 torch.FloatTensor] = (torch.cat(final_hidden_states, 0),
                                                       torch.cat(final_memory_states, 0))
        return stacked_sequence_outputs, final_state_tuple

    def load_weights(self, weight_file: str) -> None:
        """
        Load the pre-trained weights from the file.
        """
        requires_grad = self.requires_grad

        with h5py.File(cached_path(weight_file), 'r') as fin:
            for i_layer, lstms in enumerate(
                    zip(self.forward_layers, self.backward_layers)
            ):
                for j_direction, lstm in enumerate(lstms):
                    # lstm is an instance of LSTMCellWithProjection
                    cell_size = lstm.cell_size

                    dataset = fin['RNN_%s' % j_direction]['RNN']['MultiRNNCell']['Cell%s' % i_layer
                                                                                ]['LSTMCell']

                    # tensorflow packs together both W and U matrices into one matrix,
                    # but pytorch maintains individual matrices.  In addition, tensorflow
                    # packs the gates as input, memory, forget, output but pytorch
                    # uses input, forget, memory, output.  So we need to modify the weights.
                    tf_weights = numpy.transpose(dataset['W_0'][...])
                    torch_weights = tf_weights.copy()

                    # split the W from U matrices
                    input_size = lstm.input_size
                    input_weights = torch_weights[:, :input_size]
                    recurrent_weights = torch_weights[:, input_size:]
                    tf_input_weights = tf_weights[:, :input_size]
                    tf_recurrent_weights = tf_weights[:, input_size:]

                    # handle the different gate order convention
                    for torch_w, tf_w in [[input_weights, tf_input_weights],
                                          [recurrent_weights, tf_recurrent_weights]]:
                        torch_w[(1 * cell_size):(2 * cell_size), :] = tf_w[(2 * cell_size):(3 * cell_size), :]
                        torch_w[(2 * cell_size):(3 * cell_size), :] = tf_w[(1 * cell_size):(2 * cell_size), :]

                    lstm.input_linearity.weight.data.copy_(torch.FloatTensor(input_weights))
                    lstm.state_linearity.weight.data.copy_(torch.FloatTensor(recurrent_weights))
                    lstm.input_linearity.weight.requires_grad = requires_grad
                    lstm.state_linearity.weight.requires_grad = requires_grad

                    # the bias weights
                    tf_bias = dataset['B'][...]
                    # tensorflow adds 1.0 to forget gate bias instead of modifying the
                    # parameters...
                    tf_bias[(2 * cell_size):(3 * cell_size)] += 1
                    torch_bias = tf_bias.copy()
                    torch_bias[(1 * cell_size):(2 * cell_size)
                              ] = tf_bias[(2 * cell_size):(3 * cell_size)]
                    torch_bias[(2 * cell_size):(3 * cell_size)
                              ] = tf_bias[(1 * cell_size):(2 * cell_size)]
                    lstm.state_linearity.bias.data.copy_(torch.FloatTensor(torch_bias))
                    lstm.state_linearity.bias.requires_grad = requires_grad

                    # the projection weights
                    proj_weights = numpy.transpose(dataset['W_P_0'][...])
                    lstm.state_projection.weight.data.copy_(torch.FloatTensor(proj_weights))
                    lstm.state_projection.weight.requires_grad = requires_grad

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
                 output_pre_norm = False) -> None:
        super(ElmoLstmUni, self).__init__(stateful=True)

        self.reset_hidden_state = reset_hidden_state
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
                                                   correct_layer_norm = correct_layer_norm)
            lstm_input_size = hidden_size

            self.add_module('forward_layer_{}'.format(layer_index), forward_layer)
            forward_layers.append(forward_layer)
        self.forward_layers = forward_layers

        self.correct_layer_norm = correct_layer_norm

        if ln_before_act:
            self.ln_i2c = LayerNorm(hidden_size) # LN for input->cell
            #self.ln_h2c = torch.nn.LayerNorm(4 * cell_size) # LN for hidden->cell
        self.output_pre_norm = output_pre_norm

    def forward(self,  # pylint: disable=arguments-differ
                inputs: torch.Tensor,
                mask: torch.LongTensor) -> torch.Tensor:
        """
        Parameters
        ----------
        inputs : ``torch.Tensor``, required.
            A Tensor of shape ``(batch_size, sequence_length, hidden_size)``.
        mask : ``torch.LongTensor``, required.
            A binary mask of shape ``(batch_size, sequence_length)`` representing the
            non-padded elements in each sequence in the batch.

        Returns
        -------
        A ``torch.Tensor`` of shape (batch_size, sequence_length, hidden_size),
        where the num_layers dimension represents the LSTM output from that layer.
        """
        batch_size, total_sequence_length = mask.size()

        if self.input_projection:
            inputs = self.input_projection(inputs)

        if self.layer_norm:
            if self.ln_before_act:
                inputs = self.ln_i2c(inputs)

        stacked_sequence_output, final_states, restoration_indices = \
            self.sort_and_run_forward(self._lstm_forward, inputs, mask, reset_hidden_state = self.reset_hidden_state)
        num_layers, num_valid, returned_timesteps, encoder_dim = stacked_sequence_output.size()
        # Add back invalid rows which were removed in the call to sort_and_run_forward.
        # Rows with all padding was removed
        if num_valid < batch_size:
            
            zeros = stacked_sequence_output.data.new(num_layers,
                                                 batch_size - num_valid,
                                                 returned_timesteps,
                                                 encoder_dim).fill_(0)
            zeros = Variable(zeros)

            stacked_sequence_output = torch.cat([stacked_sequence_output, zeros], 1)

            # The states also need to have invalid rows added back.
            new_states = []
            for state in final_states:
                state_dim = state.size(-1)
                zeros = state.data.new(num_layers, batch_size - num_valid, state_dim).fill_(0)
                zeros = Variable(zeros)
                new_states.append(torch.cat([state, zeros], 1))
            final_states = new_states

        # It's possible to need to pass sequences which are padded to longer than the
        # max length of the sequence to a Seq2StackEncoder. However, packing and unpacking
        # the sequences mean that the returned tensor won't include these dimensions, because
        # the RNN did not need to process them. We add them back on in the form of zeros here.
        sequence_length_difference = total_sequence_length - returned_timesteps
        if sequence_length_difference > 0:
            zeros = stacked_sequence_output.data.new(num_layers,
                                                     batch_size,
                                                     sequence_length_difference,
                                                     stacked_sequence_output[0].size(-1)).fill_(0)
            zeros = Variable(zeros)
            stacked_sequence_output = torch.cat([stacked_sequence_output, zeros], 2)

        self._update_states(final_states, restoration_indices)

        # Restore the original indices and return the sequence.
        # Has shape (num_layers, batch_size, sequence_length, hidden_size)
        returned = stacked_sequence_output.index_select(1, restoration_indices)[-1]
        return returned

    def get_all_layer_output(self, 
                inputs: torch.Tensor,
                mask: torch.LongTensor,
                no_carry) -> torch.Tensor:
        """
        Parameters
        ----------
        inputs : ``torch.Tensor``, required.
            A Tensor of shape ``(batch_size, sequence_length, hidden_size)``.
        mask : ``torch.LongTensor``, required.
            A binary mask of shape ``(batch_size, sequence_length)`` representing the
            non-padded elements in each sequence in the batch.

        Returns
        -------
        A ``torch.Tensor`` of shape (num_layers, batch_size, sequence_length, hidden_size),
        where the num_layers dimension represents the LSTM output from that layer.
        """
        batch_size, total_sequence_length = mask.size()

        
        if self.input_projection:
            inputs = self.input_projection(inputs)

        original_inputs = inputs

        if self.layer_norm:
            if self.ln_before_act:
                inputs = self.ln_i2c(inputs)

        stacked_sequence_output, final_states, restoration_indices = \
            self.sort_and_run_forward(self._lstm_forward, inputs, mask)
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

        self._update_states(final_states, restoration_indices)

        # Restore the original indices and return the sequence.
        # Has shape (num_layers, batch_size, sequence_length, hidden_size)

        returned = stacked_sequence_output.index_select(1, restoration_indices)
        if self.add_embedding_layer:
            if self.reset_hidden_state:
                returned = torch.cat((original_inputs.unsqueeze(0), returned), dim = 0)
            else:
                returned = torch.cat((inputs.unsqueeze(0), returned), dim = 0)
        return returned

    def _lstm_forward(self,
                      inputs: PackedSequence,
                      initial_state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None) -> \
            Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Parameters
        ----------
        inputs : ``PackedSequence``, required.
            A batch first ``PackedSequence`` to run the stacked LSTM over.
        initial_state : ``Tuple[torch.Tensor, torch.Tensor]``, optional, (default = None)
            A tuple (state, memory) representing the initial hidden state and memory
            of the LSTM, with shape (num_layers, batch_size, 2 * hidden_size) and
            (num_layers, batch_size, 2 * cell_size) respectively.

        Returns
        -------
        output_sequence : ``torch.FloatTensor``
            The encoded sequence of shape (num_layers, batch_size, sequence_length, hidden_size)
        final_states: ``Tuple[torch.FloatTensor, torch.FloatTensor]``
            The per-layer final (state, memory) states of the LSTM, with shape
            (num_layers, batch_size, 2 * hidden_size) and  (num_layers, batch_size, 2 * cell_size)
            respectively. The last dimension is duplicated because it contains the state/memory
            for both the forward and backward layers.
        """
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
