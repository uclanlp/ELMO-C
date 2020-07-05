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


class LstmCellWithProjection(torch.nn.Module):
    def __init__(self,
                 input_size: int,
                 hidden_size: int,
                 cell_size: int,
                 go_forward: bool = True,
                 recurrent_dropout_probability: float = 0.0,
                 memory_cell_clip_value: Optional[float] = None,
                 state_projection_clip_value: Optional[float] = None,
                 layer_norm = False,
                 correct_layer_norm = False,
                 output_pre_norm = False) -> None:
        super(LstmCellWithProjection, self).__init__()
        # Required to be wrapped with a :class:`PytorchSeq2SeqWrapper`.
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.cell_size = cell_size

        self.go_forward = go_forward
        self.state_projection_clip_value = state_projection_clip_value
        self.memory_cell_clip_value = memory_cell_clip_value
        self.recurrent_dropout_probability = recurrent_dropout_probability

        # We do the projections for all the gates all at once.
        self.input_linearity = torch.nn.Linear(input_size, 4 * cell_size, bias=False)
        self.state_linearity = torch.nn.Linear(hidden_size, 4 * cell_size, bias=True)

        # Additional projection matrix for making the hidden state smaller.
        self.state_projection = torch.nn.Linear(cell_size, hidden_size, bias=False)

        self.reset_parameters()

        self.layer_norm = layer_norm
        if layer_norm:
            self.ln_cell = LayerNorm(hidden_size)

        self.correct_layer_norm = correct_layer_norm
        if correct_layer_norm:
            '''self.ln_i2h = torch.nn.LayerNorm(cell_size * 4)
            self.ln_h2h = torch.nn.LayerNorm(cell_size * 4)'''
            self.ln_cell = LayerNorm(cell_size)

        self.output_pre_norm = output_pre_norm

    def reset_parameters(self):
        # Use sensible default initializations for parameters.
        block_orthogonal(self.input_linearity.weight.data, [self.cell_size, self.input_size])
        block_orthogonal(self.state_linearity.weight.data, [self.cell_size, self.hidden_size])

        self.state_linearity.bias.data.fill_(0.0)
        # Initialize forget gate biases to 1.0 as per An Empirical
        # Exploration of Recurrent Network Architectures, (Jozefowicz, 2015).
        self.state_linearity.bias.data[self.cell_size:2 * self.cell_size].fill_(1.0)

    def forward(self,  # pylint: disable=arguments-differ
                inputs: torch.FloatTensor,
                batch_lengths: List[int],
                initial_state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None):
        """
        Parameters
        ----------
        inputs : ``torch.FloatTensor``, required.
            A tensor of shape (batch_size, num_timesteps, input_size)
            to apply the LSTM over.
        batch_lengths : ``List[int]``, required.
            A list of length batch_size containing the lengths of the sequences in batch.
        initial_state : ``Tuple[torch.Tensor, torch.Tensor]``, optional, (default = None)
            A tuple (state, memory) representing the initial hidden state and memory
            of the LSTM. The ``state`` has shape (1, batch_size, hidden_size) and the
            ``memory`` has shape (1, batch_size, cell_size).

        Returns
        -------
        output_accumulator : ``torch.FloatTensor``
            The outputs of the LSTM for each timestep. A tensor of shape
            (batch_size, max_timesteps, hidden_size) where for a given batch
            element, all outputs past the sequence length for that batch are
            zero tensors.
        final_state : ``Tuple[``torch.FloatTensor, torch.FloatTensor]``
            A tuple (state, memory) representing the initial hidden state and memory
            of the LSTM. The ``state`` has shape (1, batch_size, hidden_size) and the
            ``memory`` has shape (1, batch_size, cell_size).
        """
        batch_size = inputs.size()[0]
        total_timesteps = inputs.size()[1]
        try:
            output_accumulator = inputs.new_zeros(batch_size, total_timesteps, self.hidden_size)
        except:
            output_accumulator = torch.autograd.Variable(inputs.data.new(batch_size,
                                                      total_timesteps,
                                                      self.hidden_size).fill_(0))

        if initial_state is None:
            try:
                full_batch_previous_memory = inputs.new_zeros(batch_size, self.cell_size)
                full_batch_previous_state = inputs.new_zeros(batch_size, self.hidden_size)
            except:
                full_batch_previous_memory = torch.autograd.Variable(inputs.data.new(batch_size,
                                                                  self.cell_size).fill_(0))
                full_batch_previous_state = torch.autograd.Variable(inputs.data.new(batch_size,
                                                                 self.hidden_size).fill_(0))
        else:
            full_batch_previous_state = initial_state[0].squeeze(0)
            full_batch_previous_memory = initial_state[1].squeeze(0)

        current_length_index = batch_size - 1 if self.go_forward else 0
        if self.recurrent_dropout_probability > 0.0 and self.training:
            dropout_mask = get_dropout_mask(self.recurrent_dropout_probability,
                                            full_batch_previous_state)
        else:
            dropout_mask = None

        for timestep in range(total_timesteps):
            # The index depends on which end we start.
            index = timestep if self.go_forward else total_timesteps - timestep - 1

            # What we are doing here is finding the index into the batch dimension
            # which we need to use for this timestep, because the sequences have
            # variable length, so once the index is greater than the length of this
            # particular batch sequence, we no longer need to do the computation for
            # this sequence. The key thing to recognise here is that the batch inputs
            # must be _ordered_ by length from longest (first in batch) to shortest
            # (last) so initially, we are going forwards with every sequence and as we
            # pass the index at which the shortest elements of the batch finish,
            # we stop picking them up for the computation.
            if self.go_forward:
                while batch_lengths[current_length_index] <= index:
                    current_length_index -= 1
            # If we're going backwards, we are _picking up_ more indices.
            else:
                # First conditional: Are we already at the maximum number of elements in the batch?
                # Second conditional: Does the next shortest sequence beyond the current batch
                # index require computation use this timestep?
                while current_length_index < (len(batch_lengths) - 1) and \
                                batch_lengths[current_length_index + 1] > index:
                    current_length_index += 1

            # Actually get the slices of the batch which we
            # need for the computation at this timestep.
            # shape (batch_size, cell_size)
            previous_memory = full_batch_previous_memory[0: current_length_index + 1].clone()
            # Shape (batch_size, hidden_size)
            previous_state = full_batch_previous_state[0: current_length_index + 1].clone()
            # Shape (batch_size, input_size)
            timestep_input = inputs[0: current_length_index + 1, index]

            # Do the projections for all the gates all at once.
            # Both have shape (batch_size, 4 * cell_size)
            projected_input = self.input_linearity(timestep_input)
            projected_state = self.state_linearity(previous_state)

            '''if self.correct_layer_norm:
                projected_input = self.ln_i2h(projected_input)
                projected_state = self.ln_h2h(projected_state)'''

            # Main LSTM equations using relevant chunks of the big linear
            # projections of the hidden state and inputs.
            input_gate = torch.sigmoid(projected_input[:, (0 * self.cell_size):(1 * self.cell_size)] +
                                       projected_state[:, (0 * self.cell_size):(1 * self.cell_size)])
            forget_gate = torch.sigmoid(projected_input[:, (1 * self.cell_size):(2 * self.cell_size)] +
                                        projected_state[:, (1 * self.cell_size):(2 * self.cell_size)])
            memory_init = torch.tanh(projected_input[:, (2 * self.cell_size):(3 * self.cell_size)] +
                                     projected_state[:, (2 * self.cell_size):(3 * self.cell_size)])
            output_gate = torch.sigmoid(projected_input[:, (3 * self.cell_size):(4 * self.cell_size)] +
                                        projected_state[:, (3 * self.cell_size):(4 * self.cell_size)])
            memory = input_gate * memory_init + forget_gate * previous_memory

            # Here is the non-standard part of this LSTM cell; first, we clip the
            # memory cell, then we project the output of the timestep to a smaller size
            # and again clip it.

            if self.memory_cell_clip_value:
                # pylint: disable=invalid-unary-operand-type
                memory = torch.clamp(memory, -self.memory_cell_clip_value, self.memory_cell_clip_value)

            if not self.correct_layer_norm:
                # shape (current_length_index, cell_size)
                pre_projection_timestep_output = output_gate * torch.tanh(memory)
            else:
                pre_projection_timestep_output = output_gate * torch.tanh(self.ln_cell(memory))

            # shape (current_length_index, hidden_size)
            timestep_output = self.state_projection(pre_projection_timestep_output)
            if self.state_projection_clip_value:
                # pylint: disable=invalid-unary-operand-type
                timestep_output = torch.clamp(timestep_output,
                                              -self.state_projection_clip_value,
                                              self.state_projection_clip_value)
            if self.output_pre_norm:
                output_accumulator[0:current_length_index + 1, index] = timestep_output
            if self.layer_norm:
                timestep_output = self.ln_cell(timestep_output)

            # Only do dropout if the dropout prob is > 0.0 and we are in training mode.
            if dropout_mask is not None:
                timestep_output = timestep_output * dropout_mask[0: current_length_index + 1]

            # We've been doing computation with less than the full batch, so here we create a new
            # variable for the the whole batch at this timestep and insert the result for the
            # relevant elements of the batch into it.
            full_batch_previous_memory = full_batch_previous_memory.clone()
            full_batch_previous_state = full_batch_previous_state.clone()
            full_batch_previous_memory[0:current_length_index + 1] = memory
            full_batch_previous_state[0:current_length_index + 1] = timestep_output
            output_accumulator[0:current_length_index + 1, index] = timestep_output
            if not self.output_pre_norm:
                output_accumulator[0:current_length_index + 1, index] = timestep_output

        # Mimic the pytorch API by returning state in the following shape:
        # (num_layers * num_directions, batch_size, ...). As this
        # LSTM cell cannot be stacked, the first dimension here is just 1.
        final_state = (full_batch_previous_state.unsqueeze(0),
                       full_batch_previous_memory.unsqueeze(0))

        return output_accumulator, final_state
