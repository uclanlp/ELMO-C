import torch
from torch import nn
from torch.autograd import Variable

from forget_mult import ForgetMult

class QRNNLayer(nn.Module):
    """Applies a single layer Quasi-Recurrent Neural Network (QRNN) to an input sequence.

    Args:
        input_size: The number of expected features in the input x.
        hidden_size: The number of features in the hidden state h. If not specified, the input size is used.
        save_prev_x: Whether to store previous inputs for use in future convolutional windows (i.e. for a continuing sequence such as in language modeling). If true, you must call reset to remove cached previous values of x. Default: False.
        window: Defines the size of the convolutional window (how many previous tokens to look when computing the QRNN values). Supports 1 and 2. Default: 1.
        zoneout: Whether to apply zoneout (i.e. failing to update elements in the hidden state) to the hidden state updates. Default: 0.
        output_gate: If True, performs QRNN-fo (applying an output gate to the output). If False, performs QRNN-f. Default: True.
        use_cuda: If True, uses fast custom CUDA kernel. If False, uses naive for loop. Default: True.

    Inputs: X, hidden
        - X (seq_len, batch, input_size): tensor containing the features of the input sequence.
        - hidden (batch, hidden_size): tensor containing the initial hidden state for the QRNN.

    Outputs: output, h_n
        - output (seq_len, batch, hidden_size): tensor containing the output of the QRNN for each timestep.
        - h_n (batch, hidden_size): tensor containing the hidden state for t=seq_len
    """

    def __init__(self, input_size, hidden_size=None, save_prev_x=False, zoneout=0, window=1, output_gate=True, use_cuda=True):
        super(QRNNLayer, self).__init__()

        assert window in [1, 2], "This QRNN implementation currently only handles convolutional window of size 1 or size 2"
        self.window = window
        self.input_size = input_size
        self.hidden_size = hidden_size if hidden_size else input_size
        self.zoneout = zoneout
        self.save_prev_x = save_prev_x
        self.prevX = None
        self.output_gate = output_gate
        self.use_cuda = use_cuda

        # One large matmul with concat is faster than N small matmuls and no concat
        self.linear = nn.Linear(self.window * self.input_size, 3 * self.hidden_size if self.output_gate else 2 * self.hidden_size)

    def reset(self):
        # If you are saving the previous value of x, you should call this when starting with a new state
        self.prevX = None

    def forward(self, X, hidden=None):
        seq_len, batch_size, _ = X.size()

        source = None
        if self.window == 1:
            source = X
        elif self.window == 2:
            # Construct the x_{t-1} tensor with optional x_{-1}, otherwise a zeroed out value for x_{-1}
            Xm1 = []
            Xm1.append(self.prevX if self.prevX is not None else X[:1, :, :] * 0)
            # Note: in case of len(X) == 1, X[:-1, :, :] results in slicing of empty tensor == bad
            if len(X) > 1:
                Xm1.append(X[:-1, :, :])
            Xm1 = torch.cat(Xm1, 0)
            # Convert two (seq_len, batch_size, hidden) tensors to (seq_len, batch_size, 2 * hidden)
            source = torch.cat([X, Xm1], 2)

        # Matrix multiplication for the three outputs: Z, F, O
        Y = self.linear(source)
        # Convert the tensor back to (batch, seq_len, len([Z, F, O]) * hidden_size)
        if self.output_gate:
            Y = Y.view(seq_len, batch_size, 3 * self.hidden_size)
            Z, F, O = Y.chunk(3, dim=2)
        else:
            Y = Y.view(seq_len, batch_size, 2 * self.hidden_size)
            Z, F = Y.chunk(2, dim=2)
        ###
        Z = torch.nn.functional.tanh(Z)
        F = torch.nn.functional.sigmoid(F)

        # If zoneout is specified, we perform dropout on the forget gates in F
        # If an element of F is zero, that means the corresponding neuron keeps the old value
        if self.zoneout:
            if self.training:
                mask = Variable(F.data.new(*F.size()).bernoulli_(1 - self.zoneout), requires_grad=False)
                F = F * mask
            else:
                F *= 1 - self.zoneout

        # Ensure the memory is laid out as expected for the CUDA kernel
        # This is a null op if the tensor is already contiguous
        Z = Z.contiguous()
        F = F.contiguous()
        # The O gate doesn't need to be contiguous as it isn't used in the CUDA kernel

        # Forget Mult
        # For testing QRNN without ForgetMult CUDA kernel, C = Z * F may be useful
        C = ForgetMult()(F, Z, hidden, use_cuda=self.use_cuda)

        # Apply (potentially optional) output gate
        if self.output_gate:
            H = torch.nn.functional.sigmoid(O) * C
        else:
            H = C

        # In an optimal world we may want to backprop to x_{t-1} but ...
        if self.window > 1 and self.save_prev_x:
            self.prevX = Variable(X[-1:, :, :].data, requires_grad=False)

        return H, C[-1:, :, :]

class QRNNProjection(torch.nn.Module):
    """Applies a multiple layer Quasi-Recurrent Neural Network (QRNN) to an input sequence.

    Args:
        input_size: The number of expected features in the input x.
        hidden_size: The number of features in the hidden state h. If not specified, the input size is used.
        num_layers: The number of QRNN layers to produce.
        layers: List of preconstructed QRNN layers to use for the QRNN module (optional).
        save_prev_x: Whether to store previous inputs for use in future convolutional windows (i.e. for a continuing sequence such as in language modeling). If true, you must call reset to remove cached previous values of x. Default: False.
        window: Defines the size of the convolutional window (how many previous tokens to look when computing the QRNN values). Supports 1 and 2. Default: 1.
        zoneout: Whether to apply zoneout (i.e. failing to update elements in the hidden state) to the hidden state updates. Default: 0.
        output_gate: If True, performs QRNN-fo (applying an output gate to the output). If False, performs QRNN-f. Default: True.
        use_cuda: If True, uses fast custom CUDA kernel. If False, uses naive for loop. Default: True.

    Inputs: X, hidden
        - X (seq_len, batch, input_size): tensor containing the features of the input sequence.
        - hidden (layers, batch, hidden_size): tensor containing the initial hidden state for the QRNN.

    Outputs: output, h_n
        - output (seq_len, batch, hidden_size): tensor containing the output of the QRNN for each timestep.
        - h_n (layers, batch, hidden_size): tensor containing the hidden state for t=seq_len
    """

    def __init__(self, input_size, hidden_size,
                 projection_size,
                 layer_norm,
                 input_proj,
                 clip_value,
                 num_layers=1, bias=True, batch_first=False,
                 dropout=0, bidirectional=False, layers=None, **kwargs):
        assert bidirectional == False, 'Bidirectional QRNN is not yet supported'
        assert batch_first == False, 'Batch first mode is not yet supported'
        assert bias == True, 'Removing underlying bias is not yet supported'

        super(QRNNProjection, self).__init__()

        if projection_size:
            self.use_projection = True
        else:
            self.use_projection = False
        self.input_proj = input_proj
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = len(layers) if layers else num_layers
        self.bias = bias
        self.batch_first = batch_first
        self.dropout = dropout
        self.bidirectional = bidirectional

        if self.input_proj:
            self.input_projection = torch.nn.Linear(input_size, projection_size)
            input_size = projection_size
            
            if layer_norm:
                self.input_norm = torch.nn.LayerNorm(projection_size)

        if self.use_projection:
            self.layers = torch.nn.ModuleList(
                layers if layers else 
                [QRNNLayer(input_size if l == 0 else projection_size, hidden_size, **kwargs) for l in range(num_layers)]
                )
        else:
            self.layers = torch.nn.ModuleList(
                layers if layers else 
                [QRNNLayer(input_size if l == 0 else hidden_size, hidden_size, **kwargs) for l in range(num_layers)]
                )
        self.clip_value = clip_value
        self.projection_size = projection_size
        self.layer_norm = layer_norm
        self.projection_lst = nn.ModuleList()
        self.ln_lst = nn.ModuleList()
        for i in range(num_layers):
            if self.use_projection:
                self.projection_lst.append(nn.Linear(self.hidden_size, self.projection_size, bias = False))
            if layer_norm:
                if self.use_projection:
                    self.ln_lst.append(torch.nn.LayerNorm(self.projection_size))
                else:
                    self.ln_lst.append(torch.nn.LayerNorm(self.hidden_size))

    def reset(self):
        r'''If your convolutional window is greater than 1, you must reset at the beginning of each new sequence'''
        [layer.reset() for layer in self.layers]

    def forward(self, input, hidden=None):

        if self.input_proj:
            input = self.input_projection(input)
            if self.layer_norm:
                input = self.input_norm(input)
        next_hidden = []
        for i, layer in enumerate(self.layers):
            input, hn = layer(input, None if hidden is None else hidden[i])

            if self.clip_value != -1:
                input = torch.clamp(input,
                  -self.clip_value,
                  self.clip_value)

            if self.use_projection:
                input = self.projection_lst[i](input)

            if self.clip_value != -1:
                input = torch.clamp(input,
                  -self.clip_value,
                  self.clip_value)

            if self.layer_norm:
                input = self.ln_lst[i](input)

            next_hidden.append(hn)

            if self.dropout != 0 and i < len(self.layers) - 1:
                input = torch.nn.functional.dropout(input, p=self.dropout, training=self.training, inplace=False)

        next_hidden = torch.cat(next_hidden, 0).view(self.num_layers, *next_hidden[0].size()[-2:])

        return input, next_hidden


    def get_all_layer_output(self, input, hidden=None):
        if self.input_proj:
            input = self.input_projection(input)
            if self.layer_norm:
                input = self.input_norm(input)
        next_hidden = []
        all_layers_output = [input]
        for i, layer in enumerate(self.layers):
            input, hn = layer(input, None if hidden is None else hidden[i])

            if self.clip_value != -1:
                input = torch.clamp(input,
                  -self.clip_value,
                  self.clip_value)

            if self.use_projection:
                input = self.projection_lst[i](input)

            if self.clip_value != -1:
                input = torch.clamp(input,
                  -self.clip_value,
                  self.clip_value)

            if self.layer_norm:
                input = self.ln_lst[i](input)

            all_layers_output.append(input)
            next_hidden.append(hn)

        next_hidden = torch.cat(next_hidden, 0).view(self.num_layers, *next_hidden[0].size()[-2:])
        all_layers_output = torch.stack(all_layers_output, 0)
        return all_layers_output, next_hidden
