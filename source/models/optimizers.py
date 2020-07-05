class NoamOpt_ADAM:
    "Optim wrapper that implements rate."
    def __init__(self, model_replica, start_decay, warmup, optimizer, end_decay, base_scale):
        self.optimizer = optimizer
        self._step = 0
        self.warmup = warmup
        self.start_decay = start_decay
        self.end_decay = end_decay
        self.n = model_replica
        self.base_scale = base_scale

    def step(self):
        "Update parameters and rate"
        self._step += 1
        rate = self.rate()
        for p in self.optimizer.param_groups:
            p['lr'] = rate
        self.optimizer.step()
        
    def rate(self, step = None):
        "Implement `lrate` above"
        if step is None:
            step = self._step
        if step < self.end_decay:
            if self.warmup != 0:
                return self.base_scale * min(
                    1 + step * (self.n - 1) / (self.warmup), 
                    self.n, 
                    self.n * ((2 * self.n) **  ( (self.start_decay - step) / (self.end_decay - self.start_decay)))
                    )
            else:
                return self.base_scale * min(
                    self.n,
                    self.n * ((2 * self.n) **  ( (self.start_decay - step) / (self.end_decay - self.start_decay)))
                    )
        else:
            return self.base_scale / 2

    def zero_grad(self):
        self.optimizer.zero_grad()

    def state_dict(self):
        return self.optimizer.state_dict()
    def load_state_dict(self, x):
        return self.optimizer.load_state_dict(x)

class NoamOpt:
    "Optim wrapper that implements rate."
    def __init__(self, model_size, factor, warmup, optimizer):
        self.optimizer = optimizer
        self._step = 0
        self.warmup = warmup
        self.factor = factor
        self.model_size = model_size
        self._rate = 0
        
    def step(self):
        "Update parameters and rate"
        self._step += 1
        rate = self.rate()
        for p in self.optimizer.param_groups:
            p['lr'] = rate
        self._rate = rate
        self.optimizer.step()
        
    def rate(self, step = None):
        "Implement `lrate` above"
        if step is None:
            step = self._step
        return self.factor * (self.model_size ** (-0.5) * min(step ** (-0.5), step * self.warmup ** (-1.5)))

    def zero_grad(self):
        self.optimizer.zero_grad()

    def state_dict(self):
        return self.optimizer.state_dict()

    def load_state_dict(self, x):
        return self.optimizer.load_state_dict(x)

class NoamOpt_SGD:
    "Optim wrapper that implements rate."
    def __init__(self, model_replica, start_decay, warmup, optimizer, end_decay):
        self.optimizer = optimizer
        self._step = 0
        self.warmup = warmup
        self.start_decay = start_decay
        self.end_decay = end_decay
        self.n = model_replica

    def step(self):
        "Update parameters and rate"
        self._step += 1
        rate = self.rate()
        for p in self.optimizer.param_groups:
            p['lr'] = rate
        self.optimizer.step()
        
    def rate(self, step = None):
        "Implement `lrate` above"
        if step is None:
            step = self._step
        if step < self.end_decay:
            return min(
                1 + step * (self.n - 1) / (self.n * self.warmup), 
                self.n, 
                self.n * ((2 * self.n) **  ( (self.start_decay - step) / (self.end_decay - self.start_decay)))
                )
        else:
            return 0.0056

    def zero_grad(self):
        self.optimizer.zero_grad()

    def state_dict(self):
        return self.optimizer.state_dict()

    def load_state_dict(self, x):
        return self.optimizer.load_state_dict(x)

import numpy as np
import torch
class AdaptiveGradientClipper:
    def __init__(self, window_size, parameters):
        self.window_size = window_size
        self.norms = np.ndarray(window_size)
        self.counter = 0
        self.parameters = parameters
        return

    def __call__(self, parameters, norm_type = 2):
        if isinstance(parameters, torch.Tensor):
            parameters = [parameters]
        parameters = list(filter(lambda p: p.grad is not None, parameters))
        norm_type = float(norm_type)
        
        total_norm = 0
        for p in parameters:
            param_norm = p.grad.data.norm(norm_type)
            total_norm += param_norm.item() ** norm_type
        total_norm = total_norm ** (1. / norm_type)

        if self.counter < self.window_size:
            standard_deviation = np.std(self.norms)
        else:
            standard_deviation = 1000000
        #print("Norm This step {}, STD {}, Mean {}".format(total_norm, standard_deviation, np.mean(self.norms)))
        if abs(total_norm - np.mean(self.norms)) > standard_deviation * 4:
            print("Skipped")
            return 0
        else:
            self.norms[self.counter % self.window_size] = total_norm
            self.counter += 1
            return 1

