# This code is based on the GPTQ repository: https://github.com/IST-DASLab/gptq
# Original license: Apache 2.0

import time

import torch
import torch.nn as nn
import transformers

from lut_quant import *

DEBUG = False 

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

class GANQ:
    def __init__(self, layer, model_type):
        self.layer = layer
        self.dev = self.layer.weight.device
        self.model_type = model_type
        W = layer.weight.data.clone()
        if isinstance(self.layer, nn.Conv2d):
            W = W.flatten(1)
        if isinstance(self.layer, transformers.Conv1D):
            W = W.t()
        self.rows = W.shape[0]
        self.columns = W.shape[1]
        self.XXt = torch.zeros((self.columns, self.columns), device=self.dev)

    def add_batch(self, inp, out):
        if DEBUG:
            self.inp1 = inp
            self.out1 = out
        if len(inp.shape) == 2:
            inp = inp.unsqueeze(0)
        tmp = inp.shape[0]
        if isinstance(self.layer, nn.Linear) or isinstance(self.layer, transformers.Conv1D):
            if len(inp.shape) == 3:
                inp = inp.reshape((-1, inp.shape[-1]))
            inp = inp.t()

        inp = inp.float()
        self.XXt += inp @ inp.T

    def fasterquant(self, sparsity=0.0, bits=4, max_epoch=10, pre_process=True, full_rows=0):
        W = self.layer.weight.data.clone()
        if isinstance(self.layer, nn.Conv2d):
            W = W.flatten(1)
        if isinstance(self.layer, transformers.Conv1D):
            W = W.t()
        W = W.float()

        tick = time.time()
        quant = LUTQuant(bits=bits, W=W, XXt=self.XXt, max_epoch=max_epoch, sparsity=sparsity, model_type=self.model_type, pre_process=pre_process, full_rows=full_rows)
        W = quant.quantization()

        torch.cuda.synchronize()
        print('time %.2f' % (time.time() - tick))

        if isinstance(self.layer, transformers.Conv1D):
            W = W.t()
        self.layer.weight.data = W.reshape(self.layer.weight.shape).to(self.layer.weight.data.dtype)
        if DEBUG:
            print(torch.sum((self.layer(self.inp1) - self.out1) ** 2))

    def free(self):
        if DEBUG:
            self.inp1 = None
            self.out1 = None
        self.XXt = None
        torch.cuda.empty_cache()
