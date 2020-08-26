import paths_definitions as pth

from fastai.vision import *
from functools import partial

import matplotlib.pyplot as plt

def accuracy(input:Tensor, targs:Tensor)->Rank0Tensor:
    "Computes accuracy with `targs` when `input` is bs * n_classes."
    n = targs.shape[0]
    input = input.argmax(dim=-1).view(n,-1)
    targs = targs.view(n,-1)
    return (input==targs).float().mean()

def precision(input, targs, c):
    n = targs.shape[0]
    input = input.argmax(dim=-1).view(n, -1)
    targs = targs.view(n, -1)
    P = len([x for x in targs if x == c])
    TP = len([targs[i] for i in range(len(targs)) if targs[i]==c and input[i]==c])
    return TP/P

def recall(input, targs, c):
    n = targs.shape[0]
    input = input.argmax(dim=-1).view(n, -1)
    targs = targs.view(n, -1)
    R = len([x for x in input if x == c])
    TP = len([targs[i] for i in range(len(targs)) if targs[i] == c and input[i] == c])
    return TP/R

#TODO : recall and precision do not work


metrics = [accuracy]








