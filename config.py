import torch

def globals(vocab_size: int, ncol: int, n_embd: int=23, n_head: int=8, n_layer: int=3, 
            dropout: int=0.2, out_space:int =3, device: torch.device='cpu'):
    global a
    global b
    global c
    global d
    global e
    global f
    global g
    global h

    a = vocab_size
    b = ncol
    c = n_embd
    d = n_head
    e = n_layer
    f = dropout
    g = out_space
    h = device

    return a, b, c, d, e, f, g, h
