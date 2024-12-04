import pandas as pd
import numpy as np
import openpyxl
import seaborn as sns
import matplotlib.pyplot as plt
import string
import torch
import torch.nn as nn
from torch.nn import functional as F
import random


def get_globals():
    from scripts.config import a, b, c, d, e, f, g, h

    vocab_size = a
    ncol = b
    n_embd = c
    n_head = d
    n_layer = e
    dropout = f
    out_space = g
    device = h

    return vocab_size, out_space, ncol, n_embd, n_head, n_layer, dropout, device


class Head(nn.Module):
    def __init__(self, head_size):
        super().__init__()
        vocab_size, out_space, ncol, n_embd, n_head, n_layer, dropout, device = get_globals()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(ncol, ncol)))

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)
        q = self.query(x)

        wei = q @ k.transpose(-2, -1) * k.shape[-1] ** -0.5
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)

        v = self.value(x)
        out = wei @ v

        return out


class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, head_size):
        super().__init__()
        vocab_size, out_space, ncol, n_embd, n_head, n_layer, dropout, device = get_globals()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(head_size * num_heads, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out



class FeedForward(nn.Module):
    def __init__(self, n_embd):
        super().__init__()
        vocab_size, out_space, ncol, n_embd, n_head, n_layer, dropout, device = get_globals()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 6 * n_embd),
            nn.ReLU(),
            nn.Linear(6 * n_embd, n_embd),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)



class Block(nn.Module):
    def __init__(self, n_embd, n_head):
        super().__init__()
        head_size = n_embd // n_head

        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedForward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        y = self.sa(x)
        x = self.ln1(x)
        y = self.ffwd(x)
        x = self.ln2(x + y)
        return x



class ElectionModel(nn.Module):
    def __init__(self):

        super().__init__()

        vocab_size, out_space, ncol, n_embd, n_head, n_layer, dropout, device = get_globals()

        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.blocks = nn.Sequential(*[Block(n_embd, n_head=n_head).to(device) for _ in range(n_layer)])

        self.ln_f = nn.LayerNorm(n_embd).to(device)
        self.lm_head = nn.Linear(n_embd, out_space).to(device)
        self.lm_head2 = nn.Linear(ncol, 1).to(device)

        self.ffwd = FeedForward(n_embd).to(device)
        self.ln1 = nn.LayerNorm(n_embd).to(device)
        self.ln2 = nn.LayerNorm(n_embd).to(device)

        self.apply(self.__init__weights)

    def __init__weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    # Define a forward function that returns element-wise predictions
    def forward_element(self, index, target=None):
        # B, T = index.shape
        
        # Embed the input index
        tok_emb = self.token_embedding_table(index)

        # print(tok_emb)
        
        # Pass through layers
        x = tok_emb
        x = self.ln1(x)
        # print("After layer 1:\n", x)
        x = self.ffwd(x)
        x = self.blocks(x)
        # print("After block:\n", x)
        x = self.ln_f(x)
        # print("After last layer:\n", x)

        # x = x.view(-1)
        # print(x)
        logits = self.lm_head(x)

        print("logits = ", logits)

        probs = F.softmax(logits, dim=-1)

        # print("logits = ", logits)

        if target is None:
            return probs
        else:
            # Calculate the cross-entropy loss using the probabilities
            # Reshape logits
            logits_flat = logits.view(-1, logits.size(-1))  # Flatten along the first two dimensions

            # Reshape targets
            targets_flat = target.view(-1)  # Flatten the targets

            # Calculate cross-entropy loss
            loss = F.cross_entropy(logits_flat, targets_flat)

            return probs, loss
        
    # Define a forward function that returns a single prediction
    def forward_single(self, index, target=None):
        
        # Embed the input index
        tok_emb = self.token_embedding_table(index)

        # print(tok_emb)
        
        # Pass through layers
        x = tok_emb
        x = self.ln1(x)

        x = self.blocks(x)

        x = self.ln_f(x)

        x = self.lm_head(x)

        # transform to get a single prediction
        x = torch.permute(x, (0, 2, 1))

        x = self.lm_head2(x)

        logits = torch.permute(x, (0, 2, 1)).to('cpu')

        # print("logits = \n", logits)
        # print(logits.size())

        probs = F.softmax(logits, dim=-1)


        if target is None:
            return probs
        else:
            # Calculate the cross-entropy loss using the probabilities
            # Reshape logits
            logits_flat = logits.view(-1, logits.size(-1))  # Flatten along the first two dimensions

            # Reshape targets
            output_tensor = torch.index_select(target, dim=-1, index=torch.tensor(0))
            output_tensor = torch.squeeze(output_tensor, dim=-1)
            targets_flat = output_tensor.view(-1)

            # Calculate cross-entropy loss
            loss = F.cross_entropy(logits_flat, targets_flat)

            return probs, loss

    def generate(self, index, type):
        if type == "element":
            probs = self.forward_element(index, None)
        else: 
            probs = self.forward_single(index, None)

        # print("probabilities = ", probs)

        bat_size, sequence_length, num_classes = probs.size()
        reshaped_probabilities = probs.view(-1, num_classes)

        # Sample from each probability distribution
        guesses = torch.multinomial(reshaped_probabilities, num_samples=1, replacement=True)

        # Reshape the guesses tensor back to match the original shape
        guesses = guesses.view(bat_size, sequence_length, 1)

        return guesses, reshaped_probabilities
    