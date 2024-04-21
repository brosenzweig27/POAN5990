import pandas as pd
import numpy as np

#############################################################################
### Import and inspect the desired datasets
#############################################################################

filt = pd.read_csv("network_project/filtered_data.csv")
# filt = pd.read_csv("imputed_filt.csv")
print(filt.head())
print(len(filt))

out = pd.read_csv("network_project/targets.csv")
# out = pd.read_csv("imputed_targets.csv")
print(out.head())
print(len(out))

# Get the number of columns in the input dataframe
ncol = len(filt.columns)

#############################################################################
### Use the vocab_dictionary function to create the dictionary mapping
#############################################################################

from network_project.myfunctions import vocab_dictionary

#############################################################################
### Retrieve the vocabulary for the input and output data
#############################################################################

in_vocab = vocab_dictionary(filt)
out_vocab = vocab_dictionary(out)

vocab_size = len(in_vocab)
print(vocab_size)

#############################################################################
### Create data subsets for 2022 and other year subsets we may want
#############################################################################

# Get all data before most recent year (2022)
pre_2022 = filt[filt['year'] != 'a2022'].copy()
els_2022 = filt[filt['year'] == 'a2022'].copy()

out_pre_2022 = out[:len(pre_2022)]
out_2022 = out[len(pre_2022):]

elnos_2022 = out_2022.index.tolist()
elnos_pre2022 = pre_2022.index.tolist()

# Get data from 2020
data_2020 = filt[filt['year'] == 'a2020'].copy()
out_2020 = pd.merge(out, data_2020, left_index=True, right_index=True).copy()

elnos_2020 = data_2020.index.tolist()

# Also subset some random year, take 1988

data_1988 = filt[filt['year'] == 'a1988'].copy()
data_not1988 = filt[filt['year'] != 'a1988'].copy()

out_1988 = pd.merge(out, data_1988, left_index=True, right_index=True).copy()
out_not1988 = pd.merge(out, data_not1988, left_index=True, right_index=True).copy()

elnos_1988 = data_1988.index.tolist()

#############################################################################
### Import encoding functions to get tensors to be used for training
#############################################################################

from network_project.myfunctions import encode_instance, encode_batch, encode_specific

# Get a specific encoding of the elections from 2022, as an example
x, y = encode_specific(df_in=els_2022, df_out=out_2022, els=elnos_2022, in_vocab=in_vocab, out_vocab=out_vocab)
print(x, y)

#############################################################################
### Set globals to be used by the model in the next step
#############################################################################
from network_project.config import globals
globals(vocab_size=vocab_size, out_space=3, ncol=ncol, n_embd=23, n_head=8, n_layer=3, dropout=0.2)

#############################################################################
### Import an instance of the model
#############################################################################
from network_project.model import ElectionModel
model = ElectionModel()

#############################################################################
### Train the model
#############################################################################

from network_project.myfunctions import train

train(model, epochs=5000, data_in=data_not1988, data_out=out_not1988, in_vocab=in_vocab, out_vocab=out_vocab, 
      n_batch=100, lr = 1e-5, eval_iters = 25)

#############################################################################
### Evaluate the model on reserved data
## In this case, assume we've subsetted out 2022 house elections
#############################################################################

from network_project.myfunctions import predict

predict(model, data_in=data_1988, data_out=out_1988, els=elnos_1988, in_vocab=in_vocab, out_vocab=out_vocab)