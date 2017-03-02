# LSTM-Python
A pure Python and Numpy implementation of an LSTM Network.

This is a pure numpy and python implementation of an LSTM network. 


Required dependiencies are:
 - Numpy
 - Pandas (only if importing DataFrames)
 - Matplotlib (for visualisation)
 
The execution file is not commented as of yet, however the LSTM class object file has comments to understand what's happening. This is loosely based on a Gist by Karpathy.

The LSTM cell includes "Peep-hole" connections.

Since the implementation does not use batch-training, the network's convergence is not optimal. This can be seen in the forward projections of a time-series, as the projections have some deviations. However, the hope is that it clearly shows a pure implementation of an LSTM cell and a network to gain a deeper understanding.
