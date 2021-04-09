# Transformer-Based-Classifier
A sequence Classifier is implemented with the help of transformer attention mechanism

# How it Works?
The sequential data is classified using Transformer Attention mechanism. 
Each Point in sequence is a vector of size d_model total sequence. example in a word sequence(sentence) each word is encoded as 1-D vector of size d_model 
Total sequence length is of max_seq_len 

# The model Mainly Contains of 4 key modules

# 1. Positional encoding: 

Not only the selection of words in sequence the position of those words in sequence dictates the meaning example apple is good , is good apple both contain same words but the relative positions of words in the sequence changed the meaning of the sentence.

The information about the relative position of the words is added using a positional encoding formula as 
