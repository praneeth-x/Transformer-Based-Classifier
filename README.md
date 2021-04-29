# Transformer-Based-Classifier
A sequence Classifier is implemented with the help of transformer attention mechanism
# How to Use?
To avoid package collisions and creating new virtual environments setup.py file is not included in the package 
To use the Transformer directly you can download the entire package
keep the code you are writing tin the same folder in which the current package is downloaded inside your code file include the import as 
**from transformer import Transformer**
Now declare a Transformer object as the pass the required arguments. Transformer takes 6 arguments as shown below.
object=Transformer(
        max_seq_len, # pass the maximum length of the sequence example in case of text max number of words in sentance.
        d_model, # this is length of each vector in the sequence example in case to text it is embedded vector length. 
        num_encoders, # pass number of encoders you want the transformer to have
        num_heads, # this specifies number of attention heads inside each encoder.
        num_classes, # This is number of classes in your data which you want to classify 
        expected_length, # This argument specifies length of query key and value vectors that are computed inside the attention heads.
        )
 compile the object with your required optimizer and loss metric
 and object.fir(arguments) will train the model.
 object(input) will give the predicitons for the input passed //must be in the format and shape of the training input.
# How it Works?
The sequential data is classified using Transformer Attention mechanism. 
Each Point in sequence is a vector of size d_model total sequence. example in a word sequence(sentence) each word is encoded as 1-D vector of size d_model 
Total sequence length is of max_seq_len 
The input is going to be 2-D array with words stacked along axis=1(or can be seen as y-axis) each word is embedded as vector of size d_model that will extend on x-axis.(basically words are stacked on top of other until max_seq_len words) 

### The model Mainly Contains of 3 key modules

#### 1. Positional encoding: 

Not only the selection of words in sequence the position of those words in sequence dictates the meaning example "This apple is good" , "is good apple This" both contain same words but the relative positions of words in the sequence changed the meaning of the sentence.

The information about the relative position of the words is added using a positional encoding formula as 
    PE(𝑝𝑜𝑠,2𝑖)=𝑠𝑖𝑛(𝑝𝑜𝑠100002𝑖/𝑑𝑚𝑜𝑑𝑒𝑙),
    PE(𝑝𝑜𝑠,2𝑖+1)=𝑐𝑜𝑠(𝑝𝑜𝑠100002𝑖/𝑑𝑚𝑜𝑑𝑒𝑙).
    
where i is in the direction of the embeddings (if each word is encoded as a vector of size 512 then i runs for 0 to 511 for each word in the sequence)
pos is in the direction of sequence (ie position for first word =0 and position for second word is 1 and so on position of last word is lenngth of sequence-1)

#### 2. Encoder stack 

Encoder stack basically extracts the contextual information of the data using attention mechanism 

###### Attention Mechanism:

It is the key component of the encoder stack it is implemented using attention heads. Attention heads compute relavance of each point in sequence with every other point in the sequence and thus obtains a contextual information. this is done by computing query,key and value vectors which are obtained by multiplying the input with trainable weights.contextual information is obtained by computing the scaled dot product of the query,key and values given as

  attention(Q,K,V) = softmax (QKT √ dK ) V,

A single encoder consists of multiple attention heads all the outputs of the attention heads and concatenated and linearly scaled to get a final output which is passed as input to other consecutive encoders and final layer passes the contextual information to the classification layer.

#### 3. Classification Model.

the final step of the model is to classify the sequences from the contextual information passed this is implemented using a simple deep nueral network classifier to classify outputs based on the contextual information.
  
#### for further understanding visit https://towardsdatascience.com/transformers-141e32e69591
