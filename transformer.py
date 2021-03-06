import numpy as np
import tensorflow as tf
from tensorflow import keras
import pandas as pd
import Classification 
import encoder
from tensorflow.keras.layers import InputLayer

def positional_encoding(max_seq_len,d_model):
	positional=np.zeros(shape=(max_seq_len,d_model), dtype='float32')
	for i in range(max_seq_len):
		for j in range(0,d_model,2):
		   positional[i][j]=np.sin(i/(10000**((2*j)/d_model)))
		   if(i==max_seq_len-1):
		   	 break
		   positional[i+1][j]=np.sin(i/(10000**((2*j+1)/d_model)))
	return tf.constant(positional,shape=(max_seq_len,d_model),dtype='float32')

class Transformer(keras.Model):
	def __init__(self,max_seq_len,d_model,num_encoders,num_heads,num_classes,expected_length):
		super(Transformer,self).__init__()
		self.max_seq_len=max_seq_len
		self.d_model=d_model
		self.num_encoders=num_encoders
		self.num_heads=num_heads
		self.num_classes=num_classes
		self.expected_len=expected_length # can change the expected len of the query key and value vectors
		self.num_features=max_seq_len*d_model
		self.classify=Classification.classification_model(num_layers=4,num_classes=self.num_classes,feature_no=self.num_features)
		self.encoder_list=[]
		for i in range(self.num_encoders):
			self.encoder_list.append(encoder.encoder(self.num_heads,self.max_seq_len,self.d_model,self.expected_len))
	def call(self,dataframe):
		postional=positional_encoding(self.max_seq_len,self.d_model)
		data=tf.reshape(dataframe,shape=(self.max_seq_len,self.d_model))
		self.positional_data=data+postional
		for enc in self.encoder_list:
			self.positional_data=enc(data=self.positional_data)
		final_data=tf.reshape(self.positional_data,[1,-1])
		output=self.classify(final_data)
		return output

