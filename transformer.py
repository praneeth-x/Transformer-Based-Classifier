import numpy as np
import tensorflow as tf
from tensorflow import keras
import pandas as pd
import classification.py
import encoder.py
from tensorflow.keras.layers import InputLayer
class Transformer(keras.Model):
	def __init__(self,max_seq_len,d_model,num_encoders,num_heads,num_classes):
		super(Transformer,self).__init__()
		self.max_seq_len=max_seq_len
		self.d_model=d_model
		self.num_encoders=num_encoders
		self.num_heads=num_heads
		self.num_classes=num_classes
		self.expected_len=d_model # can change the expected len of the query key and value vectors
		self.num_features=max_seq_len*d_model
		self.classify=classification_model(num_layers=4,num_classes=7,feature_no=num_features)
		self.encoder_list=[]
		self.Input_layer=InputLayer(input_shape=(max_seq_len,d_model),batch_size=3,name='input',dtype='float32')
		for i in range(num_encoders):
			encoder_list.append(encoder(num_heads,max_seq_len,d_model,expected_len))

	def positional_encoding(max_seq_len,d_model):
		positional=tf.zeros(shape=(max_seq_len,d_model) dtype='float32')
		for i in range(max_seq_len):
			for j in range(0,d_model,2):
			   positional[i,j]=tf.sin(i/(10000**((2*j)/d_model)))
			   positional[i+1,j]=tf.sin(i/(10000**((2*j+1)/d_model)))
		return positional

	def call(self,dataframe):
		self.postional=positional_encoding(self.max_seq_len,self.d_model)
		data=Input_layer(dataframe)
		self.positional_data=data+postional
		for enc in encoder_list:
			positional_data=enc(data=positional_data,num_heads)
		final_data=tf.reshape(positional_data,[-1])
		output=classification_model(final_data)
		return output

