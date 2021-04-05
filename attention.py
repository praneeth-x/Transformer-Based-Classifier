import numpy as np
import tensorflow as tf
from tensorflow import keras
import math
#no negitive values are present in data therefore relu is not added
#can be added as per requirement demanded by data
class get_q(keras.Model):
	def __init__(self,d_model,expected_len):
		super(get_q,self).__init__()
		self.weight_q=self.add_weight(
			name='wq',
			shape=(d_model,expected_len),
			initializer='random_normal',
			trainable=True
			)
	def call(self,data):
		return tf.matmul(data,self.weight_q)

class get_k(keras.Model):
	def __init__(self,d_model,expeced_len):
		super(get_k,self).__init__()
		self.weight_k=self.add_weight(
			name='wk',
			shape=(d_model,expeced_len),
			initializer='random_normal',
			trainable=True
			)
	def call(self,data):
		return tf.matmul(data,self.weight_k)

class get_v(keras.Model):
	def __init__(self,d_model,expected_len):
		super(get_v,self).__init__()
		self.weight_v=self.add_weight(
			name='wv',
			shape=(d_model,expected_len),
			initializer='random_normal',
			trainable=True
			)
	def call(self,data):
		return tf.matmul(data,self.weight_v)

def scaled_dot_product_attention(q,k,v,d_model):
		temp=(tf.matmul(q,tf.transpose(k)))/math.sqrt(d_model)
		temp1=tf.nn.softmax(temp)
		return tf.matmul(temp1,v) 

class linearlayer(keras.Model):
	def __init__(self,max_seq_len,number_heads,d_model):
		super(linearlayer,self).__init__()
		self.w=self.add_weight(
			name='linear',
			shape=(max_seq_len,max_seq_len*number_heads),
			initializer='random_normal',
			trainable=True
			)
		self.b=self.add_weight(
			name='bias',
			shape=(max_seq_len,d_model),
			initializer='random_normal',
			trainable=True
			)
	def call(self,data):
		return tf.matmul(self.w,data)+self.b

class MultiHeadAttention(keras.Model):
	def __init__(self,number_heads,max_seq_len,d_model,expected_len):
		super(MultiHeadAttention,self).__init__()
		self.number_heads=number_heads
		self.d_model=d_model
		self.expected_len=expected_len
		self.z_list=[]
		self.max_seq_len=max_seq_len
		self.linear_layer=linearlayer(max_seq_len,number_heads,d_model)
		self.q_list=[]
		self.k_list=[]
		self.v_list=[]
		for i in range(self.number_heads):
			q1=get_q(self.d_model,self.expected_len)
			q=q1(data)
			k1=get_k(self.d_model,self.expected_len)
			k=k1(data)
			v1=get_k(self.d_model,self.expected_len)
			v=v1(data)
			self.q_list.append(q)
			self.k_list.append(k)
			self.v_list.append(v)

	def call(self,data):
		for i in range(self.number_heads):
			z=scaled_dot_product_attention(self.q_list[i],self.k_list[i],self.v_list[i],self.d_model)
			self.z_list.append(z)
		out=self.z_list[0]
		for i in range(1,len(self.z_list)):
			out=tf.concat([out,self.z_list[i]],axis=0)
		out=self.linear_layer(out)
		return out









