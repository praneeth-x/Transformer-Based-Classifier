import tensorflow as tf
from tensorflow import keras
import attention
class feed_forward(keras.Model):
	def __init__(self,max_seq_len,d_model):
		super(feed_forward).__init__()
		self.d_model=d_model
		self.max_seq_len=max_seq_len
		self.weights_encoder=self.add_weight(
			name='final_weights',
			shape=(self.d_model,self.d_model),
			initializer='random_normal',
			trainable=True
			)
		self.bias=self.add_weight(
			name='bias',
			shape=(self.max_seq_len,self.d_model),
			initializer='random_normal',
			trainable=True
			)
	def call(self,data):
		return tf.matmul(data,self.weights_encoder)+self.bias

class encoder(keras.Model):
	def __init__(self,number_heads,max_seq_len,d_model,expected_len):
		super(encoder).__init__()
		self.number_heads=number_heads
		self.max_seq_len=max_seq_len
		self.d_model=d_model
		self.expected_len=expected_len
		self.mha=attention.MultiHeadAttention(number_heads=number_heads,max_seq_len=max_seq_len,d_model=d_model,expected_len=expected_len)
		self.normalise1=tf.keras.layers.LayerNormalization(epsilon=1e-6)
		self.normalise2=tf.keras.layers.LayerNormalization(epsilon=1e-6)
		self.drop1=tf.keras.layers.Dropout(0.1)
		self.forward=feed_forward(max_seq_len,d_model)
	
	def call(self,data):
		attention_output=self.mha(data)
		attention_output=self.drop1(attention_output,training=training)
		first_out=self.normalise1(attention_output+data)
		first_out1=self.forward(first_out)
		out=self.normalise2(first_out1+first_out)
		return out


