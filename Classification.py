from tensorflow import keras
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
import math
class classification_model(keras.Model):
	def __init__(self,num_layers,num_classes,feature_no):
		super(classification_model,self).__init__()
		self.data=data
		self.num_classes=num_classes
		self.num_layers=num_layers
		self.num_neurons=math.floor(math.sqrt(self.feature_no*self.num_classes))
		self.layers_list=[]
		for i in range(num_layers-1):
			self.layers_list.append(Dense(self.num_neurons-i,activation='relu'))
			self.layers_list.append(Dropout(0.3))
		self.layers_list.append(Dense(num_classes,activation='softmax'))

	def call(self,data):
		out=Dense(num_neurons,activation='relu')(data)
		for layer in layers_list:
			out=layer(out)
		return out

