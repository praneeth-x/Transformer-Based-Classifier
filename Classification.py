from tensorflow import keras
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
import math
class classification_model(keras.Model):
	def __init__(self,num_layers,num_classes,feature_no):
		super(classification_model,self).__init__()
		self.num_classes=num_classes
		self.num_layers=num_layers
		self.num_neurons=math.floor(math.sqrt(feature_no*num_classes))
		self.layers_list=[]
		for i in range(num_layers):
			self.layers_list.append(Dense(self.num_neurons-i,activation='relu'))
			self.layers_list.append(Dropout(0.3))
		self.layers_list.append(Dense(num_classes,activation='softmax'))

	def call(self,data):
		out=self.layers_list[0](data)
		for i in range(1,len(self.layers_list)):
			out=self.layers_list[i](out)
		return out

