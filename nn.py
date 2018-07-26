import random
import math
import numpy as np
from matplotlib import pyplot as plt


data = [ [3, 1.5,   1], [2, 1,   0], [4, 1.5,   1], [3, 1,   0], [3.5, .5,   1], [2, .5,   0], [5.5, 1,   1], [1, 1,   0] ]

mystery_flower = [4.5, 1]

def sigmoid(x):
	return ((1)/(1+math.exp(-x)))
	
def sigmoid_p(x):
	return sigmoid(x) * (1-sigmoid(x))	
		
def train():

	w1 = np.random.randn()
	w2 = np.random.randn()
	b = np.random.randn()
	
	iterations = 10000
	learning_rate = 0.1
	costs = []
	w1_list = []
	w2_list = []
	b_list = []
	
	for i in range(iterations):
		#get a random point in data set
		ri = np.random.randint(len(data))
		point = data[ri]
		
		#get the value and sigmoid it
		z = point[0] * w1 + point[1] * w2 + b
		pred = sigmoid(z)
		
		target = point[2]
		
		cost = np.square(pred-target)
	
		
		if i % 100 == 0:
			c = 0
			for j in range(len(data)):
				p = data[j]
				p_pred = sigmoid(w1 * p[0] + w2 * p[1] + b)
				c += np.square(p_pred - p[2])
			costs.append(c)

		if i % 100 == 0:
			w1_list.append(w1)
			w2_list.append(w2)
			b_list.append(b)

			
		dcost_dpred = 2 * (pred-target)
		dpred_dz = sigmoid_p(z)
		
		dz_dw1 = point[0]
		dz_dw2 = point[1]
		dz_db = 1
		
		dcost_dz = dcost_dpred * dpred_dz
		
		dcost_dw1 = dcost_dz * dz_dw1
		dcost_dw2 = dcost_dz * dz_dw2
		dcost_db = dcost_dz * dz_db
		
		w1 = w1 - learning_rate * dcost_dw1
		w2 = w2 - learning_rate * dcost_dw2
		b = b - learning_rate * dcost_db
		
	plt.plot(costs)
	plt.plot(w1_list)
	plt.plot(w2_list)
	plt.plot(b_list)
	plt.legend(['costs', 'w1', 'w2', 'b'])
	plt.show()
		
	return w1, w2, b

def main():

	w1, w2, b = train()
	print(w1)
	print(w2)
	print(b)
	z = w1 * mystery_flower[0] + w2 * mystery_flower[1] + b
	print(sigmoid(z))
	print("closer to 0 = blue, closer to 1 = red")

	
main()
