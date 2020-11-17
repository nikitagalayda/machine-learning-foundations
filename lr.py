import random
import numpy as np

def prep_data(raw_data, x_0):
	data = []

	line_split = raw_data.split('\n')
	for line in line_split:
		item_split = line.split('\t')
		item_split.insert(0, x_0)
		# item_split = [float(i) for i in item_split]
		data.append(item_split)

	return data

def linear_regression(x):
	x_t = x.transpose()
	x_t_x = np.matmul(x_t, x)
	x_t_x_inverse = inv(x_t_x)
	w_lin = np.matmul(x_t_x_inverse, x_t)

	return w_lin

def squared_error(w_lin, x_in, y_in):
	y_hat = np.dot(w_lin, x_in)
	se = (y_hat - y_in)**2

	return se

def linear_regression_sgd(data, w_lin):
w_t = np.zeros(len(data))
n = 0.001

	e_in_sq = 9999999
	it = 0
	while e_in_sq > 1.01*w_lin:
		it += 1
		r = random.randint(0, len(data))
		x = data[r][:-1]
		y = data[r][-1]
		w_t = w_t + n*2*((np.dot(w_t, x)-y)*x)

	return it

f = open("hw3_train.txt", "r")
raw_data = f.read()
f.close()

data = prep_data(raw_data, 1.0)
print(data[0])