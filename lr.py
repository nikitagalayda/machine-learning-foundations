import random
import numpy as np

def prep_data(data, x_0):
    data_list = []
    line_data = data.split('\n')
    for d in line_data:
        if len(d) > 0:
            aux_list = d.split('\t')
            aux_list = [float(i) for i in aux_list]
            aux_list.insert(0, x_0)

            data_list.append(aux_list)
    
    return np.array(data_list)

def theta(s):
  return 1 / (1 + math.exp(-s))

def squared_error(w_lin, x_in, y_in):
	y_hat = np.dot(w_lin, x_in)
	se = (y_hat - y_in)**2

	return se

def average_squared_error(data, w_lin):
	avg_error = 0
	count = 0

	for d in data:
		x = d[:-1]
		y = d[-1]
		c_err = squared_error(w_lin, x, y)
		avg_error += c_err
		count += 1

	return avg_error/count

# def cross_entropy_error():

def linear_regression(data):
	x = data[:, :-1]
	y = data[:, -1]
	x_t = x.transpose()
	x_t_x = np.matmul(x_t, x)
	x_t_x_inverse = np.linalg.inv(x_t_x)
	tmp = np.matmul(x_t_x_inverse, x_t)
	w_lin = np.dot(tmp, y)

	return w_lin

def linear_regression_sgd(data, e_w_lin):
	w_t = np.zeros(11)
	n = 0.001

	e_in_sq = 9999999
	it = 0
	while e_in_sq > 1.01*e_w_lin:
		it += 1
		r = random.randint(0, len(data)-1)
		x = data[r][:-1]
		y = data[r][-1]
		w_t = w_t - n*2*((np.dot(w_t, x)-y)*x)
		e_in_sq = average_squared_error(data, w_t)

	return it

def logistic_regression_sgd(data, w_lin):
	w_t = np.zeros(len(data))
	n = 0.001

	e_in_sq = 9999999
	it = 0
	for i in range(0, 500):
		it += 1
		r = random.randint(0, len(data))
		x = data[r][:-1]
		y = data[r][-1]
		s = -(y*np.dot(w_t, x))
		tht = theta(s)
		err = n*tht*(y*x)
		w_t = w_t + err
		# e_in_sq = average_squared_error(data, w_t)

	return it

f = open("hw3_train.txt", "r")
raw_data = f.read()
f.close()

data = prep_data(raw_data, 1.0)

w_lin = linear_regression(data)
q14 = average_squared_error(data, w_lin)

q15 = 0
for i in range(0, 1000):
	q15 += linear_regression_sgd(data, q14)
q15 = q15/1000

