import numpy as np

data = np.array([
	[1, 2, 3],
	[4, 5, 6],
	[7, 8, 9],
])

x = data[:,:-1]
y = data[:, -1]

print(y)