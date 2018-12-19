
import numpy as np
import matplotlib.pyplot as plt

dim = 2
num = 10000
train_data = np.zeros((num, dim+1)) # the last colum is label
test_data = np.zeros((num, dim+1))

# generate data
# class zero
mean = [0, 0]
cov = np.eye(2,2)*20 # diagonal covariance
x = np.random.multivariate_normal(mean, cov, num)
train_data[0:(num/2), 0:2] = x[0:(num/2), :]
test_data[0:(num/2), 0:2] = x[(num/2):, :]
# plt.plot(x[:,0], x[:,1], 'x')

# class one
mean = [30, 30]
cov = np.eye(2,2)*20  #[[50, 0], [0, 50]]  # diagonal covariance
x = np.random.multivariate_normal(mean, cov, num)
train_data[(num/2):, 0:2] = x[0:(num/2), :]
train_data[(num/2):, 2] = 1

test_data[(num/2):, 0:2] = x[(num/2):, :]
test_data[(num/2):, 2] = 1

np.save("./train_data.npy", train_data)
np.save("./test_data.npy", test_data)
#plt.plot(x, y, 'x')
# plt.plot(x[:,0], x[:,1], 'x')
#
# plt.axis('equal')
# plt.show()



