
import sys
sys.path.append('/Users/lijianchen/anaconda/bin/python')
import numpy as np


class FCLayer(object):
    """define the attributes of a fully-connected layer"""

    def __init__(self, in_dim, out_dim, activation_func="ReLU", dropout_rate=0):
        self.in_dim = in_dim
        self.out_dim = out_dim
        self._act_fun = activation_func
        self._act_fun_grad = 0
        self._inner_product = 0
        self._x = 0
        self._y = 0
        self._dx = 0
        self._dw = 0
        self._lr = 0
        self._u = 0
        self._optimization = 0
        self._velocity = 0
        self._dropout_rate = dropout_rate
        self._dropout_map = 0
        self._dropout_weights = 0

        # weight initialization including biases
        self.weights = np.random.standard_normal(size=(out_dim, in_dim+1)) * 0.1
        return

    def relu(self, x):
        zero_arr = np.zeros_like(x)
        return np.maximum(zero_arr, x)

    def sigmoid(self, x):
        return 1.0 / (1 + np.exp(-x))

    def tanh(self, x):
        return np.tanh(x)

    def forward(self, x):
        self._x = np.append(x, np.ones((1, 1)), axis=0)  # append 1 for biases

        if self._dropout_rate > 0:
            # dropout
            temp = np.random.uniform(0, 1, size=self.weights.shape)
            self._dropout_map = np.zeros_like(self.weights)
            self._dropout_map[np.where(temp > self._dropout_rate)] = 1.0
            self._dropout_weights = self.weights * self._dropout_map
            self._inner_product = self._dropout_weights.dot(self._x)
        else:
            self._inner_product = self.weights.dot(self._x)  # no dropout

        if self._act_fun == "ReLU":
            self._y = self.relu(self._inner_product)
        elif self._act_fun == "sigmoid":
            self._y = self.sigmoid(self._inner_product)
        elif self._act_fun == "tanh":
            self._y = self.tanh(self._inner_product)
        elif self._act_fun == "output":
            self._y = self._inner_product

        return self._y

    def backward(self, dz):
        if self._act_fun == "ReLU":
            self._act_fun_grad = np.zeros_like(self._y)
            self._act_fun_grad[np.where(self._y > 0)] = 1.0
        elif self._act_fun == "sigmoid":
            self._act_fun_grad = self._y * (1 - self._y)  # gradient of sigmoid function
        elif self._act_fun == "tanh":
            self._act_fun_grad = np.ones_like(self._y) - self._y**2
            # print self._act_fun_grad
        elif self._act_fun == "output":
            self._act_fun_grad = np.ones_like(self._y)

        # compute the gradients with respect to layer input x
        upstream_grad = self._act_fun_grad * dz
        self._dw = upstream_grad.dot(self._x.T)

        if self._dropout_rate > 0:
            self._dx = self._dropout_weights.T.dot(upstream_grad)
        else:
            self._dx = self.weights.T.dot(upstream_grad)
        self._dx = self._dx[:self.in_dim, :]  # exclude the appended 1

        return self._dx

    def update_weights(self, optimization="SGD_momentum", lr=0.001, u=0.9):
        self._optimization = optimization
        self._lr = lr
        self._u = u

        if self._optimization == "SGD_naive":
            if self._dropout_rate > 0:
                update_mat = self._lr * self._dw * self._dropout_map
                self.weights -= update_mat
            else:
                self.weights -= self._lr * self._dw

        elif self._optimization == "SGD_momentum":
            self._velocity = self._u * self._velocity + self._dw
            update_term = self._lr * self._velocity

            if self._dropout_rate > 0:
                update_mat = update_term * self._dropout_map
                self.weights -= update_mat
            else:
                self.weights -= update_term
        return


class LossLayer(object):
    def __init__(self, loss_type="square_loss"):
        self._x = 0
        self.grad = 0
        self.squared_loss = 0
        self._label = 0
        self._loss_type = loss_type
        return

    def forward(self, x, label):
        if self._loss_type == "square_loss":
            self._x = x
            self._label = label
            self.squared_loss = np.square(self._x - self._label)
            return self.squared_loss

    def backward(self):
        if self._loss_type == "square_loss":
            self.grad = 2 * (self._x - self._label)
        return self.grad
