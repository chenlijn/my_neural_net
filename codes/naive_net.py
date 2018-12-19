
import numpy as np

from layers import *


class NaiveNet(object):
    def __init__(self):
        ###############################
        ### define the network here ###
        ###############################
        self.fc1_layer = FCLayer(2, 5, activation_func='ReLU', dropout_rate=0.2)
        self.fc2_layer = FCLayer(5, 5, activation_func='ReLU', dropout_rate=0.2)
        self.fc3_layer = FCLayer(5, 2, activation_func='sigmoid', dropout_rate=0.2)
        self.loss_layer = LossLayer()

        self.weight_dict = 0
        return

    def training_setting(self, optimization="SGD_momentum", lr=1e-4, u=0.9):
        self.optimization = optimization
        self.lr = lr
        self.u = u
        return

    def forward_prediction(self, one_image):
        fc1_out = self.fc1_layer.forward(one_image)
        fc2_out = self.fc2_layer.forward(fc1_out)
        fc3_out = self.fc3_layer.forward(fc2_out)
        return fc3_out

    def forward_loss(self, one_image, label):
        # network forward pass
        fc1_out = self.fc1_layer.forward(one_image)
        fc2_out = self.fc2_layer.forward(fc1_out)
        fc3_out = self.fc3_layer.forward(fc2_out)
        loss = self.loss_layer.forward(fc3_out, label)
        return loss

    def backward(self):
        loss_grad = self.loss_layer.backward()

        f3_grad = self.fc3_layer.backward(loss_grad)
        self.fc3_layer.update_weights(optimization=self.optimization, lr=self.lr, u=self.u)

        f2_grad = self.fc2_layer.backward(f3_grad)
        self.fc2_layer.update_weights(optimization=self.optimization, lr=self.lr, u=self.u)

        _ = self.fc1_layer.backward(f2_grad)
        self.fc1_layer.update_weights(optimization=self.optimization, lr=self.lr, u=self.u)
        return

    def save_weights(self, save_file):
        self.weight_dict = {"fc1": self.fc1_layer.weights,
                            "fc2": self.fc2_layer.weights,
                            "fc3": self.fc3_layer.weights}
        np.save(save_file, self.weight_dict)
        return

    def load_weights(self, weight_file):
        self.weight_dict = np.load(weight_file).item()
        self.fc1_layer.weights = self.weight_dict["fc1"]
        self.fc2_layer.weights = self.weight_dict["fc2"]
        self.fc3_layer.weights = self.weight_dict["fc3"]
        return
