
import numpy as np

from naive_net import *


def define_and_train_network():

    # prepare data
    train_data = np.load("./train_data.npy")
    train_num, temp = train_data.shape
    data_dim = temp - 1
    labels = train_data[:, data_dim].tolist()
    train_data = train_data[:, 0:data_dim]
    class_num = int(np.max(labels) + 1)

    # get naive net
    naive_net = NaiveNet()

    # set up training
    epoch = 10
    iter_num = epoch * train_num
    naive_net.training_setting(optimization="SGD_naive", lr=1e-4, u=0.9)

    # train the network
    for iteration in range(iter_num):

        # get a random example from training set
        idx = int(np.random.uniform(0, train_num-1))
        one_data = train_data[idx, :]
        one_data = np.reshape(one_data.T, (one_data.shape[0], 1))

        label = np.zeros((class_num, 1))
        label[int(labels[idx])] = 1

        loss = naive_net.forward_loss(one_data, label)
        print "iteration: {}, loss: {}".format(iteration, loss.sum())

        # backward pass
        naive_net.backward()

    # save network weights
    naive_net.save_weights("./naive_net_weights.npy")
    return


if __name__ == "__main__":

    define_and_train_network()
