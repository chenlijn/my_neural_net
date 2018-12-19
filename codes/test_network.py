
import numpy as np
from naive_net import *


def test_network():

    test_data = np.load("./test_data.npy")
    test_num, temp = test_data.shape
    data_dim = temp - 1
    labels = test_data[:, data_dim].tolist()
    test_data = test_data[:, 0:data_dim]

    # get the naive network
    naive_net = NaiveNet()
    naive_net.load_weights("naive_net_weights.npy")

    correct_num = 0
    for i in range(test_num):
        one_data = test_data[i, :]
        one_data = np.reshape(one_data.T, (one_data.shape[0], 1))

        label = labels[i]

        prediction = naive_net.forward_prediction(one_data)
        if label == prediction.argmax():
            correct_num += 1

        print "label: {}, prediction: {}".format(label, prediction.argmax())

    print "total_num: {}, correct_num: {}, prediction: {}".format(test_num, correct_num, correct_num/float(test_num))


if __name__ == "__main__":

    test_network()
