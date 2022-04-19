import numpy as np
import pandas as pd
import sys

def read_idx_array(filename: str):
    """
        reads an idx file from memory and returns a numpy array
    """
    
    try:
        file = open(filename, "rb")
    except OSError:
        print("file not found.")
        return

    #not important in this case
    magic_number = file.read(2)
    dtype = file.read(1)

    if dtype != b'\x08':
        print("not a ubyte file")
        return
    
    #number of dimensions
    dim_num = int.from_bytes(file.read(1), "big")
    if dim_num <= 0:
        print("dimensions must be greater than zero.")
        return

    else:
        data_sz = np.ndarray(dim_num, dtype=np.uint64)
        for i in range(dim_num):
            dim_sz = int.from_bytes(file.read(4), "big")
            data_sz[i] = dim_sz

    data = np.frombuffer(file.read(), dtype=np.uint8)
    data = data.reshape(tuple(data_sz))

    return data

class dense():
    def __init__(self, dims):
        #dims[0] number of rows for weights class and biases. dims[1] number of inputs being fed into dense
        self.out_dim, self.in_dim = dims[0], dims[1]

        self.weights = np.random.randn(self.out_dim, self.in_dim)
        self.biases = np.random.randn(self.out_dim, 1)
    
    def sum_weights_biases(self, input: np.array):
        return (np.matmul(self.weights, input) + self.biases)

class multi_layer_perceptron():
    def __init__(self, sizes):
        """
            sizes is a list of neuron sizes: from the input to the hidden layers to the output
        """
        self.sizes = sizes

        size_pairs = [(x, y) for x, y in zip(sizes[1:], sizes[:-1])]

        self.layers = [dense(dims) for dims in size_pairs]
        self.num_layers = len(self.layers)

        self.zs = None
        #activations from past calls to feed_forward
        self.activations = None

    def feed_forward(self, input: np.array):
        """
            input array takes the form of
            input =
            [ a_1 a_2 ...
              b_1 b_2 ...
              ... ... ...]
            
            we need to transpose it so that it takes the form of

            input = 
            [ a_1 b_1 ...
              a_2 b_2 ...
              ... ... ...
              a_n b_n ...]

            thus

            weights = 
            [ w_00 w_01 w_02 ....
              w_10 w_11 w_12 ....
              .... .... .... ....
              w_n0 w_n1 w_n2 ....]

            so that

            weights * input.T  + biases=
            [a*w_0 b*w_0 .....   + [bias_0
             ..... ..... .....   +  ......
             a*w_n b*w_n .....]  +  bias_n]
        """

        #need to temporarily transpose it
        out = input.T
        self.zs = []
        self.activations = [out]
        

        for layer in self.layers:
            out = layer.sum_weights_biases(out)
            self.zs.append(out)
            out = self.sigmoid(out)
            self.activations.append(out)

        #to grab output without softmax, access through self.actiations[-1]
        out = self.softmax(out)
        return out.T

    def SGD(self, train, labels, num_epoch = 100, batch_sz=500, learn_rate=0.05, decay=0):
        
        num_train = train.shape[0]
        labels = labels.reshape(-1, 1)

        #simpole trick so the first iteration starts with original learn_rate
        learn_rate += decay

        for epoch in range(num_epoch):
            
            #add decay to learn rate to slowly finetune weights
            learn_rate -= decay

            shuf_idx = np.arange(num_train)
            np.random.shuffle(shuf_idx)

            batch_indices = [shuf_idx[k:k+batch_sz] for k in range(0, num_train, batch_sz)]

            for b in batch_indices:
                batch = train[b]
                batch_labels = labels[b]

                nabla_b, nabla_w = self.back_propagation(batch, batch_labels)

                #update weights
                for i in range(self.num_layers):
                    self.layers[i].weights -= learn_rate * nabla_w[i]

                    self.layers[i].biases -= learn_rate * nabla_b[i]

            #test accuracy
            out = self.feed_forward(train)
            y_pred = np.argmax(out, axis=1).reshape(-1, 1)
            acc = np.sum(y_pred == labels) / num_train
            print("Epoch number ", epoch, ": accuracy = ", acc)

    def back_propagation(self, train, labels):

        #train isn't transposed yet
        num_train = train.shape[0]

        nabla_w = [np.zeros(layer.weights.shape) for layer in self.layers]
        nabla_b = [np.zeros(layer.biases.shape) for layer in self.layers]

        self.feed_forward(train)
        out = self.softmax(self.activations[-1])
        
        """
            the activations of each training example are in the columns of the activation matrix
        """
        #start of backwards passing
        delta = self.cross_entropy_loss_prime(out, labels)
        nabla_b[-1] = np.sum(delta, axis=1)
        nabla_w[-1] = np.matmul(delta, self.activations[-2].T)

        for layer_idx in range(2, self.num_layers+1):
            z = self.zs[-layer_idx]
            delta_sigmoid = self.sigmoid_prime(z)
            delta = np.matmul(self.layers[-layer_idx+1].weights.T, delta) * delta_sigmoid
            
            nabla_b[-layer_idx] = np.sum(delta, axis=1)
            nabla_w[-layer_idx] = np.matmul(delta, self.activations[-layer_idx - 1].T)
            

        nabla_b = [(grad / num_train).reshape(-1, 1) for grad in nabla_b]
        nabla_w = [grad / num_train for grad in nabla_w]

        return nabla_b, nabla_w

        

    def sigmoid(self, input: np.array):
        #sigmoid activation function
        temp = np.array(input, dtype=np.long)

        return 1 / (1 + np.exp(-input))

    def sigmoid_prime(self, input: np.array):
        #https://towardsdatascience.com/derivative-of-the-sigmoid-function-536880cf918e
        #first derivative of sigmoid
        return self.sigmoid(input)*(1-self.sigmoid(input))
    
    def softmax(self, input: np.array):
        #https://deepai.org/machine-learning-glossary-and-terms/softmax-layer

        #turns a vector of k real values into a vector of k real values that sum to one so they may be interpretted
        # as probabilities

        #axis=0 since the input array takes the form of [a_1 a_2 a_3 ; b_1 b_2 b_3].T
        return np.exp(input) / np.sum(np.exp(input), axis=0)

    def cross_entropy_loss(self, y_pred, y):
        """
            source: https://deepnotes.io/softmax-crossentropy#derivative-of-softmax

            y_pred is the softmax output of the neural network
            y is a list of label values

            note that in this function y_pred is transposed such that:
            input =
            [ a_1 b_1 ...
              a_2 b_2 ...
              ... ... ...]
        """
        #NOTE we softmax outside of the function
        m = y.shape[0]
        log_likelihood = -np.log(y_pred[y, range(m)])

        return np.sum(log_likelihood) / m

    def cross_entropy_loss_prime(self, y_pred, y):
        m = y.shape[0]
        
        grad = y_pred.copy()
        """
        #NOTE: we assume that y_pred has already been processed by softmax!
        grad[y, range(m)] -= 1
        #grad = grad / m
        """
        
        for i in range(m):
            grad[y[i], i] -= 1
        
        return grad

#program start
if __name__ == "__main__":
    
    if len(sys.argv) < 4:
        print("no command lines supplied. entering test 'mode'")
        get_test_acc = True
        train_fp = "train_image.csv"
        train_label_fp = "train_label.csv"
        test_fp = "test_image.csv"
        test_label_fp = "test_label.csv"
    else:
        get_test_acc = False
        train_fp = sys.argv[1] 
        train_label_fp = sys.argv[2]
        test_fp = sys.argv[3]

    train = pd.read_csv(train_fp, header=None)
    train = train.to_numpy(dtype=float)

    #train = read_idx_array(train_fp)
    #train_y = read_idx_array(train_labels_fp)

    #test sample please comment out before final submission
    sz = 10000
    random_idx = np.random.randint(train.shape[0], size=sz)
    train = train[random_idx, :]
    print(train.shape)

    train_r = train / 255.0

    """
    #attempt to normalize the data
    train_mean = np.mean(train_r)
    train_stdev = np.std(train_r)

    train_r = (train_r - train_mean) / train_stdev
    """

    train_y = np.fromfile(train_label_fp, dtype=np.int64, sep=" ")

    #test sample please comment out before final submission
    train_y = train_y[random_idx]

    #https://stats.stackexchange.com/questions/164876/what-is-the-trade-off-between-batch-size-and-number-of-iterations-to-train-a-neu

    #740, 500, 500, 10, lr=0.001 - 70% acc
    #740, 200, 100, 80, 10. lr=0.01 - 79% acc
    #740, 200, 200, 200, 200, 10. lr = 0.1, decay - 0.002 79% acc
    #784 inputs, two hidden layers (both size 16), one output layer of size 10
    mlp = multi_layer_perceptron([784, 100, 100, 100, 10])


    #third parameters - 20
    #fourth parather - sampled
    #
    epoch = 100
    batch_sz = 100
    learning_rate = 0.02
    decay = 0.0
    mlp.SGD(train_r, train_y, epoch, batch_sz, learning_rate, decay)

    test = pd.read_csv(test_fp, header=None)
    test = test.to_numpy(dtype=float)

    test_r = test / 255.0

    test_scores = mlp.feed_forward(test_r)
    test_pred = np.argmax(test_scores, axis=1).reshape(-1, 1)

    if get_test_acc:
        test_label = np.fromfile(test_label_fp, dtype=np.int64, sep=" ").reshape(-1, 1)

        acc = np.sum(test_pred == test_label) / len(test_label)
        print("Test Accuracy: ", acc)
    
    test_pred_fp = "test_predictions.csv"

    #np.savetxt doesn't seem to be able to overwrite files
    import os
    if os.path.exists(test_pred_fp):
        os.remove(test_pred_fp)

    np.savetxt(test_pred_fp, test_pred, fmt="%d")


