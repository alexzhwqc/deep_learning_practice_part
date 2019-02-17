# -*- coding: utf-8 -*-
import numpy as np
from matplotlib import pyplot as plt

data = np.load("mnist.npz")
X_train, Y_train = data["x_train"], data["y_train"]
X_test, Y_test = data["x_test"], data["y_test"]
X_train = X_train.reshape(X_train.shape[0], -1) / 255
X_test = X_test.reshape(X_test.shape[0], -1) / 255

# Split train and validation data as 50000:10000
train_size = 50000
X_train_ = X_train[:train_size]
X_valid_ = X_train[train_size:]
Y_train_ = Y_train[:train_size]
Y_valid_ = Y_train[train_size:]

X_train = X_train_
X_valid = X_valid_
Y_train = Y_train_
Y_valid = Y_valid_

print(X_train.shape)
print(X_valid.shape)
print(Y_train.shape)
print(Y_valid.shape)

DEBUG = True
def debug(*args, **kwargs):
    global DEBUG
    if DEBUG:
        print(*args, **kwargs)

# Dimensionality of MNIST input
N_INPUT = 28*28
# Total classes number
N_CLASSES = 10

# 2 hidden layers MLP with numpy
class NN(object):
    def __init__(self, hidden_dims = (450, 400),  # (number of paras between 0.5M and 1M)
                 n_hidden = 2, mode = "train",
                 datapath = None, model_path = None):
        
        #bias list for saving all bias
        self.bias = []
        # Weights list for saving all weights
        self.weights = []
                
        # Init weights and bias of first layer
        self.bias.append(np.zeros((hidden_dims[0])))
        self.weights.append(np.empty((hidden_dims[0], N_INPUT)))
                
        # Init weights and bias of hidden layers
        for i in range(n_hidden - 1):
            self.bias.append(np.zeros((hidden_dims[i+1])))
            self.weights.append(np.empty((hidden_dims[i+1], hidden_dims[i])))
                    
        # Init weights and bias of output layer
        self.bias.append(np.zeros((N_CLASSES)))
        self.weights.append(np.empty((N_CLASSES, hidden_dims[-1])))
    
    #initialize weights and bias
    def initialize_weights(self, method="glorot"):
        # Init weights in order of layers
        for idx, weight in enumerate(self.weights):
            if method == "zero":
                self.weights[idx] = np.zeros(shape = weight.shape)
            elif method == "normal":
                self.weights[idx] = np.random.normal(loc = 0, scale = 1, size = weight.shape)
            elif method == "glorot":
                d = np.sqrt(6 / (weight.shape[0] + weight.shape[1]))
                self.weights[idx] = np.random.uniform(low = -d, high = d, size = weight.shape)
    
    def forward(self, input):
        self.cache = [input]
        for weight, bias in zip(self.weights, self.bias):
            self.cache.append(self.activation(np.dot(weight, self.cache[-1]) + bias))
        last_cache = self.cache.pop()
        return self.softmax(last_cache)

    def activation(self, input):
        return np.maximum(0, input)

    def loss(self, prediction, label):
        return -(np.log(self.softmax(prediction)[label]))

    def softmax(self, input):
        max_input = np.max(input)
        return np.exp(input - max_input) / np.sum(np.exp(input - max_input))

    def backward(self, output, label):
        tmp_list = []
        for idx, o in enumerate(output):
            if idx == label:
                tmp_list.append(o - 1)
            else:
                tmp_list.append(o)
        pre_activation_grads = np.asarray(tmp_list).reshape(-1, 1)
        
        # Bias gradients
        self.bias_grads = []
        # Weight gradients 
        self.weight_grads = []
                
        for idx, (weight, _) in enumerate(zip(reversed(self.weights), reversed(self.bias))):
            layer_passed = np.asarray(list(reversed(self.cache))[idx])
            self.weight_grads.insert(0, np.dot(pre_activation_grads,layer_passed.reshape(-1, 1).T))
            self.bias_grads.insert(0, pre_activation_grads.reshape(-1))
            layer_passed_grads = np.dot(weight.T , pre_activation_grads)
            
            tmp_list = []
            for cached in list(reversed(self.cache))[idx]:
                if cached > 0:
                    tmp_list.append(1)
                else:
                    tmp_list.append(0)
            pre_activation_grads = layer_passed_grads * np.asarray(tmp_list).reshape(-1, 1)
            
    #udate wights and bias
    def update(self, learning_rate):
        for idx, (weight_grad, bias_grad) in enumerate(zip(self.weight_grads, self.bias_grads)):
            self.bias[idx] = self.bias[idx] - learning_rate * bias_grad
            self.weights[idx] = self.weights[idx] - learning_rate * weight_grad

    def train(self, inputs, labels, N_epochs = 1, lr = 0.001, log = True):
        losses = []
        for epoch in range(N_epochs):
            loss = []
            for idx, (input, label) in enumerate(zip(inputs, labels), 1):
                result = self.forward(input)
                loss.append(nn.loss(result, label))
                
                    
                self.backward(result, label)
                self.update(lr)
            losses.append(np.mean(loss))
        return losses
    
    def test(self, inputs, labels):
        loss, acc = zip(*[(self.loss(self.forward(input), label), np.argmax(self.forward(input)) == label)
                          for input,label in zip(inputs, labels)])
        return np.mean(loss), np.mean(acc)

#MLP with initialization of Zero
nn = NN()
nn.initialize_weights("zero")
zero_loss = nn.train(X_train, Y_train, N_epochs = 10, lr = 1e-3)

#MLP with initialization of standard normal distribution
nn = NN()
nn.initialize_weights("normal")
normal_loss = nn.train(X_train, Y_train, N_epochs = 10, lr = 1e-3)

#MLP with initialization of Glorot
nn = NN()
nn.initialize_weights("glorot")
glorot_loss = nn.train(X_train, Y_train, N_epochs = 10, lr = 1e-3)

plt.plot(range(1, 11), zero_loss, label="zero", color='b')
plt.plot(range(1, 11), normal_loss, label="normal", color='c')
plt.plot(range(1, 11), glorot_loss, label="glorot", color='r')
plt.legend(title = "Initialization")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.show()

# Grid search for best hyperparameters
def grid_search():
    best_h1 = -1
    best_h2 = -1
    best_lr = -1
    best_nn = None
    best_acc = -1
        
    for h1 in [450, 350, 150]:
        for h2 in [400, 200, 100]:
            for lr_item in [0.1, 0.01, 0.001]:
                nn = NN(hidden_dims = (h1, h2))
                nn.initialize_weights("glorot")
                nn.train(X_train, Y_train, N_epochs = 10, lr = lr_item, log = False)
                _, acc = nn.test(X_valid, Y_valid)
                print("h1:{0}, h2:{1}, lr:{2}, acc:{3}".format(h1, h2, lr_item, acc))
                if acc > best_acc:
                    best_acc = acc
                    best_nn = nn
                    best_h1 = h1
                    best_h2 = h2
                    best_lr = lr_item
                    
    return best_nn

_, acc = grid_search().test(X_test, Y_test)
print("Acc on the test set: {:.3%}".format(acc))