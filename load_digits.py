import numpy as np
import matplotlib.pyplot as plt
class Value:
    """stores a single scalar value and its gradient"""

    def __init__(self, data, _children=()):
        self.data = data    # data contains the value of this variable
        self.grad = 0    # grad contains the gradient, it must be initialized with zero
        # internal variables used for autograd graph construction
        self._backward = lambda: None    # initialize backward step with empty fuction
        self._prev = set(_children)    # _prev contains child nodes

    # forward and backward pass for basic operations

    def __add__(self, other):
        """operator + (addition: self + other)"""

        # other needs to be an instance of class Value (self.__class__)
        if not isinstance(other, self.__class__):
            other = self.__class__(other)

        # calculate the sum
        forward_result = self.data + other.data

        # create a new Value (self.__class__) that can store the result and the calculation tree
        out =  self.__class__(data=forward_result, _children=(self, other))

        # backward path: both children get same (full) gradient
        def _backward():
            # calculate gradient and add it to grad:
            # adding the gradient is necessary, since a Value can get gradient information
            # from multiple other Value nodes and these gradients must sum up
            self.grad += 1 * out.grad
            other.grad += 1 * out.grad

        # add this backward function to the new node
        out._backward = _backward

        return out

    def __mul__(self, other):
        """operator * (multiplication: self * other)"""

        # other needs to be an instance of class Value (self.__class__)
        if not isinstance(other, self.__class__):
            other = self.__class__(other)

        # calculate the product
        forward_result = self.data * other.data

        # create a new Value (self.__class__) that can store the result and the calculation tree
        out = self.__class__(data=forward_result, _children=(self, other))

        # backward path: partial derivative for self and other
        def _backward():
            # calculate gradient and add it to grad:
            # adding the gradient is necessary, since a Value can get gradient information
            # from multiple other Value nodes and these grudients must sum up
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad

        # add this backward function to the new node
        out._backward = _backward

        return out

    # gradient descent starting from this node
    def backward(self, grad=1):
        """backpropagates the gradient to all children starting from this node"""

        # topological order all of the children in the graph
        topo = []    # remember the order in which the nodes must be processed
        visited = set()    # remember nodes that have been visited already

        # breadth-first search to find the processing order of the nodes in the graph
        def build_topo(v):
            if v not in visited:    # if the node is not visited yet
                visited.add(v)    # remember that it is visited now and should not be visited once more
                for child in v._prev:    # for each child node of this node
                    build_topo(child)    # visit each child node
                topo.append(v)    # insert this node at the end of the list of processing order
                # (please notice, that all child nodes will be inserted earlier)

        # start the breadth-first search from this node
        build_topo(self)    # (please notice, that this node will be inserted as last element of the processing list)

        # initialize the gardient of this node with the given value
        self.grad = grad

        # go one variable at a time and apply the chain rule to get its gradient
        for v in reversed(topo):    # in reverse order of the processing list that was build above
            v._backward()    # call the _backward function that is specified for each operation

    def __pow__(self, other):
        """operator ** (power: self ** other)"""
        assert isinstance(other, (int, float)), 'only supporting int/float powers for now'

        # calculate the product
        forward_result =    self.data ** other

        # create a new Value (self.__class__) that can store the result and the calculation tree
        out = self.__class__(data=forward_result, _children=(self,))

        # backward path: deviation
        def _backward():
            self.grad +=    (other * self.data ** (other - 1)) * out.grad

        # add this backward function to the new node
        out._backward = _backward

        return out

    def __neg__(self):
        """negation: -self"""
        return self * -1

    def __sub__(self, other):
        """operator - (subtraction: self - other)"""
        return self + (-other)

    def __truediv__(self, other):
        """operator / (division: self / other)"""
        return self * other ** -1

    def __radd__(self, other):
        """operator + (addition with Value as right-hand operand: other + self)"""
        return self + other

    def __rsub__(self, other):
        """operator - (subtraction with Value as right-hand operand: other - self)"""
        return other + (-self)

    def __rmul__(self, other):
        """operator * (multiplication with Value as right-hand operand: other * self)"""
        return self * other

    def __rtruediv__(self, other):
        """operator / (division with Value as right-hand operand: other / self)"""
        return other * self ** -1

    # define how the Value class is printed
    def __repr__(self):
        """string representation of this class"""
        return f"Value(data={self.data}, grad={self.grad})"

class Module:
    """base class"""

    def zero_grad(self):
        """reset gradient for all parameters"""
        for p in self.parameters():    # iterate through the list of all parameters
            p.grad = 0    # parameter p is of type Value, it's gradient is set to zero

    def parameters(self):
        """return the parameters, should be overwritten in derived classes"""
        return []

    def forward(self, x):
        """forward pass for input x, should be overwritten in derived classes"""
        return None

    def __call__(self, x):
        """call function for forward pass with input x"""
        return self.forward(x)

import random

class Neuron(Module):
    """implements an abstract interface for a single neuron"""

    def __init__(self, in_features):
        """creation and initialization of this neuron for in_features input neurons"""
        minimum_value = -(1 / in_features) ** 0.5
        maximum_value = (1 / in_features) ** 0.5
        self.w = []    # the neuron needs one weight for each input neuron
        for _ in range(in_features):    # nin = number of input neurons
            initial_value_w_i = random.uniform(minimum_value, maximum_value)    # random uniform initialization
            w_i = Value(initial_value_w_i)    # make this weight differentiable, set initial value
            self.w.append(w_i)
        # add one differentiable bias weight, initialize it with 0
        self.b =    Value(0)

    def forward(self, x):
        """return the neuron's activation for input x"""
        activation = 0    # initial activation is zero
        for w_i, x_i in zip(self.w, x):    # iterate through all inputs and weights
            # multiply weight w_i with input x_i and add it to the activation
            activation +=    w_i*x_i
        activation += self.b    # add bias
        return activation

    def parameters(self):    # overwrite function of base class
        """return the neuron's weights and bias"""
        # concatenate the list of weights with the list of the single bias weight
        parameter_list = self.w + [self.b]
        return parameter_list

    def __repr__(self):
        """string representation of this class"""
        return f'Neuron({len(self.w)} inputs)'

class Linear(Module):
    """implements an abstract interface for a fully connected layer"""

    def __init__(self, in_features, out_features):
        """creation and initialization of this layer for in_features input neurons
        and out_features neurons in this layer"""
        self.neurons = []    # start with empty list
        for _ in range(out_features):    # add nout neurons
            neuron = Neuron(in_features)    # create and initialize a new neuron
            self.neurons.append(neuron)    # add it to the list

    def forward(self, x):
        """return the activations of all neurons in this layer for input x"""
        activations = []    # store results in list, start with empty list
        for neuron in self.neurons:    # iterate through all neurons in this layer
            # calculate the activation of this neuron
            activation = neuron(x)    
            # add it to the list
            activations.append(activation)
        if len(activations) == 1:    # if there is only a single neuron
            return activations[0]    # return its activation
        else:    # otherwise
            return activations    # return the list of activations for all neurons in this layer

    def parameters(self):    # overwrite function of base class
        """return weights and bias of all neurons in this layer"""
        parameters = []    # store all parameters in a list, start with empty list
        for neuron in self.neurons:    # iterate through all neurons in this layer
            neuron_i_parameters = neuron.parameters()    # get list of parameters for this neuron
            parameters = parameters + neuron_i_parameters    # concatenate the lists
        return parameters

    def __repr__(self):
        """string representation of this class"""
        return f'FCLayer[{", ".join(str(neuron) for neuron in self.neurons)}]'

def relu(x):
    if isinstance(x, (list, tuple)):    # in case of more then one input
        out = []    # output will be a list of multiple elements, start with empty list
        for x_i in x:    # for each element x_i in the list x
            out_i = relu(x_i)    # calculate the element's non-linear output
            out.append(out_i)    # and add it to the list
        return out    # return the list of outputs
    else:
        # x needs to be an instance of class Value
        if not isinstance(x, Value):
            x = Value(x)

        # calculate the relu output
        forward_result = max(0, x.data)

        # create a new Value that can store the result and the calculation tree
        out = Value(forward_result, (x,))

        # backward path: deviation
        def _backward():
            # calculate derivative of activation function (relu)
            if forward_result > 0:
                derivative_relu = 1
            else:
                derivative_relu = 0
            # update gradient
            x.grad += derivative_relu * out.grad

        # add this backward function to the new node
        out._backward = _backward

        return out

def sigmoid(x):
    if isinstance(x, (list, tuple)):    # in case of more then one input
        out = []    # output will be a list of multiple elements, start with empty list
        for x_i in x:    # for each element x_i in the list x
            out_i = sigmoid(x_i)    # calculate the element's non-linear output
            out.append(out_i)    # and add it to the list
        return out    # return the list of outputs
    else:
        # x needs to be an instance of class Value
        if not isinstance(x, Value):
            x = Value(x)

        # calculate the sigmoid output
        forward_result = 1 / (1 + np.exp(-x.data))

        # create a new Value that can store the result and the calculation tree
        out =  Value(forward_result, (x,)) 

        # backward path: deviation
        def _backward():
            # calculate derivative of activation function (sigmoid)
            derivative_sigmoid = forward_result * (1 - forward_result)
            # update gradient
            x.grad +=  derivative_sigmoid * out.grad

        # add this backward function to the new node
        out._backward = _backward

        return out

# define the neural network
class Network(Module):
    """implements a multi layer perceptron (MLP)
    with one hidden layer of 10 neurons and one output layer of a single neuron

    hint: construction similar to PyTorch
          --> compare with "Code Beispiel 2.9 [2/4] CNN in PyTorch: Netzwerk definieren" in slides of chapter 2
    """

    def __init__(self):
        """create and initialize the two fully connected layers"""
        # create fully connected layer: 64-dimensional input, 10 hidden neurons
        self.fc1 = Linear(64, 10)
        # create fully connected layer: 10 hidden neurons as input, 1 output neuron
        self.fc2 = Linear(10, 1)    

    def forward(self, x):
        """forward pass for input x"""
        # calculation of 1st layer: input x --> fully connected layer --> relu
        x = relu(self.fc1(x))
        # calculation of 2nd layer: output of 1st layer x --> fully connected layer --> sigmoid
        x = sigmoid(self.fc2(x)) 
        return x

    def parameters(self):
        """return parameters (weights, bias) of all layers in the network"""
        fc1_parameters = self.fc1.parameters()    # get the list of parameters for 1st layer
        fc2_parameters = self.fc2.parameters()    # get the list of parameters for 2nd layer
        return fc1_parameters + fc2_parameters    # concatenate the lists

    def __call__(self, x):
        """call function for forward pass with input x
        type checking needed to enable batch processing in case x is a numpy array"""
        if isinstance(x, (np.ndarray, np.float32)):    # in case it is a numpy array
            if x.ndim > 1:    # if it has more than one dimension (batch dimension + feature dimension)
                x = x.tolist()    # make it a list
                y = []    # output will be a list of multiple elements, start with empty list
                for x_i in x:    # for each element x_i in the list x
                    y_i = self.forward(x_i)    # calculate the output for sample x_i
                    y.append(y_i)    # and add it to the list
                return y    # return the list of outputs
            else:    # only the feature dimension
                x = x.tolist()    # make it a list
                self.forward(x)    # process it
        else:    # in case x is a single input
            return self.forward(x)    # process it

    def __repr__(self):
        """string representation of this class"""
        return f'MLP[{str(self.fc1)} --> ReLU --> {str(self.fc2)} --> Sigmoid]'

# set random seed for reproducibility (weight initialization is random)
random.seed(1337)

# create the neural network
net = Network()  

# print information about the neural network
print(net)
print(f'number of parameters = {len(net.parameters())}')

import numpy as np

def log(x):
    if isinstance(x, (list, tuple)):    # in case of more then one input
        out = []    # output will be a list of multiple elements, start with empty list
        for x_i in x:    # for each element x_i in the list x
            out_i = log(x_i)    # calculate the element's non-linear output
            out.append(out_i)    # and add it to the list
        return out    # return the list of outputs
    else:
        # x needs to be an instance of class Value
        if not isinstance(x, Value):
            x = Value(x)

        # calculate the logarithm output
        forward_result = np.log(x.data)

        # create a new Value (self.__class__) that can store the result and the calculation tree
        out = Value(forward_result, (x,))

        # backward path: deviation
        def _backward():
            # calculate derivative of activation function (log)
            derivative_log = 1.0 / x.data
            # update gradient
            x.grad += derivative_log * out.grad

        # add this backward function to the new node
        out._backward = _backward

        return out

def binary_cross_entropy_loss(y, t):
    # initialize loss with zero
    loss = 0

    # loop throug all outputs y and labels t simultaniously
    for y_i, t_i in zip(y, t):
        # calculate loss for this output and teacher (see above for the equation inside the sum)
        loss_i =  t_i * log(y_i) + (1 - t_i) * log(1 - y_i)  
        # add it to overall loss
        loss += loss_i

    # devide loss by the number of samples
    n_samples = len(y)
    loss = -(1 / n_samples) * loss

    return loss

def accuracy(y, t):
    # count number of samples and number of correct predictions
    correct = 0
    total = 0

    # loop throug all outputs y and labels t simultaniously
    for y_i, t_i in zip(y, t):
        # binarize the decisions of the neural network (class label 1 if y_i > 0.5, class label 0 otherwise)
        if y_i.data > 0.5:
            predicted_class = 1  
        else:
            predicted_class = 0   

        # count this sample
        total += 1

        # increase number of correct predictions if prediction for current sample is correct
        if predicted_class == t_i:
            correct += 1

    # calculate accuracy
    acc = correct / total

    return acc

def train_step(X, t, net):
    # forward pass with input data X
    y = net(X)  

    # compute loss and accuracy
    loss =  binary_cross_entropy_loss(y, t)  
    training_data_accuracy = accuracy(y, t) 

    # reset gradients from previous train step
    net.zero_grad()  

    # calculate all gradients
    loss.backward()  

    # update every parameter with sgd
    learning_rate = 0.4 #Lernrate erhöhen vielleicht
    for parameter in net.parameters():
        parameter.data = parameter.data - parameter.grad * learning_rate

    return net, loss, training_data_accuracy

def test_step(X, t, net):
    # forward pass with input data X
    y = net(X)  

    # calculate performance measures (compare network outputs y and teacher t)
    loss =  binary_cross_entropy_loss(y, t)  
    acc =  accuracy(y, t)  

    return loss, acc

from sklearn import datasets
digits_dataset = datasets.load_digits()
# convert image to floating point matrix
x = digits_dataset.images.astype('float32')

# reshape input x such that it is a 1D vector for each sample
samples, rows, cols = x.shape
x = x.reshape(samples, rows * cols)   

# normalize values to range [0, 1]
x = x/16.0 #np.max(x)

t = digits_dataset.target

# change the labels to 0 for even numbers and 1 for odd numbers
for i in range(len(t)):
    if t[i] % 2 == 0:    # even number
        t[i] = 0   
    else:    # odd number
        t[i] = 1   

# convert teacher to array
t = np.array(t)

# convert image to floating point matrix
x = digits_dataset.images.astype('float32')

# reshape input x such that it is a 1D vector for each sample
samples, rows, cols = x.shape
x = x.reshape(samples, rows * cols)   

# normalize values to range [0, 1]
x = x/16.0 #np.max(x)
from sklearn.model_selection import train_test_split

# split data to training and validation data (use 15% of data for validation)
x_train_all, x_val_all, t_train_all, t_val_all = train_test_split(x, t, test_size=0.15, random_state=42)


# def shuffle(data, target, batch_size):
#     rand = np.random.randint(0, len(target), size=batch_size)
#     return np.asarray([data[i, :] for i in rand]), np.asarray([target[i] for i in rand])
#%%
def getEpoch(data, target, batch_size):
    shuffled = list(np.arange(len(target)))
    random.shuffle(shuffled)
    ret1, ret2 = [data[i] for i in shuffled], [target[i] for i in shuffled]
    ret3, ret4 = [], []
    # for i in range(int(len(target)/batch_size)):
    for i in range(int(len(target)/batch_size)):
        ret3.append(ret1[i*32:(i+1)*32])
        ret4.append(ret2[i*32:(i+1)*32])
    return ret3, ret4

# lists to store performance measures during training
loss_list = []
accuracy_list = []
val_loss_list = []
val_accuracy_list = []
latest_accuracy = 0
step = 0
accs = np.zeros((5,), dtype=int)
val_accs = np.zeros((5,), dtype=int)
epoch = getEpoch(x_train_all, t_train_all, 32)
ex_iter, et_iter = iter(epoch[0]), iter(epoch[1])
#%%
# start training
#while latest_accuracy < 95.0: # 100 training steps
#while latest_accuracy < 95.0 or step <= 150: # 100 training steps
while np.min(accs) < 95.0 or np.min(val_accs) < 95.0:
    try:
        x_train, t_train = np.asarray(next(ex_iter)), np.asarray(next(et_iter))
    except:
        print("new epoch has started")
        epoch = getEpoch(x_train_all, t_train_all, 32)
        ex_iter, et_iter = iter(epoch[0]), iter(epoch[1])
        x_train, t_train = np.asarray(next(ex_iter)), np.asarray(next(et_iter))

    # if len(list(epoch)) == 0:
    # perform one training step (training data: x_train, t_train; neural network: net)
    # x_train, t_train = shuffle(x_train_all, t_train_all, 32)
    net, loss, train_acc = train_step(x_train, t_train, net)   

    # store performance measures
    loss_list.append(loss.data)
    accuracy_list.append(train_acc * 100)
    if (len(accuracy_list) >= 5):
      accs = accuracy_list[-5:]

    # print output and validate every 5th training step
    if step % 5 == 0:
        # perform one validation step (validation data: x_val, t_val; neural network: net)

        val_loss, val_accuracy = test_step(x_val_all, t_val_all, net)   

        # store performance measures
        val_loss_list.append(val_loss.data)
        val_accuracy_list.append(val_accuracy * 100)
        latest_accuracy = val_accuracy * 100
        if (len(val_accuracy_list) >= 5):
          val_accs = val_accuracy_list[-5:]

        # print training progress
        print(f"step {step:>3}:  training:   loss = {loss.data:.4f}, accuracy = {(train_acc * 100):.3f}%\n           validation: loss = {val_loss.data:.4f}, accuracy = {(val_accuracy * 100):.3f}%\n")
    step += 1


# final validation step
val_loss, val_accuracy = test_step(x_val_all, t_val_all, net)

# store final performance measures
loss_list.append(loss.data)
accuracy_list.append(train_acc * 100)
val_loss_list.append(val_loss.data)
val_accuracy_list.append(val_accuracy * 100)

print('final')
