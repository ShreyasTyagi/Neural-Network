import numpy as np
data_train = np.loadtxt("mnist_train.csv", delimiter=",", skiprows=1)
labels = data_train[:,0].astype(int)
X_data = data_train[:,1:] / 255.0
def one_hot(y,classes=10):
    one_hot_y = np.zeros((y.shape[0],classes))
    one_hot_y[np.arange(y.shape[0]),y] = 1
    return one_hot_y
y = one_hot(labels)
loss_history = []
class layer:
    def __init__(self,no_input,no_output):
        self.no_input = no_input
        self.no_output = no_output
# creating dummy weights and bias
        self.w = np.random.randn(no_output,no_input) * np.sqrt(2/no_input)
        self.bias = np.zeros(no_output)
    # Forward for a layer
    def forward (self,layer_input):
        self.output = np.dot(layer_input,self.w.T) + self.bias 
        self.layer_input = layer_input
        return self.output
def softmax(z):
    if z.ndim == 1:
        z = z - np.max(z)
        exp_z = np.exp(z)
        return exp_z / np.sum(exp_z)
    else:
        z = z - np.max(z,axis=1,keepdims=True)
        exp_z = np.exp(z)
        return exp_z / np.sum(exp_z,axis=1,keepdims=True)
def sigmoid(inputs):
    z = 1/(1+np.exp(-inputs))
    return z
def sigmoid_derivative(z):
    sigm = sigmoid(z)
    sigmoid_deri = sigm*(1-sigm)
    return sigmoid_deri
def calc_loss(softmax_out, y_true):
    loss = -np.sum(y_true * np.log(softmax_out + 1e-9))
    return loss
layer1 = layer(784,128)
layer2 = layer(128,64)
layer3 = layer(64,10)
class sequence:
    def __init__(self,sample_input):
        self.sample_input = sample_input
        self.layer1_output = layer1.forward(sample_input)
        self.acti_layer1 = sigmoid(self.layer1_output)
        self.layer2_output = layer2.forward(self.acti_layer1)
        self.acti_layer2 = sigmoid(self.layer2_output)
        self.layer3_output = layer3.forward(self.acti_layer2)
        self.softmax_layer = softmax(self.layer3_output)
epochs = 10
batch_size = 10
learning_rate = 0.1
for e in range(epochs):
    epoch_cost = 0
    index = np.random.permutation(X_data.shape[0])
    X_shuffled = X_data[index]
    y_shuffled = y[index]
    for j in range(0,len(X_data),batch_size):
        X_batch = X_shuffled[j:j+batch_size]
        y_batch = y_shuffled[j:j+batch_size]
        # forward pass 
        z1 = layer1.forward(X_batch)
        a1 = sigmoid(z1)
        z2 = layer2.forward(a1)
        a2 = sigmoid(z2)
        z3 = layer3.forward(a2)
        a3 = softmax(z3)
        # backprop
        delta3 = a3 - y_batch
        delta2 = np.dot(delta3, layer3.w) * sigmoid_derivative(z2)
        delta1 = np.dot(delta2, layer2.w) * sigmoid_derivative(z1)
        # gradients
        dw3 = np.dot(delta3.T, a2) / batch_size
        dw2 = np.dot(delta2.T, a1) / batch_size   
        dw1 = np.dot(delta1.T, X_batch) / batch_size
        db3 = np.sum(delta3, axis=0) / batch_size
        db2 = np.sum(delta2, axis=0) / batch_size
        db1 = np.sum(delta1, axis=0) / batch_size
        # update
        layer3.w -= learning_rate * dw3
        layer2.w -= learning_rate * dw2
        layer1.w -= learning_rate * dw1
        layer3.bias -= learning_rate * db3
        layer2.bias -= learning_rate * db2
        layer1.bias -= learning_rate * db1
        epoch_cost += calc_loss(a3, y_batch)
    print(f"epoch {e+1} | cost: {epoch_cost:.4f}")
    loss_history.append(epoch_cost)
def predict(sample_input):
    values = sequence(sample_input)
    prediction = np.argmax(values.softmax_layer)
    return prediction
def predict_all(X):
    predictions = []
    for i in X:
        predictions.append(predict(i))
    return np.array(predictions)
data_test = np.loadtxt("mnist_test.csv", delimiter=",", skiprows=1)
labels = data_test[:, 0].astype(int)
X_data_test = data_test[:, 1:] / 255.0
def one_hot(y,classes=10):
    one_hot_y = np.zeros((y.shape[0],classes))
    one_hot_y[np.arange(y.shape[0]), y] = 1
    return one_hot_y
y_test = one_hot(labels)
print(predict_all(X_data_test))
# accuracy
predictions = predict_all(X_data_test)
actual = np.argmax(y_test, axis=1)
accuracy = np.sum(predictions == actual) / X_data_test.shape[0]
print("accuracy:", accuracy)
import matplotlib.pyplot as plt
plt.plot(loss_history)
plt.xlabel("Epoch")
plt.ylabel("Cost")
plt.title("Loss Curve")
plt.grid(True)
plt.show()import numpy as np
data_train = np.loadtxt("mnist_train.csv", delimiter=",", skiprows=1)
labels = data_train[:,0].astype(int)
X_data = data_train[:,1:] / 255.0
def one_hot(y,classes=10):
    one_hot_y = np.zeros((y.shape[0],classes))
    one_hot_y[np.arange(y.shape[0]),y] = 1
    return one_hot_y
y = one_hot(labels)
loss_history = []
class layer:
    def __init__(self,no_input,no_output):
        self.no_input = no_input
        self.no_output = no_output
# creating dummy weights and bias
        self.w = np.random.randn(no_output,no_input) * np.sqrt(2/no_input)
        self.bias = np.zeros(no_output)
    # Forward for a layer
    def forward (self,layer_input):
        self.output = np.dot(layer_input,self.w.T) + self.bias 
        self.layer_input = layer_input
        return self.output
def softmax(z):
    if z.ndim == 1:
        z = z - np.max(z)
        exp_z = np.exp(z)
        return exp_z / np.sum(exp_z)
    else:
        z = z - np.max(z,axis=1,keepdims=True)
        exp_z = np.exp(z)
        return exp_z / np.sum(exp_z,axis=1,keepdims=True)
def sigmoid(inputs):
    z = 1/(1+np.exp(-inputs))
    return z
def sigmoid_derivative(z):
    sigm = sigmoid(z)
    sigmoid_deri = sigm*(1-sigm)
    return sigmoid_deri
def calc_loss(softmax_out, y_true):
    loss = -np.sum(y_true * np.log(softmax_out + 1e-9))
    return loss
layer1 = layer(784,128)
layer2 = layer(128,64)
layer3 = layer(64,10)
class sequence:
    def __init__(self,sample_input):
        self.sample_input = sample_input
        self.layer1_output = layer1.forward(sample_input)
        self.acti_layer1 = sigmoid(self.layer1_output)
        self.layer2_output = layer2.forward(self.acti_layer1)
        self.acti_layer2 = sigmoid(self.layer2_output)
        self.layer3_output = layer3.forward(self.acti_layer2)
        self.softmax_layer = softmax(self.layer3_output)
epochs = 10
batch_size = 10
learning_rate = 0.1
for e in range(epochs):
    epoch_cost = 0
    index = np.random.permutation(X_data.shape[0])
    X_shuffled = X_data[index]
    y_shuffled = y[index]
    for j in range(0,len(X_data),batch_size):
        X_batch = X_shuffled[j:j+batch_size]
        y_batch = y_shuffled[j:j+batch_size]
        # forward pass 
        z1 = layer1.forward(X_batch)
        a1 = sigmoid(z1)
        z2 = layer2.forward(a1)
        a2 = sigmoid(z2)
        z3 = layer3.forward(a2)
        a3 = softmax(z3)
        # backprop
        delta3 = a3 - y_batch
        delta2 = np.dot(delta3, layer3.w) * sigmoid_derivative(z2)
        delta1 = np.dot(delta2, layer2.w) * sigmoid_derivative(z1)
        # gradients
        dw3 = np.dot(delta3.T, a2) / batch_size
        dw2 = np.dot(delta2.T, a1) / batch_size   
        dw1 = np.dot(delta1.T, X_batch) / batch_size
        db3 = np.sum(delta3, axis=0) / batch_size
        db2 = np.sum(delta2, axis=0) / batch_size
        db1 = np.sum(delta1, axis=0) / batch_size
        # update
        layer3.w -= learning_rate * dw3
        layer2.w -= learning_rate * dw2
        layer1.w -= learning_rate * dw1
        layer3.bias -= learning_rate * db3
        layer2.bias -= learning_rate * db2
        layer1.bias -= learning_rate * db1
        epoch_cost += calc_loss(a3, y_batch)
    print(f"epoch {e+1} | cost: {epoch_cost:.4f}")
    loss_history.append(epoch_cost)
def predict(sample_input):
    values = sequence(sample_input)
    prediction = np.argmax(values.softmax_layer)
    return prediction
def predict_all(X):
    predictions = []
    for i in X:
        predictions.append(predict(i))
    return np.array(predictions)
data_test = np.loadtxt("mnist_test.csv", delimiter=",", skiprows=1)
labels = data_test[:, 0].astype(int)
X_data_test = data_test[:, 1:] / 255.0
def one_hot(y,classes=10):
    one_hot_y = np.zeros((y.shape[0],classes))
    one_hot_y[np.arange(y.shape[0]), y] = 1
    return one_hot_y
y_test = one_hot(labels)
print(predict_all(X_data_test))
# accuracy
predictions = predict_all(X_data_test)
actual = np.argmax(y_test, axis=1)
accuracy = np.sum(predictions == actual) / X_data_test.shape[0]
print("accuracy:", accuracy)
import matplotlib.pyplot as plt
plt.plot(loss_history)
plt.xlabel("Epoch")
plt.ylabel("Cost")
plt.title("Loss Curve")
plt.grid(True)
plt.show()
