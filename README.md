# Neural-Network from scratch using only NumPy
A framework to classify handwritten digits without using deeplearnig libraries like Tensorflow and PyTorch. Build from scratch using the basics of Machine Learning using only Numpy.

## Core concepts implemented :

- Forward propogation
- Backward propogation
- Gradient Descent
- Sigmoid activation
- Softmax activation
- Cross-entropy loss
## Mathematical Foundation

The neural network is trained using forward propagation,
cross-entropy loss, and backpropagation with gradient descent.

### Forward Propagation

For each layer:

z = Wx + b

a = σ(z)

where:
- W = weights
- b = biases
- σ = sigmoid activation function

The final layer uses the Softmax activation function to
convert outputs into probability distributions.

### Loss Function

The model uses categorical cross-entropy loss:

L = -Σ y log(ŷ)

where:
- y = true labels
- ŷ = predicted probabilities

### Backpropagation

Gradients are computed using the chain rule.

Output layer error:

δ₃ = a₃ - y

Hidden layer errors:

δ₂ = (δ₃W₃) ⊙ σ'(z₂)

δ₁ = (δ₂W₂) ⊙ σ'(z₁)

where:
- δ = layer error
- ⊙ = element-wise multiplication
- σ' = derivative of sigmoid activation

### Gradient Computation

Weight gradients:

dW₃ = (δ₃ᵀ a₂) / batch_size

dW₂ = (δ₂ᵀ a₁) / batch_size

dW₁ = (δ₁ᵀ X) / batch_size

Bias gradients:

db = Σδ / batch_size

### Parameter Updates

Weights and biases are updated using gradient descent:

W = W - α dW

b = b - α db

where α is the learning rate.
## Network Architecture
This Neural Network comprises of 4 layers and its structure is defined below
- Input Layer : This is the first layer of the network it takes its input from the handwritten image which is 28 X 28 pixel which make 784 Features. All the feature represent intensity of the pixels and they are normalized.
- Layer 1 : This layer has 128 nodes and  uses sigmoid activation.
- Layer 2 : This layer has 64 nodes and uses sigmoid activation.
- Layer 3 : This layer has 10 nodes each node represent digits from 0-9. It uses softmax activation for multiple classification.
In total this Network has 1,09,386 trainable parameters all updated via mini bacth SGD. I also used HE activation for intializing weights.
## Dataset
The project uses the MNIST handwritten digit dataset containing 70,000 grayscale images of digits (0–9).
## Training results :
On training the model on mini batch size of 10 with learning rate 0.1 for 10 epochs, We obtain :
- Accuracy of 97.48%
- Cost for last iteration is 2777.2942

# Future Improvements:
Their are few scopes for future improvements. Currently the Layer 1 and Layer 2 uses sigmoid activation but Relu activation will be more suitable. The model do not uses parrallel computing for training, But using parrallel computing will reduce the training time significantly. Also adding a GUI digit interface will be suitable.
## What I learned :

This project has helped me gain practical Machine Learning expeirence, Though it is not perfect I am really happy to finish it. It has taught me :
- Implementing neural networks from scratch
- Matrix-based computations with NumPy
- Backpropagation mathematics
- Gradient descent optimization
- Debugging training instability
