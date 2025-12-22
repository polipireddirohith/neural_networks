from typing import List, Callable, Union, Tuple
import numpy as np

class Layer:
    """Base class for neural network layers."""
    def forward(self, input_data: np.ndarray) -> np.ndarray:
        raise NotImplementedError
    
    def backward(self, output_gradient: np.ndarray, learning_rate: float) -> np.ndarray:
        raise NotImplementedError

class Dense(Layer):
    """Fully connected layer with He initialization."""
    def __init__(self, input_size: int, output_size: int):
        # He initialization: random normals * sqrt(2/n)
        self.weights = np.random.randn(input_size, output_size) * np.sqrt(2. / input_size)
        self.bias = np.zeros((1, output_size))
        self.input = None
        self.weights_grad = None
        self.bias_grad = None
    
    def forward(self, input_data: np.ndarray) -> np.ndarray:
        self.input = input_data
        return np.dot(input_data, self.weights) + self.bias
    
    def backward(self, output_gradient: np.ndarray, learning_rate: float) -> np.ndarray:
        # Compute gradients
        # Gradient of Loss w.r.t Weights = Input^T * Gradient_Output
        self.weights_grad = np.dot(self.input.T, output_gradient) / self.input.shape[0]
        # Gradient of Loss w.r.t Bias = sum(Gradient_Output)
        self.bias_grad = np.mean(output_gradient, axis=0, keepdims=True)
        
        # Gradient of Loss w.r.t Input (to pass to prev layer) = Gradient_Output * Weights^T
        input_gradient = np.dot(output_gradient, self.weights.T)
        
        # Update parameters (Gradient Descent)
        self.weights -= learning_rate * self.weights_grad
        self.bias -= learning_rate * self.bias_grad
        
        return input_gradient

class Activation(Layer):
    """Base activation layer wrapper."""
    def __init__(self, activation: Callable, activation_prime: Callable):
        self.activation = activation
        self.activation_prime = activation_prime
        self.input = None
        
    def forward(self, input_data: np.ndarray) -> np.ndarray:
        self.input = input_data
        return self.activation(input_data)
    
    def backward(self, output_gradient: np.ndarray, learning_rate: float) -> np.ndarray:
        # Element-wise multiplication
        return output_gradient * self.activation_prime(self.input)

class ReLU(Activation):
    """Rectified Linear Unit activation."""
    def __init__(self):
        super().__init__(
            lambda x: np.maximum(0, x),
            lambda x: (x > 0).astype(float)
        )

class Sigmoid(Activation):
    """Sigmoid activation for binary classification probabilities."""
    def __init__(self):
        super().__init__(
            lambda x: 1 / (1 + np.exp(-x)),
            # Derivative: s * (1 - s)
            lambda x: (1 / (1 + np.exp(-x))) * (1 - (1 / (1 + np.exp(-x))))
        )

class Tanh(Activation):
    """Hyperbolic Tangent activation."""
    def __init__(self):
        super().__init__(
            lambda x: np.tanh(x),
            lambda x: 1 - np.tanh(x) ** 2
        )

class Sequential:
    """Linear stack of layers."""
    def __init__(self, layers: List[Layer]):
        self.layers = layers
        
    def predict(self, input_data: np.ndarray) -> np.ndarray:
        """Forward pass through all layers."""
        output = input_data
        for layer in self.layers:
            output = layer.forward(output)
        return output
    
    def train(self, x_train: np.ndarray, y_train: np.ndarray, 
              epochs: int, learning_rate: float, 
              callback: Callable = None, 
              verbose: bool = False) -> List[float]:
        
        loss_history = []
        
        for e in range(epochs):
            # 1. Forward Pass
            output = self.predict(x_train)
            
            # 2. Compute Loss
            # Using Binary Cross Entropy + epsilon for stability
            
            # Clip output to prevent log(0)
            output_clipped = np.clip(output, 1e-15, 1 - 1e-15)
            loss = -np.mean(y_train * np.log(output_clipped) + (1 - y_train) * np.log(1 - output_clipped))
            loss_history.append(loss)
            
            # 3. Compute Gradient of Loss w.r.t Output
            # dLoss/dOutput = -(y/y_hat - (1-y)/(1-y_hat))
            #               = (y_hat - y) / (y_hat * (1 - y_hat))
            # We divide by N here to average the gradients over the batch
            grad = (output_clipped - y_train) / (output_clipped * (1 - output_clipped)) 
            
            # 4. Backward Pass
            for layer in reversed(self.layers):
                grad = layer.backward(grad, learning_rate)
            
            if callback:
                # Calculate accuracy for callback
                predictions = (output > 0.5).astype(int)
                accuracy = np.mean(predictions == y_train)
                callback(e, loss, accuracy)
            
            if verbose and (e % 100 == 0):
                print(f"Epoch {e}, Loss: {loss:.4f}")
                
        return loss_history

    def evaluate(self, x_test: np.ndarray, y_test: np.ndarray) -> float:
        output = self.predict(x_test)
        predictions = (output > 0.5).astype(int)
        accuracy = np.mean(predictions == y_test)
        return accuracy
