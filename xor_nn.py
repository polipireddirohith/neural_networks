import numpy as np
import matplotlib.pyplot as plt

# --- STEP 1: PREPARE DATA [1] ---
# Inputs (Logical Gates)
X = np.array([[0, 0],
              [0, 1],
              [1, 0],
              [1, 1]])

# Expected Outputs (XOR Logic: 1 if inputs are different, 0 if same)
y = np.array([[0],
              [1],
              [1],
              [0]])

# --- STEP 2: INITIALISE PARAMETERS [1] ---
np.random.seed(42)
# Layer 1: 2 inputs -> 4 hidden neurons
W1 = np.random.randn(2, 4)
b1 = np.zeros((1, 4))
# Layer 2: 4 hidden neurons -> 1 output
W2 = np.random.randn(4, 1)
b2 = np.zeros((1, 1))

# --- STEP 3: ACTIVATION FUNCTIONS [1] ---
def relu(z):
    return np.maximum(0, z)

def relu_derivative(z):
    return (z > 0).astype(float)

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# --- STEP 5: LOSS FUNCTION [2] ---
def binary_cross_entropy(y, y_hat):
    epsilon = 1e-15
    y_hat = np.clip(y_hat, epsilon, 1 - epsilon)
    return -np.mean(y * np.log(y_hat) + (1 - y) * np.log(1 - y_hat))

# --- TRAINING LOOP ---
epochs = 10000 
lr = 0.1 

print("Training started...")

for i in range(epochs):
    # --- STEP 4: FORWARD PROPAGATION ---
    Z1 = X @ W1 + b1
    A1 = relu(Z1)
    Z2 = A1 @ W2 + b2
    y_hat = sigmoid(Z2)
    
    # Monitor progress
    if i % 1000 == 0:
        loss = binary_cross_entropy(y, y_hat)
        print(f"Epoch {i}, Loss: {loss:.4f}")

    # --- STEP 6: BACKPROPAGATION ---
    m = y.shape[0]
    
    dZ2 = y_hat - y
    dW2 = (A1.T @ dZ2) / m
    db2 = np.mean(dZ2, axis=0, keepdims=True)
    
    dA1 = dZ2 @ W2.T
    dZ1 = dA1 * relu_derivative(Z1)
    dW1 = (X.T @ dZ1) / m
    db1 = np.mean(dZ1, axis=0, keepdims=True)

    # --- STEP 7: UPDATE WEIGHTS ---
    W1 -= lr * dW1
    b1 -= lr * db1
    W2 -= lr * dW2
    b2 -= lr * db2

# --- FINAL PREDICTION ---
print("\nFinal Predictions:")
print(y_hat)

# --- VISUALIZATION ---
print("\nGenerating decision boundary plot...")

# 1. Create a meshgrid
x_min, x_max = -0.5, 1.5
y_min, y_max = -0.5, 1.5
h = 0.01
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

# 2. Predict for the entire grid
grid_input = np.c_[xx.ravel(), yy.ravel()]

# Forward Prop on grid
Z1_grid = grid_input @ W1 + b1
A1_grid = relu(Z1_grid)
Z2_grid = A1_grid @ W2 + b2
y_hat_grid = sigmoid(Z2_grid)

# Reshape predictions back to grid shape
Z_out = y_hat_grid.reshape(xx.shape)

# 3. Plot
plt.figure(figsize=(10, 8))
# Contour plot for decision boundary
contour = plt.contourf(xx, yy, Z_out, levels=50, cmap="RdBu", alpha=0.8, vmin=0, vmax=1)
plt.colorbar(contour, label='Prediction Probability')

# Scatter plot for original data points
# Class 0: (0,0), (1,1) -> Red
# Class 1: (0,1), (1,0) -> Blue
plt.scatter([0, 1], [0, 1], c='red', s=100, edgecolors='white', linewidth=2, label='Target: 0 (Same)')
plt.scatter([0, 1], [1, 0], c='blue', s=100, edgecolors='white', linewidth=2, label='Target: 1 (Diff)')

plt.title("XOR Neural Network Decision Boundary")
plt.xlabel("Input 1")
plt.ylabel("Input 2")
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig("xor_decision_boundary.png")
print("Plot saved to xor_decision_boundary.png")
