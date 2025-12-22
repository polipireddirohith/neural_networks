# Neural Network Playground 🧠

An interactive educational tool to visualize how Neural Networks learn, built from scratch using NumPy and Streamlit.

![Interface Preview](interface_preview.png)

## Features

- **Modular MiniNN Library**: A custom-built neural network library (`mininn.py`) with:
  - Dense Layers
  - Activation Functions (ReLU, Sigmoid, Tanh)
  - Backpropagation engine
- **Interactive Playground**:
  - Choose datasets: XOR, Moons, Concentric Circles, Blobs.
  - Adjust noise levels.
  - Configure architecture (Hidden Neurons, Layers).
  - Watch the Decision Boundary evolve in real-time!

## Installation & Running

1. Clone the repository:
   ```bash
   git clone https://github.com/polipireddirohith/neural_networks.git
   cd neural_networks
   ```

2. Install dependencies:
   ```bash
   pip install numpy matplotlib streamlit scikit-learn
   ```

3. Run the application:
   ```bash
   streamlit run app.py
   ```

## Project Structure

- `mininn.py`: The core neural network library (Manual implementation).
- `app.py`: The Streamlit frontend interface.
- `xor_nn.py`: The original standalone script for solving XOR.
