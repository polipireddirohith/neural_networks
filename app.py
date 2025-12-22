import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons, make_circles, make_blobs
import mininn  # Our custom library

# --- PAGE CONFIG ---
st.set_page_config(page_title="Neural Playground", layout="wide")

st.title("🧠 Neural Network Playground")
st.markdown("""
This is an expanded version of the XOR project. 
It uses a **modular 'MiniNN' library** built from scratch in NumPy to solve various classification problems.
""")

# --- SIDEBAR: DATA & CONFIG ---
st.sidebar.header("1. Data Generation")
dataset_name = st.sidebar.selectbox("Choose Dataset", ["XOR", "Moons", "Concentric Circles", "Blobs"])
noise = st.sidebar.slider("Noise Level", 0.0, 0.5, 0.1)
n_samples = 300

st.sidebar.header("2. Model Architecture")
hidden_neurons = st.sidebar.slider("Hidden Neurons", 2, 20, 6)
layers_count = st.sidebar.radio("Hidden Layers", [1, 2], index=0)
activation_name = st.sidebar.selectbox("Activation Function", ["ReLU", "Tanh", "Sigmoid"])

st.sidebar.header("3. Training Config")
epochs = st.sidebar.slider("Epochs", 1000, 10000, 3000, step=1000)
lr = st.sidebar.selectbox("Learning Rate", [0.001, 0.01, 0.1, 0.5], index=2)

# --- DATA GENERATION ---
def get_data(name, noise):
    if name == "XOR":
        # XOR is manual
        X = np.random.uniform(-1, 1, (n_samples, 2))
        Y = np.logical_xor(X[:, 0] > 0, X[:, 1] > 0).astype(int).reshape(-1, 1)
        return X, Y
    elif name == "Moons":
        X, Y = make_moons(n_samples=n_samples, noise=noise, random_state=42)
    elif name == "Concentric Circles":
        X, Y = make_circles(n_samples=n_samples, noise=noise, factor=0.5, random_state=42)
    elif name == "Blobs":
        X, Y = make_blobs(n_samples=n_samples, centers=2, cluster_std=1.0 + noise, random_state=42)
    
    return X, Y.reshape(-1, 1)

X, Y = get_data(dataset_name, noise)

# --- VISUALIZATION FUNCTION ---
def plot_decision_boundary(model, X, Y):
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    h = 0.05
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    
    # Grid predictions
    grid = np.c_[xx.ravel(), yy.ravel()]
    probs = model.predict(grid)
    probs = probs.reshape(xx.shape)
    
    fig, ax = plt.subplots(figsize=(6, 5))
    contour = ax.contourf(xx, yy, probs, steps=50, cmap="RdBu", alpha=0.6, vmin=0, vmax=1)
    ax.scatter(X[:, 0], X[:, 1], c=Y.ravel(), cmap="RdBu", edgecolors="white", s=30)
    ax.set_title(f"Decision Boundary ({dataset_name})")
    return fig

# --- MAIN UI ---
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("Data Preview")
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.scatter(X[:, 0], X[:, 1], c=Y.ravel(), cmap="RdBu", edgecolors="k")
    st.pyplot(fig)

with col2:
    st.subheader("Model Status")
    start_btn = st.button("Start Training", type="primary")

if start_btn:
    # Build Model
    layers = []
    
    # Input -> Hidden 1
    layers.append(mininn.Dense(2, hidden_neurons))
    if activation_name == "ReLU":
        layers.append(mininn.ReLU())
    elif activation_name == "Tanh":
        layers.append(mininn.Tanh())
    else:
        layers.append(mininn.Sigmoid())
        
    # Hidden 1 -> Hidden 2 (Optional)
    if layers_count == 2:
        layers.append(mininn.Dense(hidden_neurons, hidden_neurons))
        if activation_name == "ReLU":
            layers.append(mininn.ReLU())
        elif activation_name == "Tanh":
            layers.append(mininn.Tanh())
        else:
            layers.append(mininn.Sigmoid())

    # Hidden -> Output
    layers.append(mininn.Dense(hidden_neurons, 1))
    layers.append(mininn.Sigmoid()) # Output is always Sigmoid for binary class
    
    model = mininn.Sequential(layers)
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    chart_placeholder = st.empty()
    
    # Callback to update UI during training
    loss_history = []
    
    # Custom training loop to yield control to UI
    # We copy logic from model.train but make it iterative
    
    # Note: Using the model.train from mininn directly, 
    # but we need to modify mininn to return history or accept callback
    # I implemented a callback support in mininn.Sequential.train!
    
    def ui_callback(epoch, loss):
        loss_history.append(loss)
        if epoch % 500 == 0:
            progress = epoch / epochs
            progress_bar.progress(progress)
            status_text.text(f"Epoch: {epoch}/{epochs} | Loss: {loss:.4f}")
            
            # Live chart update occasionally
            # chart_placeholder.line_chart(loss_history)

    # Train
    with st.spinner("Training..."):
        history = model.train(X, Y, epochs=epochs, learning_rate=lr, callback=ui_callback)
    
    progress_bar.progress(1.0)
    status_text.success("Training Complete!")
    
    # Final Visuals
    st.line_chart(history)
    
    st.subheader("Final Decision Boundary")
    fig_res = plot_decision_boundary(model, X, Y)
    st.pyplot(fig_res)

