import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons, make_circles, make_blobs, load_breast_cancer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
import mininn  # Our custom library

# --- PAGE CONFIG ---
st.set_page_config(
    page_title="Neural Cortex | Advanced AI Playground",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- CUSTOM CSS ---
st.markdown("""
<style>
    /* Global Styles */
    .stApp {
        background-color: #0E1117;
    }
    h1, h2, h3 {
        font-family: 'Inter', sans-serif;
        color: #FFFFFF;
    }
    .stButton>button {
        width: 100%;
        border-radius: 8px;
        font-weight: 600;
        background: linear-gradient(90deg, #4b6cb7 0%, #182848 100%);
        border: none;
        color: white;
    }
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(75, 108, 183, 0.5);
    }
    /* Metric Cards */
    div[data-testid="stMetricValue"] {
        font-size: 28px;
        color: #4b6cb7;
    }
</style>
""", unsafe_allow_html=True)

# --- HELPER FUNCTIONS ---
def generate_synthetic_data(name, noise, n_samples=500):
    if name == "XOR":
        X = np.random.uniform(-1, 1, (n_samples, 2))
        Y = np.logical_xor(X[:, 0] > 0, X[:, 1] > 0).astype(int).reshape(-1, 1)
    elif name == "Moons":
        X, Y = make_moons(n_samples=n_samples, noise=noise, random_state=42)
        Y = Y.reshape(-1, 1)
    elif name == "Concentric Circles":
        X, Y = make_circles(n_samples=n_samples, noise=noise, factor=0.5, random_state=42)
        Y = Y.reshape(-1, 1)
    elif name == "Blobs":
        X, Y = make_blobs(n_samples=n_samples, centers=2, cluster_std=1.0 + noise, random_state=42)
        Y = Y.reshape(-1, 1)
    return X, Y

def load_real_data():
    data = load_breast_cancer()
    X = data.data
    Y = data.target.reshape(-1, 1)
    # Scale Data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    # PCA for Visualization (2D)
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    return X_pca, Y

def plot_decision_boundary(model, X, Y, title):
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    h = 0.05
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    
    grid = np.c_[xx.ravel(), yy.ravel()]
    probs = model.predict(grid)
    probs = probs.reshape(xx.shape)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    # Dark Theme Plot
    ax.set_facecolor('#0E1117')
    fig.patch.set_facecolor('#0E1117')
    
    contour = ax.contourf(xx, yy, probs, steps=50, cmap="RdBu", alpha=0.6, vmin=0, vmax=1)
    
    # Scatter points
    scatter = ax.scatter(X[:, 0], X[:, 1], c=Y.ravel(), cmap="RdBu", edgecolors="white", s=40, alpha=0.9)
    
    ax.set_title(title, color='white', fontsize=14)
    ax.tick_params(colors='white')
    ax.spines['bottom'].set_color('white')
    ax.spines['left'].set_color('white')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    return fig

# --- SIDEBAR CONFIG ---
st.sidebar.title("Configuration")

with st.sidebar.expander("🛠️ Model Architecture", expanded=True):
    hidden_neurons = st.slider("Hidden Neurons", 2, 64, 8)
    layers_count = st.radio("Total Hidden Layers", [1, 2, 3], index=0)
    activation_name = st.selectbox("Activation Function", ["ReLU", "Tanh", "Sigmoid"])
    
with st.sidebar.expander("⚙️ Hyperparameters", expanded=True):
    epochs = st.slider("Epochs", 500, 5000, 2000, step=500)
    lr = st.select_slider("Learning Rate", options=[0.0001, 0.001, 0.01, 0.05, 0.1, 0.5], value=0.01)

# --- MAIN LAYOUT ---
st.title("🧠 Neural Cortex")
st.markdown("### Interactive Neural Network Studio")

# TABS
tab_playground, tab_realworld = st.tabs(["🧬 Synthetic Playground", "🏥 Real-World Diagnosis (Cancer)"])

# === TAB 1: SYNTHETIC ===
with tab_playground:
    col_data, col_viz = st.columns([1, 2])
    
    with col_data:
        st.markdown("#### 1. Dataset Selection")
        dataset_name = st.selectbox("Choose Shape", ["XOR", "Moons", "Concentric Circles", "Blobs"])
        noise = st.slider("Noise Level", 0.0, 0.5, 0.1)
        
        X_syn, Y_syn = generate_synthetic_data(dataset_name, noise)
        
        st.info(f"Loaded {len(X_syn)} samples.")
        
        if st.button("🚀 Train Synthetic Model", use_container_width=True):
            # Model Build
            layers = []
            layers.append(mininn.Dense(2, hidden_neurons))
            act_class = getattr(mininn, activation_name)
            layers.append(act_class())
            
            for _ in range(layers_count - 1):
                layers.append(mininn.Dense(hidden_neurons, hidden_neurons))
                layers.append(act_class())
            
            layers.append(mininn.Dense(hidden_neurons, 1))
            layers.append(mininn.Sigmoid())
            
            model = mininn.Sequential(layers)
            
            # Training Widgets
            progress = st.progress(0)
            status = st.empty()
            loss_chart = st.empty()
            acc_chart = st.empty()
            
            history_loss = []
            history_acc = []
            
            def cb(e, l, a):
                history_loss.append(l)
                history_acc.append(a)
                if e % (epochs // 20) == 0:
                    progress.progress(e / epochs)
                    status.markdown(f"**Epoch {e}** | Loss: `{l:.4f}` | Accuracy: `{a:.2%}`")
                    loss_chart.area_chart(pd.DataFrame({"Loss": history_loss}), height=150)
            
            model.train(X_syn, Y_syn, epochs, lr, callback=cb)
            
            progress.progress(1.0)
            status.success("Training Complete!")
            
            # Result
            acc = model.evaluate(X_syn, Y_syn)
            c1, c2 = st.columns(2)
            c1.metric("Final Loss", f"{history_loss[-1]:.4f}")
            c2.metric("Final Accuracy", f"{acc:.2%}")
            
            with col_viz:
                st.markdown("#### Decision Boundary")
                fig = plot_decision_boundary(model, X_syn, Y_syn, "Model Prediction Landscape")
                st.pyplot(fig)

# === TAB 2: REAL WORLD ===
with tab_realworld:
    st.markdown("""
    #### Breast Cancer Prediction (Wisconsin Diagnostic Dataset)
    This scenario uses **real measurements** from digitized images of a fine needle aspirate (FNA) of a breast mass. 
    The goal is to classify the mass as **Malignant** (Cancerous) or **Benign**.
    
    *Note: Dimensionality reduced to 2D via PCA for visualization.*
    """)
    
    X_real, Y_real = load_real_data()
    X_train, X_test, Y_train, Y_test = train_test_split(X_real, Y_real, test_size=0.2, random_state=42)
    
    col_real_stats, col_real_viz = st.columns([1, 2])
    
    with col_real_stats:
        st.metric("Total Samples", len(X_real))
        st.metric("Training Features", "30 (Reduced to 2)")
        
        if st.button("🧪 Train Medical AI", use_container_width=True):
             # Model Build (Same architecture for simplicity)
            layers = []
            # Input is still 2 because of PCA
            layers.append(mininn.Dense(2, hidden_neurons)) 
            act_class = getattr(mininn, activation_name)
            layers.append(act_class())
            
            for _ in range(layers_count - 1):
                layers.append(mininn.Dense(hidden_neurons, hidden_neurons))
                layers.append(act_class())
            
            layers.append(mininn.Dense(hidden_neurons, 1))
            layers.append(mininn.Sigmoid())
            
            model_med = mininn.Sequential(layers)
            
            # Training
            p_bar = st.progress(0)
            sts = st.empty()
            chart = st.empty()
            
            hist_loss_med = []
            
            def cb_med(e, l, a):
                hist_loss_med.append(l)
                if e % (epochs // 10) == 0:
                    p_bar.progress(e/epochs)
                    sts.text(f"Training... Loss: {l:.4f}")
            
            model_med.train(X_train, Y_train, epochs, lr, callback=cb_med)
            p_bar.progress(1.0)
            sts.success("Diagnosis Model Trained")
            
            chart.line_chart(hist_loss_med)
            
            # Evaluation
            test_acc = model_med.evaluate(X_test, Y_test)
            st.metric("Test Set Accuracy", f"{test_acc:.2%}", delta="Generalization Score")
            
            with col_real_viz:
                st.markdown("#### Diagnostic Boundary")
                fig_med = plot_decision_boundary(model_med, X_real, Y_real, "Benign vs Malignant Classification")
                st.pyplot(fig_med)

st.markdown("---")
st.markdown("*Built with Python, MiniNN (Custom), & Streamlit | 2024*")
