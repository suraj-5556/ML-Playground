import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
from sklearn.datasets import make_blobs, make_moons, make_circles
from sklearn.preprocessing import StandardScaler
import time

# --- Configuration and Custom CSS ---
st.set_page_config(
    page_title="ML Playground | Interactive Study Tool",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for Professional Dark Theme with a Teal/Blue Accent
st.markdown("""
<style>
    /* Main Streamlit App Overrides */
    .stApp {
        background-color: #0d1117; /* GitHub Dark Background */
        color: #c9d1d9; /* Light text */
    }
    
    /* Main Header */
    .main-header {
        font-size: 2.8rem;
        color: #20c997; /* Teal Accent */
        text-align: center;
        margin-bottom: 2rem;
        font-weight: 700;
        text-shadow: 1px 1px 3px rgba(0,0,0,0.5);
    }
    
    /* Sidebar Headers */
    .stSidebar .stSelectbox label, .stSidebar .stSlider label, .stSidebar .stNumberInput label {
        font-size: 1rem;
        font-weight: 600;
        color: #c9d1d9;
    }
    .stSidebar h3 {
        color: #20c997; /* Teal Accent for section titles */
        border-bottom: 2px solid #30363d;
        padding-bottom: 10px;
        margin-top: 20px;
    }

    /* Metric Card Styling */
    div[data-testid="stMetric"] {
        background-color: #161b22; /* Slightly lighter dark background */
        border: 1px solid #30363d;
        border-radius: 8px;
        padding: 10px 15px;
        margin-bottom: 10px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    div[data-testid="stMetricLabel"] p {
        color: #20c997 !important; /* Teal Accent for metric label */
        font-size: 1rem;
        font-weight: 600;
    }
    div[data-testid="stMetricValue"] {
        color: #ffffff; /* White value */
        font-weight: 700;
        font-size: 1.8rem;
    }

    /* Button Styling */
    .stButton > button {
        background: linear-gradient(135deg, #20c997, #17a2b8); /* Teal-to-Blue Gradient */
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.6rem 1rem;
        font-weight: bold;
        transition: all 0.3s;
        width: 100%;
    }
    .stButton > button:hover {
        background: linear-gradient(135deg, #17a2b8, #20c997);
        transform: translateY(-1px);
        box-shadow: 0 4px 10px rgba(32, 201, 151, 0.4);
    }
    
    /* Code/Weight Display Box */
    .weights-box {
        background-color: #161b22;
        padding: 15px;
        border-radius: 8px;
        border: 1px solid #30363d;
        font-family: monospace;
        white-space: pre-wrap;
        color: #89e0ff; /* Light blue for code/weights */
    }
    
</style>
""", unsafe_allow_html=True)

# Fixed hyperparameters for SVM
FIXED_LAMBDA = 0.03
FIXED_GAMMA = 3.02

# --- Custom Algorithms (Same as provided, but with a slight fix to the Pegasos loop for clarity) ---
def pegasos_kernel_svm(data: np.ndarray, labels: np.ndarray, kernel='linear', lambda_val=0.03, iterations=100, sigma=1.0) -> tuple:
    # This implementation models the weights as dual coefficients (alpha)
    w = np.zeros(len(labels)) 
    b = 0
    t = 0
    gamma = 1 / (2 * sigma * sigma)
    
    weights_history = []
    loss_history = []
    
    def ker(x, y, kernel):
        if kernel == 'linear':
            return np.dot(x, y)
        elif kernel == 'rbf':
            dist_sq = np.sum((x - y)**2)
            val = np.exp(-gamma * dist_sq)
            return val
    
    def compute_loss(w, b, data, labels):
        loss = 0
        for i in range(len(labels)):
            curr_x = data[i]
            sumation = 0
            for k in range(len(labels)):
                sumation += w[k] * labels[k] * ker(curr_x, data[k], kernel)
            pred = sumation + b
            loss += max(0, 1 - labels[i] * pred)
        return loss / len(labels) + lambda_val * np.sum(w**2) / 2
    
    # Store initial (zero) state
    weights_history.append([w.copy(), b])
    loss_history.append(compute_loss(w, b, data, labels))
    
    for i in range(iterations):
        shuffled_indices = np.arange(len(labels))
        np.random.shuffle(shuffled_indices)
        
        for j_idx in shuffled_indices:
            t += 1
            curr_x = data[j_idx]
            sumation = 0
            for k in range(len(labels)):
                sumation += w[k] * labels[k] * ker(curr_x, data[k], kernel)
            pred = sumation + b
            
            # Pegasos/Online update rule (as per the user's provided logic structure)
            eta_t = 1 / (lambda_val * t)
            
            if labels[j_idx] * pred < 1:
                w[j_idx] += eta_t
                # A simplified or alternative bias update
                b += labels[j_idx] * eta_t * 0.1 
            
        weights_history.append([w.copy(), b])
        loss_history.append(compute_loss(w, b, data, labels))
    
    return w, b, weights_history, loss_history

def train_logreg(X: np.ndarray, y: np.ndarray, learning_rate: float, iterations: int) -> tuple:
    w = np.zeros(len(X[0]))
    b = 0
    loss_array = []
    weights_array = []
    
    def sigmoid(z):
        z = np.clip(z, -250, 250)
        return 1 / (1 + np.exp(-z))
    
    def feed_forward(x, w, b):
        return sigmoid(np.dot(x, w) + b)
    
    def loss(x, y, w, b):
        y_pred = feed_forward(x, w, b)
        y_pred = np.clip(y_pred, 1e-7, 1-1e-7)
        cost = -np.mean(y * np.log(y_pred) + (1 - y) * np.log(1 - y_pred))
        return cost
    
    # Store initial (zero) state
    loss_array.append(round(loss(X, y, w, b), 4))
    weights_array.append([np.round(w.copy(), 3), round(b, 3)])
    
    for i in range(iterations):
        # Calculate gradients (Vectorized)
        y_pred = feed_forward(X, w, b)
        error = y_pred - y
        
        grad_w = np.dot(X.T, error) / len(y)
        grad_b = np.sum(error) / len(y)
        
        # Update parameters
        w = w - learning_rate * grad_w
        b = b - learning_rate * grad_b
        
        loss_array.append(round(loss(X, y, w, b), 4))
        weights_array.append([np.round(w.copy(), 3), round(b, 3)])
    
    result = [round(weight, 3) for weight in w]
    result.insert(0, round(b, 3))
    return result, loss_array, weights_array

# Function to precompute SVM decision boundaries for all steps
def precompute_svm_boundaries(X, y_svm, weights_history, gamma):
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 25), 
                         np.linspace(y_min, y_max, 25))
    
    mesh_points = np.c_[xx.ravel(), yy.ravel()]
    gamma_val = 1 / (2 * (1/gamma) * (1/gamma))
    
    mesh_expanded = mesh_points[:, np.newaxis, :]
    X_expanded = X[np.newaxis, :, :]
    
    dist_sq = np.sum((mesh_expanded - X_expanded)**2, axis=2)
    kernel_matrix = np.exp(-gamma_val * dist_sq)
    
    boundaries = []
    y_svm_array = np.array(y_svm)
    
    for step_weights, step_bias in weights_history:
        step_weights_array = np.array(step_weights)
        max_idx = min(len(step_weights_array), len(X), len(y_svm_array))
        
        weighted_kernels = kernel_matrix[:, :max_idx] * (step_weights_array[:max_idx] * y_svm_array[:max_idx])
        Z = np.sum(weighted_kernels, axis=1) + step_bias
        Z = Z.reshape(xx.shape)
        boundaries.append((xx, yy, Z))
    
    return boundaries

# --- Session State Initialization ---
if 'training_step' not in st.session_state:
    st.session_state.training_step = 0
if 'is_training' not in st.session_state:
    st.session_state.is_training = False
if 'weights_history' not in st.session_state:
    st.session_state.weights_history = []
if 'loss_history' not in st.session_state:
    st.session_state.loss_history = []
if 'svm_boundaries' not in st.session_state:
    st.session_state.svm_boundaries = []
if 'accuracy_cache' not in st.session_state:
    st.session_state.accuracy_cache = []
if 'data_ready' not in st.session_state:
    st.session_state.data_ready = False
if 'X' not in st.session_state:
    st.session_state.X = None
if 'y' not in st.session_state:
    st.session_state.y = None
if 'y_svm' not in st.session_state:
    st.session_state.y_svm = None

# --- Main Title ---
st.markdown('<h1 class="main-header">🧠 Interactive Machine Learning Viz</h1>', unsafe_allow_html=True)

# --- Sidebar Controls (Data and Model Setup) ---
with st.sidebar:
    st.markdown("### 🛠️ Model & Data Setup")

    model_type = st.selectbox(
        "Choose Algorithm",
        ["Logistic Regression", "Pegasos Kernel SVM"],
        key="model_type",
    )
    
    # --- Data Loading Logic ---
    with st.expander("📊 Data Source & Features", expanded=True):
        dataset_type = st.radio(
            "Select Data Source",
            ["Demo Dataset", "Upload CSV"],
            key="dataset_type",
        )
        
        # Placeholder variables for X, y, y_svm
        X, y, y_svm = None, None, None
        data_is_ready = False
        
        if dataset_type == "Demo Dataset":
            # Dataset controls
            demo_dataset = st.selectbox("Choose Demo Dataset", ["blobs", "moons", "circles"], key="demo_dataset")
            n_samples = st.slider("Number of Samples", 50, 500, 200, 10, key="n_samples")
            noise = st.slider("Noise Level", 0.0, 0.5, 0.1, 0.01, key="noise")
            
            # Generate data
            if demo_dataset == "blobs":
                X_raw, y_raw = make_blobs(n_samples=n_samples, centers=2, n_features=2, random_state=42, cluster_std=1.5)
            elif demo_dataset == "moons":
                X_raw, y_raw = make_moons(n_samples=n_samples, noise=noise, random_state=42)
            elif demo_dataset == "circles":
                X_raw, y_raw = make_circles(n_samples=n_samples, noise=noise, factor=0.3, random_state=42)
            
            scaler = StandardScaler()
            X = scaler.fit_transform(X_raw)
            y = y_raw
            y_svm = 2 * y - 1
            data_is_ready = True
            
        else: # Upload CSV
            uploaded_file = st.file_uploader("Upload CSV", type=['csv'])
            if uploaded_file is not None:
                try:
                    df = pd.read_csv(uploaded_file)
                    st.write("Dataset Preview:")
                    st.dataframe(df.head(), use_container_width=True)
                    
                    cols = df.columns.tolist()
                    x1_col = st.selectbox("Select X1 feature", cols, key="x1_col")
                    x2_col = st.selectbox("Select X2 feature", cols, key="x2_col")
                    y_col = st.selectbox("Select Target", cols, key="y_col")
                    
                    if x1_col and x2_col and y_col:
                        X_raw = df[[x1_col, x2_col]].values
                        scaler = StandardScaler()
                        X = scaler.fit_transform(X_raw)
                        
                        y_raw = df[y_col].values
                        unique_labels = np.unique(y_raw)
                        if len(unique_labels) == 2:
                            y = (y_raw == unique_labels[1]).astype(int)
                            y_svm = 2 * y - 1
                            data_is_ready = True
                        else:
                            st.error("Please select a binary target variable.")
                except Exception as e:
                    st.error(f"Error loading CSV: {e}")
            
        # Update session state with current data
        st.session_state.X = X
        st.session_state.y = y
        st.session_state.y_svm = y_svm
        st.session_state.data_ready = data_is_ready
    
    # --- Hyperparameter Controls (Only shown if data is ready) ---
    if st.session_state.data_ready:
        st.markdown("### ⚙️ Hyperparameters")
        
        with st.expander(f"{model_type} Settings", expanded=True):
            if model_type == "Logistic Regression":
                learning_rate = st.number_input("Learning Rate (α)", 0.0001, 0.1, 0.01, 0.0001, format="%.4f", key="lr")
                epochs = st.slider("Epochs (Iterations)", 10, 200, 50, key="epochs")
            else:

                epochs = st.slider("Epochs (Iterations)", 10, 100, 50, key="svm_epochs")

        st.markdown("### 🚀 Training & Animation")
        
        # --- Training Initialization ---
        if st.button("🚀 Train/Retrain Model", key="train_btn"):
            # Reset and prepare for new training
            st.session_state.weights_history = []
            st.session_state.loss_history = []
            st.session_state.svm_boundaries = []
            st.session_state.training_step = 0
            st.session_state.is_training = False
            if 'accuracy_cache' in st.session_state:
                 del st.session_state.accuracy_cache
            
            X, y, y_svm = st.session_state.X, st.session_state.y, st.session_state.y_svm
            
            with st.spinner(f"Training {model_type}..."):
                start_time = time.time()
                if model_type == "Logistic Regression":
                    _, loss_history, weights_history = train_logreg(X, y, learning_rate, epochs)
                    st.session_state.weights_history = weights_history
                    st.session_state.loss_history = loss_history
                else:
                    w, b, weights_history, loss_history = pegasos_kernel_svm(
                        X, y_svm, kernel='rbf', lambda_val=FIXED_LAMBDA,
                        iterations=epochs, sigma=1/FIXED_GAMMA
                    )
                    st.session_state.weights_history = weights_history
                    st.session_state.loss_history = loss_history
                    
                    with st.spinner("Precomputing RBF boundaries..."):
                        st.session_state.svm_boundaries = precompute_svm_boundaries(
                            X, y_svm, st.session_state.weights_history, FIXED_GAMMA
                        )
                
                end_time = time.time()
                st.success(f"Training Complete! ({end_time - start_time:.2f}s)")
        
        # --- Animation Controls ---
        history_len = len(st.session_state.weights_history)
        with st.container(border=True):
            st.markdown("**Animation Controls**")
            col1_anim, col2_anim = st.columns(2)
            with col1_anim:
                if st.button("▶️ Auto-Run", key="start_btn", disabled=history_len <= 1):
                    st.session_state.is_training = True
                if st.button("⏹️ Stop/Pause", key="stop_btn", disabled=history_len <= 1):
                    st.session_state.is_training = False
            
            with col2_anim:
                if st.button("👆 Single Step", key="step_btn", disabled=history_len <= 1):
                    if st.session_state.training_step < history_len - 1:
                        st.session_state.training_step += 1
                if st.button("🔄 Reset View", key="reset_btn", disabled=history_len <= 1):
                    st.session_state.training_step = 0
                    st.session_state.is_training = False
            
            delay = st.slider("Animation Delay (seconds)", 0.05, 1.0, 0.25, 0.05, key="delay")


# --- Main Content Area (Visualization) ---

if not st.session_state.data_ready:
    st.info("👆 **Select a Demo Dataset** or **Upload a CSV** in the sidebar to begin the visualization.")
    st.stop() # Stop execution if data isn't ready

# Use the data from session state
X, y, y_svm = st.session_state.X, st.session_state.y, st.session_state.y_svm
history_len = len(st.session_state.weights_history)

# If no training has occurred, run the training function once to populate the initial state (step 0)
if history_len == 0:
    # Use dummy values for learning rate/epochs as they won't be used past step 0
    if model_type == "Logistic Regression":
        _, loss_history, weights_history = train_logreg(X, y, 0.01, 0)
        st.session_state.weights_history = weights_history
        st.session_state.loss_history = loss_history
    else:
        w, b, weights_history, loss_history = pegasos_kernel_svm(
            X, y_svm, kernel='rbf', lambda_val=FIXED_LAMBDA,
            iterations=0, sigma=1/FIXED_GAMMA
        )
        st.session_state.weights_history = weights_history
        st.session_state.loss_history = loss_history

# Re-fetch history length after potential initial setup
history_len = len(st.session_state.weights_history)
current_step = min(st.session_state.training_step, history_len - 1)


# --- 1. Decision Boundary & Metrics ---
st.markdown("## ⚙️ Real-time Visualization & Performance")

col_viz, col_metrics, col_weights = st.columns([2, 1, 1])

# --- Visualization Column ---
with col_viz:
    st.markdown(f"### {model_type} Decision Boundary")
    
    current_w, current_b = st.session_state.weights_history[current_step]
    current_w_array = np.array(current_w)

    fig = plt.figure(figsize=(8, 6))
    plt.style.use('dark_background')
    
    colors = ['#ff6b6b', '#4ecdc4'] # Red/Teal
    labels = ['Class 0', 'Class 1']
    
    # Plot Data Points
    for i in range(2):
        mask = y == i
        plt.scatter(X[mask, 0], X[mask, 1], c=colors[i], label=labels[i],
                    alpha=0.8, s=60, edgecolors='black', linewidth=0.5)
    
    # Plot Decision Boundary (Only if weights are not all zeros, or it's the LogReg initial state)
    if current_step > 0 or model_type == "Logistic Regression":
        if model_type == "Logistic Regression":
            x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
            y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
            xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100),
                                 np.linspace(y_min, y_max, 100))
            
            mesh_points = np.c_[xx.ravel(), yy.ravel()]
            
            def sigmoid(z):
                z = np.clip(z, -250, 250)
                return 1 / (1 + np.exp(-z))
            
            Z = sigmoid(mesh_points @ current_w_array + current_b)
            Z = Z.reshape(xx.shape)
            
            # Decision boundary (P=0.5)
            plt.contour(xx, yy, Z, levels=[0.5], colors=['#20c997'], linewidths=3, alpha=1.0)
            plt.contourf(xx, yy, Z, levels=[0, 0.5, 1], colors=['#ff6b6b', '#4ecdc4'], alpha=0.15)
        
        else: # Pegasos Kernel SVM
            # Use precomputed boundaries only if available (i.e., after full training)
            if current_step < len(st.session_state.svm_boundaries):
                xx, yy, Z = st.session_state.svm_boundaries[current_step]
                
                # Decision boundary (Z=0)
                plt.contour(xx, yy, Z, levels=[0], colors=['#20c997'], linewidths=3, alpha=1.0)
                # Margin (Z=-1, Z=1)
                plt.contour(xx, yy, Z, levels=[-1, 1], colors=['#888'], linestyles=['--', '--'], linewidths=1.5, alpha=0.7)
                
                # Classification fill
                plt.contourf(xx, yy, Z, levels=[-100, 0, 100], colors=['#ff6b6b', '#4ecdc4'], alpha=0.15)
            else:
                plt.title(f'Feature Space - Initial State (Train to see boundary)', fontsize=14, color='white', pad=15)
    
    plt.title(f'Feature Space - Epoch {current_step} / {history_len - 1}',
              fontsize=14, color='white', pad=15)
    plt.xlabel('Feature 1 (Scaled)', fontsize=12, color='white')
    plt.ylabel('Feature 2 (Scaled)', fontsize=12, color='white')
    plt.legend(fontsize=10, loc='upper right')
    plt.grid(True, alpha=0.15)
    plt.gca().set_aspect('equal', adjustable='box')
    
    st.pyplot(fig)
    plt.close()

# --- Metrics Column ---
with col_metrics:
    st.markdown("### 📊 Metrics")
    
    current_loss = st.session_state.loss_history[current_step] if current_step < history_len else 0
    
    # Calculate Current Accuracy (Vectorized for speed)
    if model_type == "Logistic Regression":
        def sigmoid(z):
            z = np.clip(z, -250, 250)
            return 1 / (1 + np.exp(-z))
        
        y_pred_proba = sigmoid(X @ current_w_array + current_b)
        y_pred = (y_pred_proba > 0.5).astype(int)
        accuracy = np.mean(y_pred == y) * 100
    else:
        # SVM accuracy needs to be calculated only if history exists
        if history_len > 0:
            gamma_val = 1 / (2 * (1/FIXED_GAMMA) * (1/FIXED_GAMMA))
            max_idx = min(len(current_w), len(X), len(y_svm))
            
            X_expanded = X[:, np.newaxis, :]
            X_train_expanded = X[np.newaxis, :max_idx, :]
            dist_sq = np.sum((X_expanded - X_train_expanded)**2, axis=2)
            kernel_vals = np.exp(-gamma_val * dist_sq)
            
            weights_labels = np.array(current_w[:max_idx]) * np.array(y_svm[:max_idx])
            decisions = np.sum(kernel_vals * weights_labels, axis=1) + current_b
            y_pred = (decisions > 0).astype(int)
            accuracy = np.mean(y_pred == y) * 100
        else:
            accuracy = 0.0 # Should not happen with the initial setup logic
            
    st.metric(label="Current Loss", value=f"{current_loss:.4f}", delta_color="inverse")
    st.metric(label="Current Accuracy", value=f"{accuracy:.2f}%", delta_color="normal")
    st.metric(label="Current Epoch", value=f"{current_step}", delta_color="off")

# --- Weights Column ---
with col_weights:
    st.markdown("### 🔢 Parameters")
    st.markdown(f"**Step: {current_step}**")
    
    weights_text = ""
    if model_type == "Logistic Regression":
        weights_text += f"Bias (b): {current_b:.4f}\n"
        weights_text += f"Weight 1 (w₁): {current_w_array[0]:.4f}\n"
        weights_text += f"Weight 2 (w₂): {current_w_array[1]:.4f}\n"
        st.markdown(f"""
        <div class="weights-box">
            {weights_text}
        </div>
        """, unsafe_allow_html=True)
        st.markdown("LogReg Boundary: $w_1 x_1 + w_2 x_2 + b = 0$")
    else: # Pegasos Kernel SVM
        support_vectors = np.count_nonzero(current_w)
        total_data = len(X)
        st.markdown(f"""
        <div class="weights-box">
            Bias (b): {current_b:.4f}\n
            Support Vectors: {support_vectors}\n
            Total Data Points: {total_data}
        </div>
        """, unsafe_allow_html=True)
        st.markdown("Kernel SVM Boundary: $\\sum_i \\alpha_i y_i K(x, x_i) + b = 0$")


st.markdown("---")

# --- 2. Loss & Accuracy Curves ---
st.markdown("## 📈 Training History")
col_loss, col_acc = st.columns(2)

# --- Loss Curve ---
with col_loss:
    loss_data = st.session_state.loss_history[:current_step + 1]
    epochs_data = list(range(len(loss_data)))
    
    fig_loss = go.Figure()
    fig_loss.add_trace(go.Scatter(
        x=epochs_data, y=loss_data,
        mode='lines',
        name='Loss',
        line=dict(color='#ff6b6b', width=3),
    ))
    
    fig_loss.update_layout(
        title="Loss Curve (Hinge or BCE)",
        xaxis_title="Epoch",
        yaxis_title="Loss Value",
        template="plotly_dark",
        height=350,
        showlegend=False,
        title_font_color="#c9d1d9"
    )
    st.plotly_chart(fig_loss, use_container_width=True)

# --- Accuracy Curve ---
with col_acc:
    # --- Accuracy Cache Computation (Same logic as before, but only runs after full training) ---
    if history_len > 1 and ('accuracy_cache' not in st.session_state or len(st.session_state.accuracy_cache) != history_len):
         with st.spinner("Calculating full accuracy history..."):
            accuracy_history = []
            for step in range(history_len):
                step_weights, step_bias = st.session_state.weights_history[step]
                
                if model_type == "Logistic Regression":
                    step_w = np.array(step_weights)
                    y_pred_proba = sigmoid(X @ step_w + step_bias)
                    y_pred = (y_pred_proba > 0.5).astype(int)
                    step_accuracy = np.mean(y_pred == y) * 100
                else:
                    gamma_val = 1 / (2 * (1/FIXED_GAMMA) * (1/FIXED_GAMMA))
                    max_idx = min(len(step_weights), len(X), len(y_svm))
                    
                    X_expanded = X[:, np.newaxis, :]
                    X_train_expanded = X[np.newaxis, :max_idx, :]
                    dist_sq = np.sum((X_expanded - X_train_expanded)**2, axis=2)
                    kernel_vals = np.exp(-gamma_val * dist_sq)
                    
                    weights_labels = np.array(step_weights[:max_idx]) * np.array(y_svm[:max_idx])
                    decisions = np.sum(kernel_vals * weights_labels, axis=1) + step_bias
                    y_pred = (decisions > 0).astype(int)
                    step_accuracy = np.mean(y_pred == y) * 100
                
                accuracy_history.append(step_accuracy)
            
            st.session_state.accuracy_cache = accuracy_history
    
    # Use cached accuracy data up to current step
    accuracy_data = st.session_state.accuracy_cache[:current_step + 1] if history_len > 1 else [accuracy]
    epochs_data = list(range(len(accuracy_data)))
    
    fig_acc = go.Figure()
    fig_acc.add_trace(go.Scatter(
        x=epochs_data, y=accuracy_data,
        mode='lines',
        name='Accuracy',
        line=dict(color='#4ecdc4', width=3),
    ))
    
    fig_acc.update_layout(
        title="Accuracy Curve",
        xaxis_title="Epoch",
        yaxis_title="Accuracy (%)",
        template="plotly_dark",
        height=350,
        yaxis=dict(range=[max(0, min(accuracy_data) * 0.95) if accuracy_data else 0, 100]),
        showlegend=False,
        title_font_color="#c9d1d9"
    )
    
    st.plotly_chart(fig_acc, use_container_width=True)

# --- Auto-advance animation ---
if st.session_state.is_training and st.session_state.training_step < history_len - 1:
    time.sleep(delay)
    st.session_state.training_step += 1
    st.rerun()
elif st.session_state.is_training and st.session_state.training_step >= history_len - 1:
    st.session_state.is_training = False
    st.toast("Animation finished!", icon="🎉")

# --- Footer ---
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #888; padding: 10px; font-size: 0.9em;'>
    **ML Playground** - Interactive Machine Learning Visualization | A Study Tool<br>
    Built with Streamlit, NumPy, Matplotlib & Plotly | Designed for Educational Exploration
</div>
""", unsafe_allow_html=True)
