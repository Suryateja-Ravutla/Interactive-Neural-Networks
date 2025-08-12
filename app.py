
import time
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

import streamlit as st
import plotly.graph_objects as go

q = st.query_params
if q.get("ping", ["0"])[0] == "1":
    st.write("OK")
    st.stop()
    

# -----------------------
# Utility helpers
# -----------------------
def set_seeds(seed: int = 42):
    np.random.seed(seed)
    torch.manual_seed(seed)


@dataclass
class DataBundle:
    x: np.ndarray
    y: np.ndarray
    y_clean: np.ndarray
    x_train: np.ndarray
    y_train: np.ndarray
    x_val: np.ndarray
    y_val: np.ndarray
    x_test: np.ndarray
    y_test: np.ndarray
    x_grid: np.ndarray
    y_clean_grid: np.ndarray
    x_mean: float
    x_std: float
    y_mean: float
    y_std: float


def true_function(x: np.ndarray, kind: str) -> np.ndarray:
    """Return the clean (noise-free) function values for a given kind."""
    if kind == "Sine (sin(2πx))":
        return np.sin(2 * np.pi * x)
    if kind == "Sine (sin(3x) + 0.5 sin(7x))":
        return np.sin(3 * x) + 0.5 * np.sin(7 * x)
    if kind == "Polynomial (x^3 - 0.5x^2 + 0.2x)":
        return x**3 - 0.5 * x**2 + 0.2 * x
    if kind == "Gaussian bump":
        return np.exp(-((x - 0.2) ** 2) / 0.01) - 0.7 * np.exp(-((x - 0.75) ** 2) / 0.02)
    if kind == "Piecewise (abs(x - 0.5))":
        return np.abs(x - 0.5)
    if kind == "Rational (x^2 + 10x) / (1 + x^2)":
        return (x**2 + 10 * x) / (1 + x**2)
    # default (linear-ish)
    return 2 * x - 1


def generate_data(
    n_points: int,
    x_range: Tuple[float, float],
    noise_std: float,
    kind: str,
    seed: int,
    normalize_x: bool,
    normalize_y: bool,
    splits: Tuple[float, float, float],
) -> DataBundle:
    set_seeds(seed)
    x = np.random.uniform(x_range[0], x_range[1], size=n_points)
    x = np.sort(x)
    y_clean = true_function(x, kind)
    noise = np.random.normal(0.0, noise_std, size=n_points)
    y = y_clean + noise

    # train/val/test split
    idx = np.arange(n_points)
    np.random.shuffle(idx)
    n_train = int(splits[0] * n_points)
    n_val = int(splits[1] * n_points)
    train_idx = idx[:n_train]
    val_idx = idx[n_train : n_train + n_val]
    test_idx = idx[n_train + n_val :]

    x_train, y_train = x[train_idx], y[train_idx]
    x_val, y_val = x[val_idx], y[val_idx]
    x_test, y_test = x[test_idx], y[test_idx]

    x_mean = x_train.mean() if normalize_x else 0.0
    x_std = x_train.std() if normalize_x else 1.0
    if normalize_x and x_std == 0:
        x_std = 1.0

    y_mean = y_train.mean() if normalize_y else 0.0
    y_std = y_train.std() if normalize_y else 1.0
    if normalize_y and y_std == 0:
        y_std = 1.0

    # grid for smooth curves
    x_grid = np.linspace(x_range[0], x_range[1], 400)
    y_clean_grid = true_function(x_grid, kind)

    return DataBundle(
        x=x, y=y, y_clean=y_clean,
        x_train=x_train, y_train=y_train,
        x_val=x_val, y_val=y_val,
        x_test=x_test, y_test=y_test,
        x_grid=x_grid, y_clean_grid=y_clean_grid,
        x_mean=x_mean, x_std=x_std, y_mean=y_mean, y_std=y_std
    )


class MLP(nn.Module):
    def __init__(self, in_dim: int, hidden_layers: List[int], activation: str, out_dim: int = 1):
        super().__init__()
        acts = {
            "ReLU": nn.ReLU,
            "Tanh": nn.Tanh,
            "Sigmoid": nn.Sigmoid,
            "GELU": nn.GELU,
            "LeakyReLU": nn.LeakyReLU,
            "ELU": nn.ELU,
        }
        Act = acts.get(activation, nn.ReLU)

        layers = []
        prev = in_dim
        for h in hidden_layers:
            layers.append(nn.Linear(prev, h))
            layers.append(Act())
            prev = h
        layers.append(nn.Linear(prev, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def build_model(hidden_spec: str, default_units: int, activation: str) -> List[int]:
    """Parse hidden layer spec like '64,64,32' or use count * default_units."""
    hidden_spec = hidden_spec.strip()
    if hidden_spec:
        try:
            layers = [int(s) for s in hidden_spec.split(",") if s.strip()]
            layers = [max(1, int(v)) for v in layers]
            return layers
        except Exception:
            pass  # fallback to default
    return [default_units] if default_units > 0 else []


def train_once(
    data: DataBundle,
    hidden_layers: List[int],
    activation: str,
    optimizer_name: str,
    lr: float,
    weight_decay: float,
    batch_size: int,
    epochs: int,
    patience: Optional[int] = None,
    seed: int = 42,
) -> Dict:
    set_seeds(seed)

    # Normalization helpers
    def nx(a): return (a - data.x_mean) / data.x_std
    def ny(a): return (a - data.y_mean) / data.y_std
    def iy(a): return a * data.y_std + data.y_mean

    # tensors
    xtr = torch.tensor(nx(data.x_train), dtype=torch.float32).unsqueeze(1)
    ytr = torch.tensor(ny(data.y_train), dtype=torch.float32).unsqueeze(1)
    xv = torch.tensor(nx(data.x_val), dtype=torch.float32).unsqueeze(1)
    yv = torch.tensor(ny(data.y_val), dtype=torch.float32).unsqueeze(1)
    xte = torch.tensor(nx(data.x_test), dtype=torch.float32).unsqueeze(1)
    yte = torch.tensor(ny(data.y_test), dtype=torch.float32).unsqueeze(1)

    model = MLP(1, hidden_layers, activation, 1)

    # optimizer
    if optimizer_name == "Adam":
        opt = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif optimizer_name == "SGD":
        opt = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay)
    elif optimizer_name == "RMSprop":
        opt = optim.RMSprop(model.parameters(), lr=lr, weight_decay=weight_decay)
    else:
        opt = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    loss_fn = nn.MSELoss()

    # data loaders
    train_dl = DataLoader(TensorDataset(xtr, ytr), batch_size=batch_size, shuffle=True)

    history = {"train_loss": [], "val_loss": []}
    best_val = float("inf")
    best_state = None
    bad_epochs = 0
    start_time = time.time()

    for epoch in range(1, epochs + 1):
        model.train()
        running = 0.0
        for xb, yb in train_dl:
            opt.zero_grad(set_to_none=True)
            pred = model(xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            opt.step()
            running += loss.item() * xb.size(0)

        train_loss = running / len(train_dl.dataset)
        model.eval()
        with torch.no_grad():
            val_pred = model(xv)
            val_loss = loss_fn(val_pred, yv).item()

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)

        # early stopping
        if patience is not None:
            if val_loss < best_val - 1e-8:
                best_val = val_loss
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                bad_epochs = 0
            else:
                bad_epochs += 1
                if bad_epochs >= patience:
                    break
        else:
            # track best even without early stopping
            if val_loss < best_val:
                best_val = val_loss
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    # load best
    if best_state is not None:
        model.load_state_dict(best_state)

    train_time = time.time() - start_time

    # predictions
    model.eval()
    with torch.no_grad():
        ytr_pred = iy(model(xtr).squeeze(1).cpu().numpy())
        yv_pred = iy(model(xv).squeeze(1).cpu().numpy())
        yte_pred = iy(model(xte).squeeze(1).cpu().numpy())
        xg = torch.tensor((data.x_grid - data.x_mean) / data.x_std, dtype=torch.float32).unsqueeze(1)
        yg_pred_grid = iy(model(xg).squeeze(1).cpu().numpy())

    # metrics in original scale
    def metrics(y_true, y_pred):
        return dict(
            MSE=float(mean_squared_error(y_true, y_pred)),
            MAE=float(mean_absolute_error(y_true, y_pred)),
            R2=float(r2_score(y_true, y_pred))
        )

    metrics_out = {
        "Train": metrics(data.y_train, ytr_pred),
        "Val": metrics(data.y_val, yv_pred),
        "Test": metrics(data.y_test, yte_pred),
    }

    # param count
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    return dict(
        model=model,
        history=history,
        metrics=metrics_out,
        train_time=train_time,
        params=params,
        preds={
            "train": (data.x_train, ytr_pred),
            "val": (data.x_val, yv_pred),
            "test": (data.x_test, yte_pred),
            "grid": (data.x_grid, yg_pred_grid),
        },
    )


def make_plot(data: DataBundle, preds: Dict, show_true: bool = True) -> go.Figure:
    fig = go.Figure()

    # scatter: train/val/test
    fig.add_trace(go.Scatter(x=data.x_train, y=data.y_train, mode="markers", name="Train"))
    fig.add_trace(go.Scatter(x=data.x_val, y=data.y_val, mode="markers", name="Val"))
    fig.add_trace(go.Scatter(x=data.x_test, y=data.y_test, mode="markers", name="Test"))

    # predicted smooth curve
    xg, yg = preds["grid"]
    fig.add_trace(go.Scatter(x=xg, y=yg, mode="lines", name="NN prediction"))

    # true function
    if show_true:
        fig.add_trace(go.Scatter(x=data.x_grid, y=data.y_clean_grid, mode="lines", name="True function"))

    fig.update_layout(
        xaxis_title="x",
        yaxis_title="y",
        hovermode="x",
        legend_title_text="Legend",
        margin=dict(l=10, r=10, t=30, b=10),
        height=500
    )
    return fig


def make_loss_plot(history: Dict) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(y=history["train_loss"], mode="lines", name="Train loss"))
    fig.add_trace(go.Scatter(y=history["val_loss"], mode="lines", name="Val loss"))
    fig.update_layout(
        xaxis_title="Epoch",
        yaxis_title="Loss (MSE)",
        hovermode="x",
        margin=dict(l=10, r=10, t=30, b=10),
        height=400
    )
    return fig


# -----------------------
# Streamlit UI
# -----------------------
st.set_page_config(page_title="1D Function Fitter (PyTorch)", layout="wide")
st.title("1D Random Function → Neural Network Model")
st.write(
    "Generate synthetic 1D data from a chosen function, then fit a PyTorch MLP. "
    "Tweak layers, activations, optimizer, and training settings; view predictions and metrics."
)

# Sidebar controls
with st.sidebar:
    st.header("Data")
    func_kind = st.selectbox(
        "Synthetic function",
        [
            "Sine (sin(2πx))",
            "Sine (sin(3x) + 0.5 sin(7x))",
            "Polynomial (x^3 - 0.5x^2 + 0.2x)",
            "Gaussian bump",
            "Piecewise (abs(x - 0.5))",
            "Rational (x^2 + 10x) / (1 + x^2)",
            "Linear-ish (2x - 1)",
        ],
        index=0,
    )
    n_points = st.slider("Number of points", 50, 5000, 400, step=50)
    x_min, x_max = st.slider("x range", -2.0, 2.0, (-1.0, 1.0), step=0.1)
    noise_std = st.slider("Noise std", 0.0, 1.0, 0.1, step=0.01)
    seed = st.number_input("Random seed", min_value=0, max_value=999999, value=42, step=1)

    st.subheader("Splits")
    c1, c2, c3 = st.columns(3)
    with c1:
        train_frac = st.slider("Train %", 10, 90, 70, step=5)
    with c2:
        val_frac = st.slider("Val %", 0, 80, 15, step=5)
    with c3:
        test_frac = 100 - train_frac - val_frac
        st.write(f"Test %: **{test_frac}**")
    if train_frac + val_frac > 100:
        st.error("Train + Val cannot exceed 100%. Adjust sliders.")
    normalize_x = st.checkbox("Normalize x", value=True)
    normalize_y = st.checkbox("Normalize y", value=True)

    st.header("Model")
    hidden_spec = st.text_input("Hidden layers (comma-separated, e.g., 64,64,32)", value="64,64")
    default_units = st.number_input("Fallback units per layer (if field empty)", 1, 1024, 64, step=1)
    activation = st.selectbox("Activation", ["ReLU", "Tanh", "Sigmoid", "GELU", "LeakyReLU", "ELU"])

    st.header("Training")
    optimizer_name = st.selectbox("Optimizer", ["Adam", "SGD", "RMSprop"])
    lr = st.number_input("Learning rate", min_value=1e-6, max_value=1.0, value=1e-3, format="%.6f")
    weight_decay = st.number_input("Weight decay (L2)", min_value=0.0, max_value=1.0, value=0.0, step=1e-4, format="%.4f")
    batch_size = st.selectbox("Batch size", [16, 32, 64, 128, 256, 512], index=2)
    epochs = st.slider("Epochs", 1, 5000, 500, step=10)
    use_early = st.checkbox("Early stopping", value=True)
    patience = st.slider("Patience (epochs)", 5, 200, 50, step=5) if use_early else None

    st.divider()
    auto_train = st.checkbox("Auto-train on change", value=False)
    train_clicked = st.button("Train / Re-train")


# Data generation
if train_frac + val_frac > 100:
    st.stop()

splits = (train_frac / 100.0, val_frac / 100.0, (100 - train_frac - val_frac) / 100.0)

data = generate_data(
    n_points=n_points,
    x_range=(x_min, x_max),
    noise_std=noise_std,
    kind=func_kind,
    seed=seed,
    normalize_x=normalize_x,
    normalize_y=normalize_y,
    splits=splits,
)

hidden_layers = build_model(hidden_spec, default_units, activation)

# Session state to control training
if "last_params" not in st.session_state:
    st.session_state["last_params"] = None
if "train_result" not in st.session_state:
    st.session_state["train_result"] = None

current_params = dict(
    func_kind=func_kind,
    n_points=n_points,
    x_min=x_min,
    x_max=x_max,
    noise_std=noise_std,
    seed=seed,
    splits=splits,
    normalize_x=normalize_x,
    normalize_y=normalize_y,
    hidden_layers=hidden_layers,
    activation=activation,
    optimizer_name=optimizer_name,
    lr=lr,
    weight_decay=weight_decay,
    batch_size=batch_size,
    epochs=epochs,
    patience=patience if use_early else None,
)

# Decide whether to train
should_train = False
if train_clicked:
    should_train = True
elif auto_train and current_params != st.session_state["last_params"]:
    should_train = True

if should_train:
    with st.spinner("Training..."):
        result = train_once(
            data=data,
            hidden_layers=hidden_layers,
            activation=activation,
            optimizer_name=optimizer_name,
            lr=lr,
            weight_decay=weight_decay,
            batch_size=batch_size,
            epochs=epochs,
            patience=patience if use_early else None,
            seed=seed,
        )
        st.session_state["train_result"] = result
        st.session_state["last_params"] = current_params

result = st.session_state.get("train_result", None)

# Layout
col_main, col_side = st.columns([4, 1.6])

with col_main:
    if result is None:
        st.info("Configure settings in the sidebar, then click **Train / Re-train**.")
    else:
        # Prediction plot
        fig = make_plot(data, result["preds"], show_true=True)
        st.plotly_chart(fig, use_container_width=True)

        # Loss curve
        with st.expander("Loss curves", expanded=False):
            loss_fig = make_loss_plot(result["history"])
            st.plotly_chart(loss_fig, use_container_width=True)

with col_side:
    if result is not None:
        st.subheader("Metrics")
        for split_name in ["Train", "Val", "Test"]:
            m = result["metrics"][split_name]
            st.metric(f"{split_name} R²", f"{m['R2']:.4f}")
            st.caption(f"MSE: {m['MSE']:.6f} · MAE: {m['MAE']:.6f}")
            st.divider()
        st.subheader("Model")
        st.write(f"Layers: {hidden_layers if hidden_layers else '[linear]'}")
        st.write(f"Activation: {activation}")
        st.write(f"Params: **{result['params']:,}**")
        st.write(f"Train time: {result['train_time']:.2f}s")

st.caption("Tip: increase points and epochs for smoother fits; try Tanh for periodic signals; adjust LR if training diverges.")
