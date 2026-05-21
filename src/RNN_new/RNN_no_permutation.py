import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


############################################################
# 数据
############################################################

def load_subject_data(csv_path, subject_id, condition):

    df = pd.read_csv(csv_path)
    df = df[(df["iSub"] == subject_id) & (df["condition"] == condition)]
    df = df.sort_values(["iSession","iBlock","iTrial"])

    features = df[["feature1","feature2","feature3","feature4"]].values
    choices = df["choice"].values.astype(int)
    feedback = df["feedback"].values
    category = df["category"].values.astype(int)

    return features, choices, feedback, category


def one_hot(x, num_classes):
    v = np.zeros(num_classes)
    v[x] = 1.0
    return v


############################################################
# RNN
############################################################

class TinyRNN(nn.Module):

    def __init__(self, hidden_dim=16):
        super().__init__()

        H = hidden_dim

        self.W_H = nn.Linear(H, H, bias=False)
        self.W_X = nn.Linear(4, H, bias=False)
        self.W_C = nn.Linear(4, H, bias=False)
        self.W_F = nn.Linear(1, H, bias=False)

        self.bias = nn.Parameter(torch.zeros(H))

        self.readout = nn.Linear(H, 4)

    def forward(self, x, c, f, h):

        rho = self.W_H(h) + self.W_C(c) + self.W_F(f) + self.bias
        h_next = torch.tanh(rho + self.W_X(x))
        logits = self.readout(h_next)

        return logits, h_next, rho


############################################################
# 训练
############################################################

def train(features, choices, feedback,
          H=16, epochs=150, lr=1e-3):

    model = TinyRNN(H).to(DEVICE)
    opt = optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()

    T = len(features)

    loss_curve = []
    saved_models = {}

    save_points = [0, epochs//3, 2*epochs//3, epochs-1]

    for ep in range(epochs):

        h = torch.zeros(1, H).to(DEVICE)
        total_loss = 0

        for t in range(T):

            if t == 0:
                c_prev = np.zeros(4)
                f_prev = 0
            else:
                c_prev = one_hot(choices[t-1]-1, 4)
                f_prev = feedback[t-1]

            x_t = torch.tensor(features[t], dtype=torch.float32).view(1,-1).to(DEVICE)
            c_t = torch.tensor(c_prev, dtype=torch.float32).view(1,-1).to(DEVICE)
            f_t = torch.tensor([[f_prev]], dtype=torch.float32).to(DEVICE)

            logits, h, _ = model(x_t, c_t, f_t, h)

            target = torch.tensor([choices[t]-1], dtype=torch.long).to(DEVICE)

            loss = loss_fn(logits, target)

            opt.zero_grad()
            loss.backward()
            opt.step()

            total_loss += loss.item()
            h = h.detach()

        loss_curve.append(total_loss / T)

        print(f"Epoch {ep}: loss={loss_curve[-1]:.4f}")

        if ep in save_points:
            saved_models[ep] = TinyRNN(H).to(DEVICE)
            saved_models[ep].load_state_dict(model.state_dict())
            saved_models[ep].eval()

    return model, loss_curve, saved_models


############################################################
# accuracy curve
############################################################

def compute_sliding_accuracy(pred, target, window=30):
    return np.array([
        np.mean(pred[i:i+window] == target[i:i+window])
        for i in range(len(pred)-window+1)
    ])


def get_predictions(model, features, choices, feedback):

    model.eval()
    preds = []

    h = torch.zeros(1, model.W_H.in_features).to(DEVICE)

    for t in range(len(features)):

        if t == 0:
            c_prev = np.zeros(4)
            f_prev = 0
        else:
            c_prev = one_hot(choices[t-1]-1, 4)
            f_prev = feedback[t-1]

        x_t = torch.tensor(features[t], dtype=torch.float32).view(1,-1).to(DEVICE)
        c_t = torch.tensor(c_prev, dtype=torch.float32).view(1,-1).to(DEVICE)
        f_t = torch.tensor([[f_prev]], dtype=torch.float32).to(DEVICE)

        with torch.no_grad():
            logits, h, _ = model(x_t, c_t, f_t, h)

        preds.append(torch.argmax(logits).item())

    return np.array(preds)


def plot_accuracy(pred, choices, category):

    acc1 = compute_sliding_accuracy(pred, choices)
    acc2 = compute_sliding_accuracy(pred, category)
    acc3 = compute_sliding_accuracy(choices, category)

    plt.plot(acc1, label="Model vs Human")
    plt.plot(acc2, label="Model vs True")
    plt.plot(acc3, label="Human vs True")
    plt.legend()
    plt.title("Accuracy Curves")
    plt.show()


############################################################
# rho
############################################################

def compute_rho_traj(model, features, choices, feedback):

    model.eval()
    rho_list = []

    h = torch.zeros(1, model.W_H.in_features).to(DEVICE)

    for t in range(len(features)):

        if t == 0:
            c_prev = np.zeros(4)
            f_prev = 0
        else:
            c_prev = one_hot(choices[t-1]-1, 4)
            f_prev = feedback[t-1]

        x_t = torch.tensor(features[t], dtype=torch.float32).view(1,-1).to(DEVICE)
        c_t = torch.tensor(c_prev, dtype=torch.float32).view(1,-1).to(DEVICE)
        f_t = torch.tensor([[f_prev]], dtype=torch.float32).to(DEVICE)

        with torch.no_grad():
            _, h, rho = model(x_t, c_t, f_t, h)

        rho_list.append(rho.cpu().numpy().flatten())

    return np.array(rho_list)


############################################################
# PCA phase
############################################################

def plot_rho_phase(rho):

    pca = PCA(n_components=2)
    proj = pca.fit_transform(rho)

    colors = np.linspace(0, 1, len(proj))

    plt.scatter(proj[:,0], proj[:,1], c=colors, s=10)
    plt.title("rho phase portrait (PCA)")
    plt.show()


############################################################
# 平行坐标图
############################################################

def plot_parallel(model, rho, N=1000, title=""):

    colors = ['red', 'blue', 'green', 'orange']

    X = np.random.rand(N, 4)
    preds = []

    for x in X:

        x_t = torch.tensor(x, dtype=torch.float32).view(1,-1).to(DEVICE)
        rho_t = torch.tensor(rho, dtype=torch.float32).view(1,-1).to(DEVICE)

        with torch.no_grad():
            h = torch.tanh(rho_t + model.W_X(x_t))
            logits = model.readout(h)
            pred = torch.argmax(logits).item()

        preds.append(pred)

    X = np.array(X)
    preds = np.array(preds)

    fig, axes = plt.subplots(1, 4, figsize=(20,5), sharey=True)

    for cls in range(4):

        # 背景
        axes[cls].plot(X.T, color='gray', alpha=0.01)

        idx = preds == cls

        if np.sum(idx) > 0:
            axes[cls].plot(X[idx].T, color=colors[cls], alpha=0.2)

        axes[cls].set_title(f"Class {cls}")

    fig.suptitle(title)
    plt.show()


############################################################
# 主流程
############################################################

def run(csv_path):

    features, choices, feedback, category = load_subject_data(
        csv_path, 302, 3
    )

    model, loss_curve, saved_models = train(features, choices, feedback)

    plt.plot(loss_curve)
    plt.title("Learning Curve")
    plt.show()

    pred = get_predictions(model, features, choices, feedback)
    plot_accuracy(pred, choices-1, category-1)

    rho = compute_rho_traj(model, features, choices, feedback)
    plot_rho_phase(rho)

    T = len(rho)
    for t in [0, T//3, 2*T//3, T-1]:
        plot_parallel(model, rho[t], title=f"Trial {t}")

    for ep, m in saved_models.items():
        rho = compute_rho_traj(m, features, choices, feedback)
        plot_parallel(m, rho[-1], title=f"Epoch {ep}")


############################################################

if __name__ == "__main__":

    if os.path.exists("Task2_processed.csv"):
        run("Task2_processed.csv")
    else:
        print("CSV not found")