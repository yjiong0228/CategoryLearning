import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

############################################################
# 字体（中文）
############################################################
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

plt.rcParams['font.size'] = 12

############################################################
# 输出目录
############################################################
OUT_DIR = "exp_results"
os.makedirs(OUT_DIR, exist_ok=True)

############################################################
# 数据
############################################################
def load_subject_data(csv_path, subject_id, condition):

    df = pd.read_csv(csv_path)
    df = df[(df["iSub"] == subject_id) & (df["condition"] == condition)]
    df = df.sort_values(["iSession", "iBlock", "iTrial"])

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
# permutation
############################################################
def apply_two_permutations(features, mode_sequence=None):

    perm_A = [0,1,2,3]
    perm_B = [1,2,3,0]

    T = len(features)

    if mode_sequence is None:
        mode_sequence = [(t // 40) % 2 for t in range(T)]

    new_features = []

    for t in range(T):
        x = features[t]
        perm = perm_A if mode_sequence[t] == 0 else perm_B
        new_features.append(x[perm])

    return np.array(new_features), np.array(mode_sequence)

############################################################
# RNN
############################################################
class TinyRNN(nn.Module):

    def __init__(self, hidden_dim=16):
        super().__init__()

        H = hidden_dim
        self.H = H

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
# train
############################################################
def train(features, choices, feedback, H, epochs=500, lr=1e-3):

    model = TinyRNN(H).to(DEVICE)
    opt = optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()

    T = len(features)

    loss_curve = []
    best_acc = 0
    best_model = None

    for ep in range(epochs):

        h = torch.zeros(1, H).to(DEVICE)
        opt.zero_grad()

        loss_sum = 0
        correct = 0

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

            pred = torch.argmax(logits, dim=1).item()
            correct += (pred == choices[t]-1)

            loss = loss_fn(logits, torch.tensor([choices[t]-1], dtype=torch.long).to(DEVICE)) / T
            loss.backward()
            loss_sum += loss.item()

            h = h.detach()

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        acc = correct / T
        loss_curve.append(loss_sum)

        if acc > best_acc:
            best_acc = acc
            best_model = deepcopy(model.state_dict())

        if ep % 50 == 0:
            print(f"Epoch {ep} loss={loss_sum:.4f} acc={acc:.3f}")

    model.load_state_dict(best_model)
    return model, loss_curve

############################################################
# prediction
############################################################
def get_predictions(model, features, choices, feedback):

    model.eval()
    preds = []
    h = torch.zeros(1, model.H).to(DEVICE)

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

############################################################
# rho
############################################################
def compute_rho_traj(model, features, choices, feedback):

    model.eval()
    rho_list = []

    h = torch.zeros(1, model.H).to(DEVICE)

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
# 5阶段平行坐标图
############################################################
def plot_parallel_time_evolution(model, rho, save_path=None):

    colors = ['red', 'blue', 'green', 'orange']

    T = len(rho)

    selected = [
        (0, "初始状态"),
        (1, "第1试次"),
        (T//4, "学习前期"),
        (T//2, "学习中期"),
        (T-1, "学习末期")
    ]

    X = np.random.rand(300, 4)

    fig, axes = plt.subplots(len(selected), 4, figsize=(14, 12), sharey=True)

    for row, (t, label) in enumerate(selected):

        rho_t = torch.tensor(rho[t], dtype=torch.float32).view(1, -1).to(DEVICE)

        preds = []

        for x in X:
            x_t = torch.tensor(x, dtype=torch.float32).view(1, -1).to(DEVICE)

            with torch.no_grad():
                h = torch.tanh(rho_t + model.W_X(x_t))
                logits = model.readout(h)
                preds.append(torch.argmax(logits).item())

        preds = np.array(preds)

        for i in range(4):

            ax = axes[row, i]

            # 全灰背景（所有点）
            ax.plot(X.T, color='gray', alpha=0.02)

            # 只画当前 class（单色）
            idx = preds == i
            if np.sum(idx) > 0:
                ax.plot(X[idx].T, color=colors[i], alpha=0.35)

            ax.set_xticks([0,1,2,3])
            ax.set_xticklabels(["颈", "头", "腿", "尾"], fontsize=11)

            ax.tick_params(axis='y', labelsize=10)

            if i == 0:
                ax.set_ylabel(label + "\n特征取值", fontsize=13)

    fig.suptitle("决策空间演化过程", fontsize=16)

    legend_handles = [
        plt.Line2D([0],[0], color='red', label='类别1'),
        plt.Line2D([0],[0], color='blue', label='类别2'),
        plt.Line2D([0],[0], color='green', label='类别3'),
        plt.Line2D([0],[0], color='orange', label='类别4'),
    ]

    fig.legend(handles=legend_handles,
               loc="lower center",
               ncol=4,
               frameon=False,
               fontsize=12)

    plt.tight_layout(rect=[0,0.05,1,0.95])

    if save_path:
        plt.savefig(save_path, dpi=300)

    plt.close()

############################################################
# loss curve
############################################################
def plot_loss(loss_curve, save_path):

    plt.figure(figsize=(6,4))
    plt.plot(loss_curve, label="损失函数")
    plt.title("训练损失曲线", fontsize=14)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(alpha=0.3)

    plt.savefig(save_path, dpi=300)
    plt.close()

############################################################
# accuracy
############################################################
def compute_sliding_accuracy(pred, target, window=30):

    return np.array([
        np.mean(pred[i:i+window] == target[i:i+window])
        for i in range(len(pred)-window+1)
    ])


def plot_accuracy(pred, choices, category, save_path):

    acc1 = compute_sliding_accuracy(pred, choices-1)
    acc2 = compute_sliding_accuracy(pred, category)
    acc3 = compute_sliding_accuracy(choices-1, category)

    plt.figure(figsize=(6,4))
    plt.plot(acc1, label="模型 vs 人类")
    plt.plot(acc2, label="模型 vs 真值")
    plt.plot(acc3, label="人类 vs 真值")

    plt.title("准确率滑动曲线", fontsize=14)
    plt.legend()
    plt.grid(alpha=0.3)

    plt.savefig(save_path, dpi=300)
    plt.close()

############################################################
# hidden size curve
############################################################
def plot_hidden_size_curve(dims, accs, save_path):

    plt.figure(figsize=(6,4))
    plt.plot(dims, accs, marker='o')

    plt.title("隐层维度 vs 性能", fontsize=14)
    plt.xlabel("Hidden Size")
    plt.ylabel("Accuracy")

    plt.xticks(dims, dims) 
    plt.grid(alpha=0.3)
    plt.ylim(0,1)

    plt.savefig(save_path, dpi=300)
    plt.close()

############################################################
# run
############################################################
def run_single(features, choices, feedback, category, H):

    features, _ = apply_two_permutations(features)

    model, loss_curve = train(features, choices, feedback, H)

    pred = get_predictions(model, features, choices, feedback)

    rho = compute_rho_traj(model, features, choices, feedback)

    np.save(f"{OUT_DIR}/rho_H{H}.npy", rho)
    np.save(f"{OUT_DIR}/pred_H{H}.npy", pred)
    np.save(f"{OUT_DIR}/loss_H{H}.npy", loss_curve)

    # 平行坐标
    plot_parallel_time_evolution(
        model, rho,
        save_path=f"{OUT_DIR}/parallel_H{H}.png"
    )

    # loss
    plot_loss(
        loss_curve,
        f"{OUT_DIR}/loss_H{H}.png"
    )

    # accuracy
    plot_accuracy(
        pred,
        choices-1,
        category-1,
        f"{OUT_DIR}/acc_H{H}.png"
    )

    return np.mean(pred == (choices-1))

############################################################
# main
############################################################
def run(csv_path):

    features, choices, feedback, category = load_subject_data(
        csv_path, 302, 3
    )

    dims = [16,32,48,64]
    accs = []

    for H in dims:
        print(f"hidden size H={H}")
        acc = run_single(features, choices, feedback, category, H)
        accs.append(acc)

    plot_hidden_size_curve(
        dims,
        accs,
        f"{OUT_DIR}/hidden_curve.png"
    )

    print("All saved in:", OUT_DIR)


if __name__ == "__main__":
    run("Task2_processed.csv")