import copy  # 深拷贝模型参数（保存最优权重）
import json  # 保存指标为 JSON
import time  # 记录训练耗时
from pathlib import Path  # 跨平台路径处理

import matplotlib  # 绘图库

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch  # 深度学习框架
import torch.nn as nn  # 网络层与损失函数
from sklearn.datasets import make_friedman1
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split  # 数据划分
from sklearn.preprocessing import StandardScaler  # 标准化
from torch.utils.data import DataLoader, TensorDataset  # 数据管道

# 中文字体：Windows 用微软雅黑，避免中文显示成方块
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

SEED = 42
BATCH_SIZE = 64  # 每个 batch 的样本数
LEARNING_RATE = 0.01
NUM_EPOCHS = 500
PATIENCE = 50
HIDDEN_DIMS = [64, 32]  # MLP 的隐藏层结构：两层，分别 64、32 个神经元

OUT_DIR = Path(__file__).parent / "outputs"
OUT_DIR.mkdir(exist_ok=True)

np.random.seed(SEED)
torch.manual_seed(SEED)

class LinearRegressionModel(nn.Module):

    def __init__(self, in_features: int):
        super().__init__()
        self.linear = nn.Linear(in_features, 1)  # y_hat = X Wᵀ + b

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


class MLPRegressionModel(nn.Module):

    def __init__(self, in_features: int, hidden_dims):
        super().__init__()
        layers = []  # 用列表动态拼接网络
        prev = in_features
        for h in hidden_dims:
            layers.append(nn.Linear(prev, h))  # 全连接层：维度 prev → h
            layers.append(nn.ReLU())
            prev = h  # 更新下一层的输入维度
        layers.append(nn.Linear(prev, 1))  # 输出层：最后映射成 1 维预测值
        self.net = nn.Sequential(*layers)  # Sequential 把层按顺序串成一条流水线

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)  # 数据依次流过每一层

def build_dataset():
    X, y = make_friedman1(n_samples=1500, n_features=10, noise=1.0, random_state=42)
    print(f"[数据] X.shape={X.shape}, y.shape={y.shape}, "
          f"y 范围=[{y.min():.2f}, {y.max():.2f}]")

    X_train, X_rest, y_train, y_rest = train_test_split(X, y, test_size=0.4, random_state=SEED)
    X_val, X_test, y_val, y_test = train_test_split(X_rest, y_rest, test_size=0.5, random_state=SEED)
    print(f"[划分] 训练 {len(X_train)} / 验证 {len(X_val)} / 测试 {len(X_test)}")

    x_scaler = StandardScaler().fit(X_train)
    y_scaler = StandardScaler().fit(y_train.reshape(-1, 1))

    X_train_s = x_scaler.transform(X_train)
    X_val_s = x_scaler.transform(X_val)
    X_test_s = x_scaler.transform(X_test)
    y_train_s = y_scaler.transform(y_train.reshape(-1, 1)).ravel()
    y_val_s = y_scaler.transform(y_val.reshape(-1, 1)).ravel()
    y_test_s = y_scaler.transform(y_test.reshape(-1, 1)).ravel()

    to_tensor = lambda arr: torch.tensor(np.asarray(arr), dtype=torch.float32)
    train_ds = TensorDataset(to_tensor(X_train_s), to_tensor(y_train_s).view(-1, 1))
    val_ds = TensorDataset(to_tensor(X_val_s), to_tensor(y_val_s).view(-1, 1))
    test_ds = TensorDataset(to_tensor(X_test_s), to_tensor(y_test_s).view(-1, 1))

    return {
        "train_ds": train_ds, "val_ds": val_ds, "test_ds": test_ds,
        "x_scaler": x_scaler, "y_scaler": y_scaler,
        "X_test": X_test,  # 原始测试特征，供残差诊断
        "y_train": y_train, "y_val": y_val, "y_test": y_test,
    }

def train_model(model, train_ds, val_ds):

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    history = {"train": [], "val": []}
    best_val = float("inf")
    best_state = None
    wait = 0
    t0 = time.time()

    for epoch in range(1, NUM_EPOCHS + 1):
        #训练阶段
        model.train()
        running = 0.0
        for xb, yb in train_loader:
            optimizer.zero_grad()  # 清空上一轮梯度
            loss = criterion(model(xb), yb)  # 前向 + 计算 MSE
            loss.backward()
            optimizer.step()  # 更新参数
            running += loss.item() * xb.size(0)  # 按样本数加权累计
        train_loss = running / len(train_ds)

        #验证阶段
        model.eval()
        with torch.no_grad():
            val_pred = torch.cat([model(xb) for xb, _ in val_loader], dim=0)
            val_loss = criterion(val_pred, val_ds.tensors[1]).item()

        history["train"].append(train_loss)
        history["val"].append(val_loss)

        #早停
        if val_loss < best_val - 1e-6:
            best_val = val_loss
            best_state = copy.deepcopy(model.state_dict())  # 保存最优权重
            wait = 0
        else:
            wait += 1
            if wait >= PATIENCE:
                break

    elapsed = time.time() - t0
    if best_state is not None:
        model.load_state_dict(best_state)  # 回滚到最优权重
    return history, best_val, len(history["train"]), elapsed

def evaluate(model, ds, y_true_original, y_scaler):
    model.eval()
    with torch.no_grad():
        pred_scaled = torch.cat([model(xb) for xb, _ in DataLoader(ds, batch_size=BATCH_SIZE)], dim=0)
    pred = y_scaler.inverse_transform(pred_scaled.numpy()).ravel()  # 反标准化

    mse = mean_squared_error(y_true_original, pred)
    return pred, {
        "MSE": float(mse),
        "RMSE": float(np.sqrt(mse)),
        "MAE": float(mean_absolute_error(y_true_original, pred)),
        "R2": float(r2_score(y_true_original, pred)),
    }


def count_params(model):
    #统计可学习参数总数
    return sum(p.numel() for p in model.parameters())

# 可视化

def plot_comparison(hist_lin, hist_mlp, y_test, pred_lin, pred_mlp, X_test_raw,
                    params_lin, params_mlp, save_dir: Path):

    res_lin = y_test - pred_lin
    res_mlp = y_test - pred_mlp
    fig, axes = plt.subplots(2, 3, figsize=(19, 10))

    # (a) 学习曲线：4 条线（两模型 × 训练/验证）
    ax = axes[0, 0]
    ax.plot(hist_lin["train"], color="#7F77DD", linestyle="-", label="线性-训练")
    ax.plot(hist_lin["val"], color="#7F77DD", linestyle="--", label="线性-验证")
    ax.plot(hist_mlp["train"], color="#D85A30", linestyle="-", label="MLP-训练")
    ax.plot(hist_mlp["val"], color="#D85A30", linestyle="--", label="MLP-验证")
    ax.set_xlabel("训练轮数 epoch");
    ax.set_ylabel("MSE（标准化尺度）")
    ax.set_title("(a) 学习曲线：MLP 下降更快、更低")
    ax.legend(fontsize=9);
    ax.grid(alpha=0.3)

    # (b)(c) 预测 vs 真实，两个模型并排
    for ax, pred, tag, name, color in [
        (axes[0, 1], pred_lin, "b", "线性模型", "#185FA5"),
        (axes[0, 2], pred_mlp, "c", "MLP", "#E24B4A"),
    ]:
        ax.scatter(y_test, pred, s=10, alpha=0.45, color=color)
        lim = [min(y_test.min(), pred.min()) - 1, max(y_test.max(), pred.max()) + 1]
        ax.plot(lim, lim, "--", color="#444441", linewidth=1.3)
        ax.set_xlim(lim);
        ax.set_ylim(lim)
        ax.set_xlabel("真实值 y");
        ax.set_ylabel("预测值 ŷ")
        ss_res = np.sum((y_test - pred) ** 2)
        ss_tot = np.sum((y_test - y_test.mean()) ** 2)
        ax.set_title(f"({tag}) {name}：预测 vs 真实（R²={1 - ss_res / ss_tot:.3f}）")
        ax.grid(alpha=0.3)

    # (d) 残差诊断：按 x2 分箱，看残差均值是否还有 U 形结构
    x2 = X_test_raw[:, 2]
    ax = axes[1, 0]
    edges = np.quantile(x2, np.linspace(0, 1, 9))
    bin_idx = np.clip(np.digitize(x2, edges[1:-1]), 0, 7)
    centers = [x2[bin_idx == k].mean() for k in range(8)]
    for res, name, color in [(res_lin, "线性模型", "#185FA5"), (res_mlp, "MLP", "#E24B4A")]:
        means = [res[bin_idx == k].mean() for k in range(8)]
        ax.plot(centers, means, "o-", color=color, linewidth=1.8, markersize=5, label=name)
    ax.axhline(0, color="#444441", linestyle="--", linewidth=1)
    ax.set_xlabel("特征 x2（真实规律含 20·(x2−0.5)²）");
    ax.set_ylabel("分箱残差均值")
    ax.set_title("(d) 残差诊断：线性呈 U 形，MLP 明显被拉平")
    ax.legend(fontsize=9);
    ax.grid(alpha=0.3)

    # (e) 前 80 个测试样本：真实 / 线性 / MLP 三条线
    ax = axes[1, 1]
    n = 80
    ax.plot(range(n), y_test[:n], color="#534AB7", marker="o", markersize=3, label="真实值")
    ax.plot(range(n), pred_lin[:n], color="#185FA5", marker="x", markersize=3, label="线性预测")
    ax.plot(range(n), pred_mlp[:n], color="#E24B4A", marker="^", markersize=3, label="MLP 预测")
    ax.set_xlabel("测试样本序号");
    ax.set_ylabel("y")
    ax.set_title("(e) 逐样本对比：MLP 更贴合峰谷")
    ax.legend(fontsize=9);
    ax.grid(alpha=0.3)

    # (f) 复杂度 vs 性能：柱状图对比参数量，折线叠加测试 R²
    ax = axes[1, 2]
    names = ["线性模型", "MLP"]
    params = [params_lin, params_mlp]

    def _r2(pred):
        return 1 - np.sum((y_test - pred) ** 2) / np.sum((y_test - y_test.mean()) ** 2)

    r2s = [_r2(pred_lin), _r2(pred_mlp)]
    xpos = np.arange(2)
    bars = ax.bar(xpos, params, width=0.5, color=["#7F77DD", "#D85A30"], alpha=0.85)
    ax.set_xticks(xpos);
    ax.set_xticklabels(names)
    ax.set_ylabel("可学习参数数量", color="#444441")
    for rect, p in zip(bars, params):
        ax.text(rect.get_x() + rect.get_width() / 2, p, f"{p:,}", ha="center", va="bottom", fontsize=10)
    ax2 = ax.twinx()
    ax2.plot(xpos, r2s, "D--", color="#0F6E56", markersize=8, label="测试 R²")
    for xi, r in zip(xpos, r2s):
        ax2.text(xi, r + 0.02, f"R²={r:.3f}", ha="center", color="#0F6E56", fontsize=10)
    ax2.set_ylabel("测试集 R²", color="#0F6E56");
    ax2.set_ylim(0, 1.15)
    ax.set_title("(f) 复杂度 vs 性能：参数暴增换来的性能提升")

    fig.suptitle("任务 2.2 线性模型 vs 全连接神经网络（MLP）对比", fontsize=16)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    path = save_dir / "task2_2_comparison.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[绘图] 已保存：{path}")

def main():
    data = build_dataset()

    lin_model = LinearRegressionModel(in_features=10)
    params_lin = count_params(lin_model)
    print("\n[模型-线性] 结构：Linear(10 → 1)")
    print(f"[模型-线性] 可学习参数：{params_lin}")
    hist_lin, bv_lin, ep_lin, t_lin = train_model(lin_model, data["train_ds"], data["val_ds"])

    mlp_model = MLPRegressionModel(in_features=10, hidden_dims=HIDDEN_DIMS)
    params_mlp = count_params(mlp_model)
    print("\n[模型-MLP] 结构：Linear(10→64) → ReLU → Linear(64→32) → ReLU → Linear(32→1)")
    print(mlp_model)
    print(f"[模型-MLP] 可学习参数：{params_mlp}")
    hist_mlp, bv_mlp, ep_mlp, t_mlp = train_model(mlp_model, data["train_ds"], data["val_ds"])

    pred_lin_tr, m_lin_tr = evaluate(lin_model, data["train_ds"], data["y_train"], data["y_scaler"])
    pred_lin_va, m_lin_va = evaluate(lin_model, data["val_ds"], data["y_val"], data["y_scaler"])
    pred_lin, m_lin_te = evaluate(lin_model, data["test_ds"], data["y_test"], data["y_scaler"])

    pred_mlp_tr, m_mlp_tr = evaluate(mlp_model, data["train_ds"], data["y_train"], data["y_scaler"])
    pred_mlp_va, m_mlp_va = evaluate(mlp_model, data["val_ds"], data["y_val"], data["y_scaler"])
    pred_mlp, m_mlp_te = evaluate(mlp_model, data["test_ds"], data["y_test"], data["y_scaler"])

    print("\n" + "=" * 78)
    print("对比表（测试集为主，训练/验证见下）")
    print("=" * 78)
    header = f"{'指标':<22}{'线性模型':>16}{'MLP':>16}{'变化':>18}"
    print(header);
    print("-" * 78)

    def row(label, a, b, fmt="{:>16.4f}"):
        diff = b - a
        print(f"{label:<22}" + fmt.format(a) + fmt.format(b) + f"{diff:>+18.4f}")

    row("测试 MSE", m_lin_te["MSE"], m_mlp_te["MSE"])
    row("测试 RMSE", m_lin_te["RMSE"], m_mlp_te["RMSE"])
    row("测试 MAE", m_lin_te["MAE"], m_mlp_te["MAE"])
    row("测试 R²", m_lin_te["R2"], m_mlp_te["R2"])
    row("训练 R²", m_lin_tr["R2"], m_mlp_tr["R2"])
    row("验证 R²", m_lin_va["R2"], m_mlp_va["R2"])
    print(f"{'可学习参数':<22}{params_lin:>16,}{params_mlp:>16,}"
          f"{params_mlp - params_lin:>+18,}")
    print(f"{'收敛轮数':<22}{ep_lin:>16d}{ep_mlp:>16d}{ep_mlp - ep_lin:>+18d}")
    print(f"{'训练耗时(秒)':<22}{t_lin:>16.2f}{t_mlp:>16.2f}{t_mlp - t_lin:>+18.2f}")
#保留JSON指标
    metrics = {
        "linear": {"train": m_lin_tr, "val": m_lin_va, "test": m_lin_te,
                   "params": params_lin, "epochs_ran": ep_lin, "train_seconds": round(t_lin, 2)},
        "mlp": {"train": m_mlp_tr, "val": m_mlp_va, "test": m_mlp_te,
                "params": params_mlp, "epochs_ran": ep_mlp, "train_seconds": round(t_mlp, 2),
                "architecture": "10 → 64 → 32 → 1（ReLU）"},
        "hyperparameters": {"batch_size": BATCH_SIZE, "lr": LEARNING_RATE,
                            "max_epochs": NUM_EPOCHS, "patience": PATIENCE, "seed": SEED},
    }
    (OUT_DIR / "metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")

    #可视化
    plot_comparison(hist_lin, hist_mlp, data["y_test"], pred_lin, pred_mlp, data["X_test"],
                    params_lin, params_mlp, OUT_DIR)
    np.save(OUT_DIR / "pred_linear.npy", pred_lin)
    np.save(OUT_DIR / "pred_mlp.npy", pred_mlp)

    print("\n" + "=" * 78)
    print("简要分析")
    print("=" * 78)
    print(f"1) 预测性能：MLP 测试 R² 从 {m_lin_te['R2']:.3f} 提升到 {m_mlp_te['R2']:.3f}，"
          f"RMSE 从 {m_lin_te['RMSE']:.2f} 降到 {m_mlp_te['RMSE']:.2f}。")
    print("   线性模型受限于只能拟合线性关系，属于「欠拟合」；MLP 借助隐藏层+ReLU 逼近非线性规律。")
    print(f"2) 训练过程：MLP 在更少/相近轮数内达到更低损失，但单轮计算更重（耗时 {t_mlp:.1f}s vs {t_lin:.1f}s）。")
    gap_lin = m_lin_tr["R2"] - m_lin_va["R2"]
    gap_mlp = m_mlp_tr["R2"] - m_mlp_va["R2"]
    print(f"3) 过拟合情况：线性模型 训练-验证 R²差距 {gap_lin:+.3f}；MLP 差距 {gap_mlp:+.3f}。")
    print(f"4) 模型复杂度：参数从 {params_lin} 个增加到 {params_mlp} 个"
          f"（约 {params_mlp / params_lin:.0f} 倍），换来 R² 提升 {m_mlp_te['R2'] - m_lin_te['R2']:.3f}。")
    print("5) 残差诊断：线性模型残差对 x2 呈明显 U 形（漏掉平方项）；MLP 的残差曲线明显被拉平，接近 0。")


if __name__ == "__main__":
    main()
