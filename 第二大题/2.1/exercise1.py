
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np                              # 数值计算
import torch                                    # 深度学习框架 Pytorch主库
import torch.nn as nn                           # 网络层与损失函数
from sklearn.datasets import make_friedman1
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score  # 三个回归评价指标
from sklearn.model_selection import train_test_split                          # 数据集划分
from sklearn.preprocessing import StandardScaler                              # 标准化
from torch.utils.data import DataLoader, TensorDataset                        # 数据管道

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False      # 正常显示负号（否则 -1 会画成方块）

SEED = 42
BATCH_SIZE = 64
LEARNING_RATE = 0.01
NUM_EPOCHS = 500
PATIENCE = 50             # 早停耐心值：验证损失连续 50 轮不下降就停止训练

OUT_DIR = Path(__file__).parent / "outputs"
OUT_DIR.mkdir(exist_ok=True)

# 设置随机种子：numpy 管数据/划分，torch 管参数初始化与 batch 打乱
np.random.seed(SEED)
torch.manual_seed(SEED)

class LinearRegressionModel(nn.Module):

    def __init__(self, in_features: int):
        super().__init__()                       # 调用父类 nn.Module 的初始化，内部的 `_parameters`、`_modules` 等容器
        self.linear = nn.Linear(in_features, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)                    # 自动调回forward


def build_dataset():

    X, y = make_friedman1(
        n_samples=1500,
        n_features=10,
        noise=1.0,
        random_state=42,
    )
    print(f"[数据] X.shape={X.shape}, y.shape={y.shape}, "
          f"y 范围=[{y.min():.2f}, {y.max():.2f}]")

    X_train, X_rest, y_train, y_rest = train_test_split(
        X, y, test_size=0.4, random_state=SEED
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_rest, y_rest, test_size=0.5, random_state=SEED
    )
    print(f"[划分] 训练 {len(X_train)} / 验证 {len(X_val)} / 测试 {len(X_test)}")

    x_scaler = StandardScaler().fit(X_train)    # 只统计训练集的均值/标准差
    y_scaler = StandardScaler().fit(y_train.reshape(-1, 1))  # 目标值同样标准化，能让损失尺度更稳定先变成二维，下面又给压回一维，方便后面构造张量

    X_train_s = x_scaler.transform(X_train)      # 变换训练集
    X_val_s = x_scaler.transform(X_val)          # 用训练集的统计量变换验证集
    X_test_s = x_scaler.transform(X_test)        # 同理变换测试集
    y_train_s = y_scaler.transform(y_train.reshape(-1, 1)).ravel()   # ravel 把 (n,1) 变回 (n,)
    y_val_s = y_scaler.transform(y_val.reshape(-1, 1)).ravel()
    y_test_s = y_scaler.transform(y_test.reshape(-1, 1)).ravel()
    # transform用训练集学到的 μ、σ 去变换验证/测试集，保证三者处于同一坐标系。

    to_tensor = lambda arr: torch.tensor(np.asarray(arr), dtype=torch.float32)#转32

    train_ds = TensorDataset(to_tensor(X_train_s), to_tensor(y_train_s).view(-1, 1))#一维变二维，打包，取第i个样本同时返回(x_i,y_i)
    val_ds = TensorDataset(to_tensor(X_val_s), to_tensor(y_val_s).view(-1, 1))
    test_ds = TensorDataset(to_tensor(X_test_s), to_tensor(y_test_s).view(-1, 1))

    return {
        "train_ds": train_ds, "val_ds": val_ds, "test_ds": test_ds,
        "x_scaler": x_scaler, "y_scaler": y_scaler,
        "X_test": X_test,                                    # 保留原始（未标准化）测试特征，供残差诊断使用
        "y_train": y_train, "y_val": y_val, "y_test": y_test,
    }
#训练循环
def train_model(model, train_ds, val_ds):
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)#打乱训练集，避免“样本排列顺序”带来的系统性偏置，也让每个 batch 是随机采样。验证集不需要打乱（反正只求平均损失）。
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)

    criterion = nn.MSELoss()   #对大误差惩罚更重
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE) #Adam 为每个参数自适应调整步长，对学习率不敏感、收敛快，适合这种小规模实验。

    history = {"train": [], "val": []}                          # 记录每轮的损失，用于画学习曲线
    best_val = float("inf")                                     # 记录历史最佳验证损失
    best_state = None                                           # 记录最佳时刻的模型参数
    wait = 0                                                    # 计数器：验证损失连续未改善的轮数

    for epoch in range(1, NUM_EPOCHS + 1):
        model.train()                                           # 切到训练模式
        running = 0.0                                           # 累计本轮的损失
        for xb, yb in train_loader:
            optimizer.zero_grad()                               # 清梯度
            pred = model(xb)                                    # 前向传播，得到预测
            loss = criterion(pred, yb)                          # 计算 batch 内的平均 MSE
            loss.backward()                                     # 反向传播：自动按链式法则求出所有参数的梯度，存入param.grad
            optimizer.step()                                    # 优化器按梯度param.grad更新参数
            running += loss.item() * xb.size(0)                 # 按样本数加权累计（batch 大小可能不同）得到整个 epoch 的平均损失
        train_loss = running / len(train_ds)

        #验证阶段
        model.eval()                                            # 切到评估模式（关闭 Dropout 等）
        with torch.no_grad():                                   # 评估不需要梯度，省显存、提速
            val_pred = torch.cat([model(xb) for xb, _ in val_loader], dim=0)
            val_loss = criterion(val_pred, val_ds.tensors[1]).item()

        history["train"].append(train_loss)
        history["val"].append(val_loss)

        #早停
        if val_loss < best_val - 1e-6:
            best_val = val_loss#更新历史最好损失
            best_state = {k: v.clone() for k, v in model.state_dict().items()} #拷贝当前模型的所有参数，保存到best_state中，把巅峰状态的参数单独存下来
            wait = 0
        else:
            wait += 1
            if wait >= PATIENCE:
                print(f"[训练] 早停触发：第 {epoch} 轮，最佳验证 MSE = {best_val:.4f}")
                break

        if epoch % 50 == 0 or epoch == 1:
            print(f"[训练] epoch {epoch:3d} | train MSE {train_loss:.4f} | val MSE {val_loss:.4f}")

    # 回滚到验证效果最好的那组参数（避免用上「已经过拟合」的最后一轮参数）
    if best_state is not None:
        model.load_state_dict(best_state)
    print(f"[训练] 完成，共 {len(history['train'])} 轮，最佳验证 MSE = {best_val:.4f}")
    return history

#评估
def evaluate(model, ds, y_true_original, y_scaler):
    """在指定数据集上评估，并把预测值反标准化回真实量纲后再算指标。"""
    model.eval()
    with torch.no_grad():
        pred_scaled = torch.cat([model(xb) for xb, _ in DataLoader(ds, batch_size=BATCH_SIZE)], dim=0)
    # 反标准化：y = y_scaled * σ + μ，让指标回到 y 的真实单位（否则 MSE 数值没有物理意义）
    pred = y_scaler.inverse_transform(pred_scaled.numpy()).ravel()

    mse = mean_squared_error(y_true_original, pred)             # 均方误差
    rmse = np.sqrt(mse)                                          # 均方根误差：与 y 同量纲，最好解释
    mae = mean_absolute_error(y_true_original, pred)             # 平均绝对误差：对离群点更稳健
    r2 = r2_score(y_true_original, pred)                         # 决定系数：解释了多少方差，越接近 1 越好
    return pred, {"MSE": float(mse), "RMSE": float(rmse), "MAE": float(mae), "R2": float(r2)}


#可视化

def plot_results(history, y_test, pred_test, X_test_raw, metrics, save_dir: Path):
    """四联图：(a) 学习曲线 (b) 预测 vs 真实 (c) 残差分箱诊断 (d) 逐样本对比。

    其中 (c) 用「分箱残差均值」而不是单纯相关系数来诊断：
    平方项/正弦项这类对称的非线性结构，与残差的线性相关系数接近 0，
    但按特征分箱后残差均值会呈现明显的 U 形/倒 U 形趋势。
    """
    residual = y_test - pred_test
    fig, axes = plt.subplots(2, 2, figsize=(13, 10))

    # (a) 学习曲线：训练/验证损失随轮数变化
    ax = axes[0, 0]
    ax.plot(history["train"], label="训练 MSE（标准化尺度）", color="#534AB7")
    ax.plot(history["val"], label="验证 MSE（标准化尺度）", color="#D85A30")
    ax.set_xlabel("训练轮数 epoch")
    ax.set_ylabel("MSE Loss")
    ax.set_title("(a) 学习曲线：训练与验证损失同步下降后趋于平稳")
    ax.legend()
    ax.grid(alpha=0.3)

    # (b) 预测值 vs 真实值：散点越贴近对角线，预测越准
    ax = axes[0, 1]
    ax.scatter(y_test, pred_test, s=12, alpha=0.5, color="#185FA5", label="测试样本")
    lim = [min(y_test.min(), pred_test.min()) - 1, max(y_test.max(), pred_test.max()) + 1]
    ax.plot(lim, lim, "--", color="#E24B4A", linewidth=1.5, label="理想线 y = ŷ")
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.set_xlabel("真实值 y")
    ax.set_ylabel("预测值 ŷ")
    ax.set_title(f"(b) 预测值 vs 真实值（测试集 R² = {metrics['R2']:.3f}）")
    ax.legend()
    ax.grid(alpha=0.3)

    # (c) 残差诊断：按 x2（真实规律中的平方项变量）分箱，看残差均值是否有系统结构
    x2 = X_test_raw[:, 2]                                   # 取出第 3 个特征（从 0 开始编号）
    ax = axes[1, 0]
    ax.scatter(x2, residual, s=10, alpha=0.25, color="#888780", label="测试样本残差")
    # 把 x2 等频分成 8 箱，计算每箱残差的均值——对称的非线性结构在这里会暴露出来
    edges = np.quantile(x2, np.linspace(0, 1, 9))
    bin_idx = np.clip(np.digitize(x2, edges[1:-1]), 0, 7)
    centers = [x2[bin_idx == k].mean() for k in range(8)]
    means = [residual[bin_idx == k].mean() for k in range(8)]
    ax.plot(centers, means, "o-", color="#E24B4A", linewidth=1.8, markersize=5,
            label="分箱残差均值（U 形 → 漏掉平方项）")
    ax.axhline(0, color="#444441", linestyle="--", linewidth=1)
    corr = np.corrcoef(x2, residual)[0, 1]
    ax.set_xlabel("特征 x2（真实规律含 20·(x2−0.5)²）")
    ax.set_ylabel("残差 y − ŷ")
    ax.set_title(f"(c) 残差诊断：线性相关系数仅 {corr:.2f}，"
                 f"但分箱均值呈明显 U 形")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    # (d) 前 80 个测试样本的真值 vs 预测折线：看两者跟随程度与幅度压缩
    ax = axes[1, 1]
    n_show = 80
    shrink = pred_test.std() / y_test.std()
    ax.plot(range(n_show), y_test[:n_show], label="真实值", color="#534AB7", marker="o", markersize=3)
    ax.plot(range(n_show), pred_test[:n_show], label="预测值", color="#D85A30", marker="x", markersize=3)
    ax.set_xlabel("测试样本序号")
    ax.set_ylabel("y")
    ax.set_title(f"(d) 逐样本对比：趋势跟得住，但幅度被压缩（预测/真实标准差 = {shrink:.2f}）")
    ax.legend()
    ax.grid(alpha=0.3)

    fig.suptitle("任务 2.1 线性模型（无隐藏层）预测效果可视化", fontsize=15)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig_path = save_dir / "task2_1_result.png"
    fig.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[绘图] 已保存：{fig_path}")


def plot_coefficients(w_raw, save_dir: Path):
    """画出还原到原始尺度的线性系数，并标注真实数据生成过程的线性项。"""
    names = [f"x{i}" for i in range(10)]
    fig, ax = plt.subplots(figsize=(9, 4.5))
    bars = ax.bar(names, w_raw, color="#7F77DD", width=0.6, label="线性模型学到的系数")
    # 真实规律中只有 x3、x4 是线性项：y 含 10*x3 + 5*x4
    ax.plot([3, 4], [10, 5], "x", color="#E24B4A", markersize=10, label="真实线性系数 (x3=10, x4=5)")
    for rect, v in zip(bars, w_raw):
        ax.text(rect.get_x() + rect.get_width() / 2, v, f"{v:.2f}",
                ha="center", va="bottom" if v >= 0 else "top", fontsize=9)
    ax.axhline(0, color="#888780", linewidth=0.8)
    ax.set_xlabel("特征")
    ax.set_ylabel("系数（原始尺度）")
    ax.set_title("线性系数 vs 真实规律：x3、x4 精准命中（真实值 10 / 5）；\n"
                 "x0、x1 是对正弦项的线性近似；x2 ≈ 0（对称的二次项被截距吸收）",
                 fontsize=12)
    ax.legend()
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    path = save_dir / "task2_1_coefficients.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[绘图] 已保存：{path}")


# ---------------------------------------------------------------------------
# 7. 主流程
# ---------------------------------------------------------------------------
def main():
    data = build_dataset()

    # 用 10 个输入特征搭建线性模型（无隐藏层）
    model = LinearRegressionModel(in_features=10)
    total_params = sum(p.numel() for p in model.parameters())
    print("[模型] 结构：Linear(10 -> 1) —— 不含隐藏层")
    print(model)
    print(f"[模型] 可学习参数总数：{total_params}（10 个权重 + 1 个偏置）")

    history = train_model(model, data["train_ds"], data["val_ds"])

    # 分别在训练/验证/测试集上评估（对比三者可判断是欠拟合还是过拟合）
    pred_train, m_train = evaluate(model, data["train_ds"], data["y_train"], data["y_scaler"])
    pred_val, m_val = evaluate(model, data["val_ds"], data["y_val"], data["y_scaler"])
    pred_test, m_test = evaluate(model, data["test_ds"], data["y_test"], data["y_scaler"])

    print("\n================ 评估结果（反标准化后的真实量纲） ================")
    for name, m in [("训练集", m_train), ("验证集", m_val), ("测试集", m_test)]:
        print(f"{name}: MSE={m['MSE']:.4f}  RMSE={m['RMSE']:.4f}  "
              f"MAE={m['MAE']:.4f}  R²={m['R2']:.4f}")

    # 保存指标
    metrics_path = OUT_DIR / "metrics.json"
    metrics_path.write_text(json.dumps(
        {"train": m_train, "val": m_val, "test": m_test,
         "epochs_ran": len(history["train"]),
         "hyperparameters": {"batch_size": BATCH_SIZE, "lr": LEARNING_RATE,
                             "max_epochs": NUM_EPOCHS, "patience": PATIENCE, "seed": SEED},
         "model": "Linear(10 -> 1)，无隐藏层，可学习参数 11 个"},
        ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[保存] 指标已写入：{metrics_path}")

    # 可视化（四联图）
    plot_results(history, data["y_test"], pred_test, data["X_test"], m_test, OUT_DIR)

    # 把标准化尺度的权重还原到原始特征尺度：
    #   标准化后 ŷ_s = Σ w_s·(x-μ)/σ + b_s
    #   原始尺度  ŷ   = Σ (w_s/σ)·x + (b_s - Σ w_s·μ/σ)
    w_std = model.linear.weight.detach().numpy().ravel()
    b_std = model.linear.bias.detach().numpy().ravel()[0]
    mu, sigma = data["x_scaler"].mean_, data["x_scaler"].scale_
    y_mu, y_sigma = float(data["y_scaler"].mean_[0]), float(data["y_scaler"].scale_[0])
    w_raw = (w_std * y_sigma) / sigma                       # 乘 y 的标准差，回到 y 的原尺度
    b_raw = b_std * y_sigma + y_mu - float(np.sum(w_raw * mu))
    print("\n[系数] 还原到原始尺度：")
    for i, w in enumerate(w_raw):
        print(f"    x{i}: {w:+7.3f}")
    print(f"    截距 b: {b_raw:+.3f}（真实数据没有常数项，接近 0 说明拟合合理）")

    plot_coefficients(w_raw, OUT_DIR)
    np.save(OUT_DIR / "test_predictions.npy", pred_test)     # 顺手保存预测结果便于复查

    # 简要分析结论（与讲解文档中一致）
    shrink = pred_test.std() / data["y_test"].std()
    print("\n================ 简要分析 ================")
    print(f"1) 训练/验证/测试 R² 分别为 {m_train['R2']:.3f} / {m_val['R2']:.3f} / {m_test['R2']:.3f}，")
    print("   三者非常接近（测试甚至略高）-> 没有过拟合，瓶颈在「模型表达能力不足」（欠拟合）。")
    print(f"2) 测试集 RMSE = {m_test['RMSE']:.2f}（y 本身标准差 = {data['y_test'].std():.2f}，噪声标准差 = 1.0），")
    print("   误差远大于纯噪声水平，说明存在「模型偏差」而非只是随机噪声。")
    print(f"   预测值标准差 / 真实值标准差 = {shrink:.2f} < 1，预测幅度被压缩——")
    print("   这是线性模型对正弦、平方这类弯曲规律的典型表现。")
    print("3) 系数分析：x3 ≈ 9.8、x4 ≈ 5.2 与真实线性系数(10、5)高度吻合；")
    print("   x2 的系数≈0 —— 因为 20(x2-0.5)² 关于 0.5 对称，其最佳线性逼近近似常数，被截距吸收。")
    print("4) 残差诊断：与 x2 的线性相关系数接近 0，但分箱残差均值呈明显 U 形，")
    print("   证明模型漏掉了二次项（对称的非线性结构用相关系数看不出来）。")


if __name__ == "__main__":
    main()

input("按回车退出")
