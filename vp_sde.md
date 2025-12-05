# VP-SDE 2D 高斯实验说明
## 配置
- 全局参数位于 `config.py`：
  - `mu`、`sigma`：目标高斯分布的均值与协方差。
  - `config_path`：训练时保存 / 采样时加载的 checkpoint 路径（默认 `vp_sde_2d_best.pt`）。
  - 训练超参：`batch_size`、`epochs`、`lr`、`weight_decay` 等。
- 设备选择由 `get_device()` 自动完成：若检测到 CUDA 则用 GPU，否则用 CPU。需要强制 CPU 可在命令前加 `CUDA_VISIBLE_DEVICES=""`。

## 训练
脚本：`run_training.py`

运行：
```bash
# GPU 自动检测；如需 CPU，请在前面加 CUDA_VISIBLE_DEVICES=""
python run_training.py
```

流程摘要：
- 生成训练 / 验证集（高斯采样）。
- 构建 `ScoreNet` 模型并迁移到设备。
- 训练过程中保存最优指标的 checkpoint 到 `config["config_path"]`。

常用可调参数（均在 `config.py`）：
- 数据量：`n_train`、`n_val`。
- 学习率与衰减：`lr`、`weight_decay`、`lr_min`。
- 训练时长：`epochs`。
- SDE 超参：`beta_min`、`beta_max`、`eps`、`num_steps` 等。

## 采样
脚本：`run_sampling.py`

运行：
```bash
python run_sampling.py
```

流程摘要：
- 读取 `config["config_path"]` 对应的 checkpoint 并合并配置。
- 生成测试集样本（用于对比）。
- 生成模型样本：
  - PC 采样：`pc_sample`
  - PF-ODE 采样：`pf_sample`
  - 基准 Oracle 采样：`oracle_sample`
- 绘制散点图，并打印均值/方差/协方差的统计对比。

可选参数：
- 采样步数、SNR、校正步等可在 `config.py` 中调节（例如 `num_steps`、`snr`、`corrector_steps`）。

## 检查/评估
- `grid_error_metrics.py`：在网格上对比模型对数密度与解析解。
- `show_nll_error.py`：绘制 NFE 与 NLL 偏差、KL 的收敛曲线。

