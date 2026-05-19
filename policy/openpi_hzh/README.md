# openpi_hzh

双臂仿真用 OpenPI/pi0.5 适配器，匹配
`/home/tianji/hzh/study/openpi/convert_hdf5_to_new_format_simulation_bimanual_openpi.py`
生成的数据格式。

## 1. 启动 OpenPI policy server

在 OpenPI/pi0.5 环境中运行：

```bash
python policy/openpi_hzh/serve_policy_openpi_bimanual_sim.py \
  --openpi-repo-root /home/tianji/hzh/study/openpi \
  --config-name YOUR_TRAIN_CONFIG \
  --checkpoint-dir /path/to/openpi/checkpoints/YOUR_RUN/STEP \
  --pytorch-device cuda:0 \
  --port 8000
```

默认会把 OpenPI config 的推理 transform 替换成双臂格式：

- `observation.images.image`
- `observation.images.left_wrist_image`
- `observation.images.right_wrist_image`
- `observation.state`，16 维
- 输出 `actions`，14 维

这可以避免复用 `LeRobotLiberoDataConfig` 时推理端只返回 7 维动作。

## 2. 启动 RoboTwin eval

在 RoboTwin 环境中运行：

```bash
bash policy/openpi_hzh/eval.sh TASK_NAME TASK_CONFIG CKPT_SETTING SEED GPU_ID localhost 8000
```

第 8 个可选参数可覆盖每次执行的 action chunk 步数：

```bash
bash policy/openpi_hzh/eval.sh TASK_NAME TASK_CONFIG CKPT_SETTING SEED GPU_ID localhost 8000 10
```
