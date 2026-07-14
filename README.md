# Tianji Marin Pro Embodiment 使用说明

本文档说明如何在 RoboTwin 中使用Tianji公司双臂机器人 Marin Pro 进行仿真任务运行和数据采集。本仓库中该机器人以 `tianji` 作为 embodiment 名称注册，机器人资产位于 `assets/embodiments/tianji/`。

## 任务数据生成成功率

已采集任务的官方 vs 实测成功率对比见：[docs/task_success_rate_comparison.md](docs/task_success_rate_comparison.md)。

原始数据托管在 Hugging Face：

- Dataset: [Continuity3/marvin_pro_robotwin](https://huggingface.co/datasets/Continuity3/marvin_pro_robotwin)
- 数据路径：[`data/`](https://huggingface.co/datasets/Continuity3/marvin_pro_robotwin/tree/main/data)

| Task | 官方 (Aloha-AgileX) | 我们实测 (Tianji) |
| --- | ---: | ---: |
| adjust_bottle | 93% | 93% |
| beat_block_hammer | 64% | 71% |
| click_alarmclock | 92% | 88% |
| grab_roller | 95% | 91% |
| lift_pot | 27% | 34% |
| move_playingcard_away | 99% | 95% |
| move_stapler_pad | 92% | 92% |
| open_laptop | 82% | 87% |
| place_burger_fries | 97% | 93% |
| place_container_plate | 89% | 89% |
| place_empty_cup | 92% | 96% |
| place_object_scale | 78% | 84% |
| place_object_stand | 97% | 94% |
| place_phone_stand | 66% | 73% |
| press_stapler | 98% | 98% |
| shake_bottle | 89% | 85% |
| shake_bottle_horizontally | 90% | 90% |

## 目录内容

```text
assets/embodiments/tianji/
├── config.yml                         # RoboTwin 读取的机器人主配置
├── marvin_robot.urdf                  # SAPIEN / 运动学使用的 URDF
├── marvin_moveit_config/              # MoveIt / SRDF 配置
├── curobo_left_tmp.yml                # cuRobo 左臂路径模板
├── curobo_right_tmp.yml               # cuRobo 右臂路径模板
├── curobo_left.yml                    # 由模板生成的左臂 cuRobo 配置
├── curobo_right.yml                   # 由模板生成的右臂 cuRobo 配置
├── collision_left.yml                 # cuRobo 左臂碰撞球配置
├── collision_right.yml                # cuRobo 右臂碰撞球配置
├── meshes/                            # 机器人网格
└── configuration/, *.usd, *.usda       # Isaac / USD 相关资产
```

> 注意：文件名中保留了 `marvin_robot`，但在 RoboTwin 任务配置中使用的 embodiment 名称是 `tianji`。

## 0. 安装 RoboTwin 环境

在使用 Tianji / Marin Pro 机器人配置前，需要先按照 RoboTwin 官方教程完成基础环境、依赖和资产安装。具体可参考 RoboTwin 2.0 官方文档的 Install & Download 页面：

https://robotwin-platform.github.io/doc/usage/robotwin-install.html

## 1. 确认机器人已注册

RoboTwin 通过 `task_config/_embodiment_config.yml` 查找可用机器人。当前仓库已经包含如下注册项：

```yaml
tianji:
  file_path: "./assets/embodiments/tianji"
```
如果直接clone该仓库，则`task_config/_embodiment_config.yml`中已经存在该注册项。
如果复制的是RoboTwin官方仓库，则不存在上述注册项，请往‘task_config/_embodiment_config.yml’中手动添加上述字段。

## 2. 首次运行前更新 cuRobo 绝对路径

`curobo_left.yml` 和 `curobo_right.yml` 中需要写入当前机器上的仓库绝对路径。换机器、换目录或重新 clone 后，在仓库根目录执行：

```bash
python script/update_embodiment_config_path.py
```

该脚本会读取 `*_tmp.yml` 模板，把 `${ASSETS_PATH}` 替换为当前仓库路径，并生成对应的 `curobo_left.yml` / `curobo_right.yml`。生成后脚本会检查 URDF 和碰撞球配置是否存在。如果不执行这一步，cuRobo 可能仍指向旧机器路径，导致规划器找不到 URDF 或碰撞配置。

## 3. 选择使用 Tianji 机器人

数据采集使用 `task_config/_embodiment_config.yml` 中的 `embodiment` 字段决定机器人。要使用 Marin Pro，将任务配置写成：

```yaml
embodiment:
- tianji
```

当前 `task_config/demo_randomized.yml` 已经配置为：

```yaml
render_freq: 0
episode_num: 1500
use_seed: false
save_freq: 15
need_topp: false
embodiment:
- tianji
language_num: 100
domain_randomization:
  random_background: true
  cluttered_table: true
  clean_background_rate: 0.02
  random_head_camera_dis: 0
  random_table_height: 0.03
  random_light: true
  crazy_random_light_rate: 0.02
camera:
  head_camera_type: D435_Tianji_head
  wrist_camera_type: D435_Tianji_wrist
  collect_head_camera: true
  collect_wrist_camera: true
data_type:
  rgb: true
  third_view: false
  depth: false
  pointcloud: false
  observer: false
  endpose: true
  qpos: true
  mesh_segmentation: false
  actor_segmentation: false
pcd_down_sample_num: 1024
pcd_crop: true
save_path: ./data
clear_cache_freq: 5
collect_data: true
eval_video_log: true
```

常用字段说明：

- `episode_num`: 需要成功采集的 episode 数量。
- `use_seed`: `false` 时先搜索成功 seed 并生成轨迹；`true` 时复用已有 `seed.txt`。
- `collect_data`: 是否生成最终 HDF5 数据。
- `camera`: 选择头部相机、腕部相机类型，以及是否采集对应图像。
- `data_type`: 控制保存 RGB、深度、点云、末端位姿、关节动作等数据。
- `domain_randomization`: 控制背景、光照、桌面杂物、桌高、头部相机位置扰动等随机化。

## 4. 运行数据采集

在仓库根目录执行：

```bash
bash collect_data.sh <task_name> <task_config_name> <gpu_id>
```

示例：

```bash
bash collect_data.sh beat_block_hammer demo_randomized 0
```

参数含义：

- `task_name`: 任务名，对应 `envs/<task_name>.py`，例如 `beat_block_hammer`、`move_can_pot`、`stack_blocks_two`。
- `task_config_name`: 配置文件名，不包含 `.yml` 后缀，例如 `demo_randomized`。
- `gpu_id`: 使用的 GPU 编号，会写入 `CUDA_VISIBLE_DEVICES`。

采集流程分两步：

1. 搜索能够成功完成任务的随机 seed，并在 `_traj_data/` 中保存规划轨迹。
2. 复用成功 seed 回放轨迹，保存 HDF5、视频、场景信息和语言指令。

## 5. 数据输出位置

以上示例会输出到：

```text
data/beat_block_hammer/demo_randomized/
├── _traj_data/              # 规划阶段保存的轨迹 pkl
├── data/                    # 最终 episodeN.hdf5 数据
├── video/                   # 可视化视频
├── instructions/            # episode 语言指令
├── seed.txt                 # 成功 seed 列表
└── scene_info.json          # 每个 episode 的场景信息
```

单个 HDF5 通常包含：

```text
endpose/
joint_action/
observation/head_camera/
observation/left_camera/
observation/right_camera/
pointcloud
```

其中 `joint_action/vector` 为双臂关节与夹爪动作拼接结果，`observation/*/rgb` 为编码后的相机图像，`observation/*/intrinsic_cv` 和 `observation/*/extrinsic_cv` 为相机内外参。

## 6. 机器人关键配置

`assets/embodiments/tianji/config.yml` 是 RoboTwin 加载 Marin Pro 的主入口，重点字段如下：

- `urdf_path`: 当前使用 `./marvin_robot.urdf`。
- `srdf_path`: 当前使用 `./marvin_moveit_config/config/marvin_robot.srdf`。
- `planner`: 当前为 `curobo`。
- `dual_arm`: 当前为 `True`，表示单个 URDF 中包含双臂。
- `arm_joints_name`: 左右臂各 7 个关节名称。
- `move_group`: 左右末端 link，当前为 `left_tool`、`right_tool`。
- `gripper_name`: 左右夹爪主关节及 mimic 关节。
- `homestate`: 双臂初始关节姿态。
- `robot_pose`: 机器人在场景中的根位姿。
- `wrist_camera_pose`: 腕部相机相对末端的位姿。
- `static_camera_list`: 静态头部相机位姿，当前包含 `head_camera`。
- `disable_collision_pairs`: 需要在 SAPIEN 中忽略碰撞的局部 link 对。

如果调整 URDF、相机、夹爪或末端 link，请同步检查 `config.yml`、`curobo_left_tmp.yml`、`curobo_right_tmp.yml`、`collision_left.yml` 和 `collision_right.yml`。

## 7. 常见问题

### cuRobo 报 URDF 或 collision 配置找不到

通常是 `curobo_left.yml` / `curobo_right.yml` 中仍是旧机器的绝对路径。重新执行：

```bash
python script/update_embodiment_config_path.py
```

### 修改任务配置后没有生效

运行命令中的第二个参数是配置名，不带 `.yml`。例如配置文件是 `task_config/demo_randomized.yml`，命令应写：

```bash
bash collect_data.sh beat_block_hammer demo_randomized 0
```

### 想快速测试流程

先把目标配置中的 `episode_num` 改小，例如 `1` 或 `2`，确认 seed 搜索、轨迹回放、HDF5 输出都正常后，再扩大采集规模。

### 想复用已有成功 seed

保留 `data/<task>/<config>/seed.txt`，并在任务配置中设置：

```yaml
use_seed: true
```

这样会跳过 seed 搜索，直接根据已有 seed 回放采集数据。

## 8. 推荐采集顺序

1. 确认 `task_config/_embodiment_config.yml` 中存在 `tianji`。
2. 执行 `python script/update_embodiment_config_path.py` 更新本机路径。
3. 复制或修改一个任务配置，将 `embodiment` 设置为 `tianji`。
4. 先用少量 `episode_num` 试跑目标任务。
5. 检查 `data/<task>/<config>/data/episode0.hdf5`、`video/` 和 `scene_info.json`。
6. 扩大 `episode_num`，正式批量采集。
