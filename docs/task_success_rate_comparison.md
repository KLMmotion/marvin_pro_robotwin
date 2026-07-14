# RoboTwin 任务数据生成成功率对比

对比依据为 RoboTwin 2.0 官方文档中各任务的 **Data Generation Success Rate（Aloha-AgileX）**，以及本仓库 Tianji / Marin Pro embodiment 上实测的数据生成成功率。

原始采集数据：

- Hugging Face Dataset: [Continuity3/marvin_pro_robotwin](https://huggingface.co/datasets/Continuity3/marvin_pro_robotwin)
- 数据路径：[`data/`](https://huggingface.co/datasets/Continuity3/marvin_pro_robotwin/tree/main/data)

官方任务页：https://robotwin-platform.github.io/doc/tasks/

| Task | 官方成功率 (Aloha-AgileX) | 我们实测成功率 (Tianji) | 对比 | Hugging Face 数据路径 |
| --- | ---: | ---: | --- | --- |
| adjust_bottle | 93% | 93% | 持平 | [`data/adjust_bottle`](https://huggingface.co/datasets/Continuity3/marvin_pro_robotwin/tree/main/data/adjust_bottle) |
| beat_block_hammer | 64% | 71% | 高于官方 | [`data/beat_block_hammer`](https://huggingface.co/datasets/Continuity3/marvin_pro_robotwin/tree/main/data/beat_block_hammer) |
| click_alarmclock | 92% | 88% | 低于官方 | [`data/click_alarmclock`](https://huggingface.co/datasets/Continuity3/marvin_pro_robotwin/tree/main/data/click_alarmclock) |
| grab_roller | 95% | 91% | 低于官方 | [`data/grab_roller`](https://huggingface.co/datasets/Continuity3/marvin_pro_robotwin/tree/main/data/grab_roller) |
| lift_pot | 27% | 34% | 高于官方 | [`data/lift_pot`](https://huggingface.co/datasets/Continuity3/marvin_pro_robotwin/tree/main/data/lift_pot) |
| move_playingcard_away | 99% | 95% | 低于官方 | [`data/move_playingcard_away`](https://huggingface.co/datasets/Continuity3/marvin_pro_robotwin/tree/main/data/move_playingcard_away) |
| move_stapler_pad | 92% | 92% | 持平 | [`data/move_stapler_pad`](https://huggingface.co/datasets/Continuity3/marvin_pro_robotwin/tree/main/data/move_stapler_pad) |
| open_laptop | 82% | 87% | 高于官方 | [`data/open_laptop`](https://huggingface.co/datasets/Continuity3/marvin_pro_robotwin/tree/main/data/open_laptop) |
| place_burger_fries | 97% | 93% | 低于官方 | [`data/place_burger_fries`](https://huggingface.co/datasets/Continuity3/marvin_pro_robotwin/tree/main/data/place_burger_fries) |
| place_container_plate | 89% | 89% | 持平 | [`data/place_container_plate`](https://huggingface.co/datasets/Continuity3/marvin_pro_robotwin/tree/main/data/place_container_plate) |
| place_empty_cup | 92% | 96% | 高于官方 | [`data/place_empty_cup`](https://huggingface.co/datasets/Continuity3/marvin_pro_robotwin/tree/main/data/place_empty_cup) |
| place_object_scale | 78% | 84% | 高于官方 | [`data/place_object_scale`](https://huggingface.co/datasets/Continuity3/marvin_pro_robotwin/tree/main/data/place_object_scale) |
| place_object_stand | 97% | 94% | 低于官方 | [`data/place_object_stand`](https://huggingface.co/datasets/Continuity3/marvin_pro_robotwin/tree/main/data/place_object_stand) |
| place_phone_stand | 66% | 73% | 高于官方 | [`data/place_phone_stand`](https://huggingface.co/datasets/Continuity3/marvin_pro_robotwin/tree/main/data/place_phone_stand) |
| press_stapler | 98% | 98% | 持平 | [`data/press_stapler`](https://huggingface.co/datasets/Continuity3/marvin_pro_robotwin/tree/main/data/press_stapler) |
| shake_bottle | 89% | 85% | 低于官方 | [`data/shake_bottle`](https://huggingface.co/datasets/Continuity3/marvin_pro_robotwin/tree/main/data/shake_bottle) |
| shake_bottle_horizontally | 90% | 90% | 持平 | [`data/shake_bottle_horizontally`](https://huggingface.co/datasets/Continuity3/marvin_pro_robotwin/tree/main/data/shake_bottle_horizontally) |

## 说明

- **官方成功率**：来自 RoboTwin 2.0 任务文档的 Data Generation Success Rate（Aloha-AgileX 列）。
- **我们实测成功率**：Tianji / Marin Pro 在相同任务定义下的数据生成成功率（seed 搜索阶段成功比例）。
- 各任务原始 HDF5 / instructions / seed 等见 Hugging Face 对应子目录。
