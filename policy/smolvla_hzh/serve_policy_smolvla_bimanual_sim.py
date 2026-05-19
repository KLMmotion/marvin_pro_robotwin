#!/usr/bin/env python3
"""SmolVLA 双臂仿真专用 WebSocket policy server。

这个脚本运行在 policy/LeRobot 环境中，职责很窄：
1. 加载 LeRobot 格式的 SmolVLA checkpoint；
2. 接收 RoboTwin client 通过 OpenPI websocket 协议发来的 observation；
3. 按你转换脚本的双臂仿真数据格式组 batch；
4. 返回模型预测的 14 维双臂 EEF delta action chunk。

注意：这里不把 delta action 转成 RoboTwin 的 absolute ee action。这个转换依赖
RoboTwin 当前环境 state，应该留在 RoboTwin client/deploy_policy.py 里完成。
"""

from __future__ import annotations

import argparse
import logging
import os
import pathlib
import socket
import sys
import time
from typing import Any

import numpy as np
import torch


IMAGE_KEYS_BY_STYLE = {
    # 和 convert_hdf5_to_new_format_simulation_bimanual.py 的默认 key 一致。
    "generic": {
        "main": "observation.images.image",
        "left_wrist": "observation.images.left_wrist_image",
        "right_wrist": "observation.images.right_wrist_image",
    },
    # 兼容 ALOHA 风格命名。
    "aloha": {
        "main": "observation.images.cam_high",
        "left_wrist": "observation.images.cam_left_wrist",
        "right_wrist": "observation.images.cam_right_wrist",
    },
}

DEFAULT_OPENPI_ROOT = "/home/tianji/hzh/study/openpi"
DEFAULT_WRIST_VIS_DIR = "debug_outputs/smolvla_wrist_server"


def _env_flag(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in ("1", "true", "yes", "y", "on")


def _env_int(name: str, default: int) -> int:
    value = os.environ.get(name)
    if value is None:
        return default
    try:
        return int(value)
    except ValueError:
        logging.warning("Invalid integer env %s=%r, using %d", name, value, default)
        return default


def _insert_path(path: str | os.PathLike[str] | None) -> None:
    if not path:
        return
    path = str(pathlib.Path(path).expanduser().resolve())
    if path not in sys.path:
        sys.path.insert(0, path)


def _setup_import_paths(openpi_repo_root: str | None, lerobot_repo_root: str | None) -> None:
    """把 openpi/lerobot 源码路径加到 sys.path。

    这个 server 通常在 `lerobot` conda 环境里启动，但 OpenPI 的 websocket
    server/client 包可能没有 pip install，所以这里支持直接指向源码目录。
    """
    openpi_root = openpi_repo_root or os.environ.get("OPENPI_ROOT")
    if openpi_root is None and pathlib.Path(DEFAULT_OPENPI_ROOT).exists():
        openpi_root = DEFAULT_OPENPI_ROOT

    if openpi_root:
        root = pathlib.Path(openpi_root).expanduser().resolve()
        _insert_path(root / "src")
        _insert_path(root / "packages" / "openpi-client" / "src")
        _insert_path(root)

    if lerobot_repo_root:
        root = pathlib.Path(lerobot_repo_root).expanduser().resolve()
        candidates = [
            root / "src",
            root / "lerobot" / "src",
            root / "lerobot",
            root,
        ]
        for candidate in candidates:
            if candidate.exists():
                _insert_path(candidate)


def _import_openpi_websocket_modules():
    try:
        from openpi.serving import websocket_policy_server
        from openpi_client import base_policy as base_policy
    except ImportError as exc:
        raise ImportError(
            "无法导入 OpenPI websocket 模块。请确认已经安装 openpi/openpi-client，"
            "或者启动时传入 --openpi-repo-root /path/to/openpi。"
        ) from exc
    return websocket_policy_server, base_policy


def _load_pretrained_config_class():
    try:
        from lerobot.configs.policies import PreTrainedConfig

        return PreTrainedConfig
    except Exception as first_error:
        try:
            from lerobot.common.policies.pretrained import PreTrainedConfig

            return PreTrainedConfig
        except Exception as second_error:
            raise ImportError(
                "无法导入 LeRobot PreTrainedConfig。请在 SmolVLA/LeRobot 环境中运行，"
                "或用 --lerobot-repo-root 指向正确源码。"
            ) from second_error or first_error


def _resolve_policy_class(policy_type: str, explicit_policy_class: str | None):
    """根据 checkpoint config.type 找到对应的 policy class。"""
    if explicit_policy_class:
        module_name, class_name = explicit_policy_class.split(":", maxsplit=1)
        module = __import__(module_name, fromlist=[class_name])
        return getattr(module, class_name)

    errors = []
    for factory_path in (
        "lerobot.policies.factory",
        "lerobot.common.policies.factory",
    ):
        try:
            module = __import__(factory_path, fromlist=["get_policy_class"])
            return module.get_policy_class(policy_type)
        except Exception as exc:
            errors.append(f"{factory_path}: {exc}")

    raise RuntimeError(
        "无法根据 config.type 自动解析 policy class。"
        f" policy_type={policy_type!r}。可以用 --policy-class module.path:ClassName 显式指定。"
        f" 尝试过的 factory 错误：{errors}"
    )


def _as_uint8_hwc(image: Any) -> np.ndarray:
    """把 websocket 收到的图像整理成 HWC uint8 RGB。"""
    image = np.asarray(image)
    if image.ndim == 2:
        image = np.repeat(image[:, :, None], 3, axis=2)
    elif image.ndim == 3 and image.shape[0] in (1, 3, 4) and image.shape[-1] not in (3, 4):
        # 兼容误传 CHW 的情况。
        image = np.transpose(image, (1, 2, 0))
    elif image.ndim != 3:
        raise ValueError(f"图像必须是 HWC/CHW 3 维数组，实际 shape={image.shape}")

    if image.shape[-1] == 1:
        image = np.repeat(image, 3, axis=2)
    elif image.shape[-1] == 4:
        image = image[:, :, :3]
    elif image.shape[-1] != 3:
        raise ValueError(f"图像通道数必须是 1/3/4，实际 shape={image.shape}")

    if image.dtype != np.uint8:
        image = image.astype(np.float32)
        if np.nanmax(image) <= 1.0:
            image = image * 255.0
        image = np.clip(image, 0, 255).astype(np.uint8)
    return np.ascontiguousarray(image)


def _image_to_tensor(image: Any) -> torch.Tensor:
    """HWC uint8/float image -> [C, H, W] float tensor in 0~1.

    LeRobot 的 policy_preprocessor.json 里已经包含 to_batch_processor 和
    device_processor，所以这里不要提前加 batch 维，也不要提前搬到 cuda。
    """
    image = _as_uint8_hwc(image).astype(np.float32) / 255.0
    image = np.transpose(image, (2, 0, 1))
    return torch.from_numpy(np.ascontiguousarray(image))


class WristCameraVisualizer:
    """Save or display the wrist images exactly as the websocket server receives them."""

    def __init__(
        self,
        *,
        enabled: bool,
        output_dir: str | os.PathLike[str] | None,
        every_n: int,
        show: bool,
    ) -> None:
        self.enabled = enabled
        self.show = show
        self.every_n = max(1, int(every_n))
        self.counter = 0
        self.window_name = "SmolVLA server wrist cameras"
        self.output_dir = pathlib.Path(output_dir).expanduser() if output_dir else None
        self._cv2 = None

        if not self.enabled:
            return

        try:
            import cv2
        except ImportError as exc:
            raise ImportError(
                "启用腕部图像可视化需要 cv2。请在 server 环境安装 opencv-python，"
                "或关闭 --visualize-wrist-cameras。"
            ) from exc
        self._cv2 = cv2

        if self.output_dir is not None:
            self.output_dir.mkdir(parents=True, exist_ok=True)

    def update(self, obs: dict[str, Any], image_keys: dict[str, str], task: str) -> None:
        if not self.enabled:
            return

        self.counter += 1
        if (self.counter - 1) % self.every_n != 0:
            return

        try:
            panel_bgr = self._build_panel_bgr(
                obs[image_keys["left_wrist"]],
                obs[image_keys["right_wrist"]],
                task,
            )
            if self.output_dir is not None:
                latest_path = self.output_dir / "latest_wrist_cameras.jpg"
                frame_path = self.output_dir / f"wrist_cameras_{self.counter:06d}.jpg"
                self._cv2.imwrite(str(latest_path), panel_bgr)
                self._cv2.imwrite(str(frame_path), panel_bgr)
            if self.show:
                self._cv2.imshow(self.window_name, panel_bgr)
                self._cv2.waitKey(1)
        except Exception as exc:
            logging.warning("Wrist camera visualization failed; disabling it: %s", exc)
            self.enabled = False

    def _build_panel_bgr(self, left_image: Any, right_image: Any, task: str) -> np.ndarray:
        left_rgb = _as_uint8_hwc(left_image)
        right_rgb = _as_uint8_hwc(right_image)

        if left_rgb.shape[0] != right_rgb.shape[0]:
            target_h = left_rgb.shape[0]
            target_w = max(1, round(right_rgb.shape[1] * target_h / right_rgb.shape[0]))
            right_rgb = self._cv2.resize(right_rgb, (target_w, target_h), interpolation=self._cv2.INTER_AREA)

        panel_rgb = np.concatenate([left_rgb, right_rgb], axis=1)
        panel_bgr = self._cv2.cvtColor(panel_rgb, self._cv2.COLOR_RGB2BGR)

        left_w = left_rgb.shape[1]
        stamp = time.strftime("%Y-%m-%d %H:%M:%S")
        self._cv2.putText(
            panel_bgr,
            f"left_wrist #{self.counter}",
            (8, 24),
            self._cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2,
            self._cv2.LINE_AA,
        )
        self._cv2.putText(
            panel_bgr,
            "right_wrist",
            (left_w + 8, 24),
            self._cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2,
            self._cv2.LINE_AA,
        )
        self._cv2.putText(
            panel_bgr,
            stamp,
            (8, max(48, panel_bgr.shape[0] - 12)),
            self._cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
            self._cv2.LINE_AA,
        )
        if task:
            self._cv2.putText(
                panel_bgr,
                task[:80],
                (8, max(72, panel_bgr.shape[0] - 34)),
                self._cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1,
                self._cv2.LINE_AA,
            )
        return panel_bgr


def _state_to_tensor(state: Any) -> torch.Tensor:
    state = np.asarray(state, dtype=np.float32)
    if state.ndim != 1 or state.shape[-1] != 16:
        raise ValueError(f"双臂仿真 SmolVLA 需要 observation.state shape=(16,)，实际 {state.shape}")
    return torch.from_numpy(np.ascontiguousarray(state))


def _get_task(obs: dict[str, Any], default_task: str) -> str:
    # 同时兼容训练 key `task` 和 OpenPI 常见 key `prompt`。
    task = obs.get("task", obs.get("prompt", default_task))
    if isinstance(task, (list, tuple)):
        task = task[0] if task else default_task
    return "" if task is None else str(task)


def _normalize_actions(action: Any, max_action_steps: int | None) -> np.ndarray:
    """把 LeRobot policy 输出统一成 [T, 14] float32。"""
    if isinstance(action, dict):
        if "action" in action:
            action = action["action"]
        elif "actions" in action:
            action = action["actions"]
        else:
            action = next(iter(action.values()))

    if isinstance(action, torch.Tensor):
        action = action.detach().cpu().numpy()
    action = np.asarray(action, dtype=np.float32)

    if action.ndim == 1:
        actions = action[None, :]
    elif action.ndim == 2:
        actions = action
    elif action.ndim == 3:
        actions = action[0]
    else:
        raise ValueError(f"不支持的 action 输出维度：shape={action.shape}")

    if actions.shape[-1] != 14:
        raise ValueError(f"双臂仿真 SmolVLA 需要 14 维 delta action，实际 shape={actions.shape}")
    if max_action_steps is not None and max_action_steps > 0:
        actions = actions[:max_action_steps]
    return np.ascontiguousarray(actions, dtype=np.float32)


def _policy_device(policy, fallback: str) -> str:
    if hasattr(policy, "parameters"):
        try:
            return str(next(policy.parameters()).device)
        except StopIteration:
            pass
    return fallback


def _load_lerobot_policy(
    checkpoint_dir: str,
    *,
    device: str,
    policy_class: str | None,
    tokenizer_local_files_only: bool,
):
    """加载 SmolVLA/LeRobot pretrained policy 以及配套 pre/post processor。"""
    checkpoint = pathlib.Path(checkpoint_dir).expanduser().resolve()
    if not checkpoint.exists():
        raise FileNotFoundError(f"checkpoint_dir 不存在：{checkpoint}")

    # 有些 LeRobot 版本会在 import lerobot.policies 时注册 config/policy 子类。
    try:
        import lerobot.policies  # noqa: F401
    except Exception:
        try:
            import lerobot.common.policies  # noqa: F401
        except Exception:
            pass

    PreTrainedConfig = _load_pretrained_config_class()
    cfg = PreTrainedConfig.from_pretrained(str(checkpoint))
    cfg.device = device

    policy_cls = _resolve_policy_class(cfg.type, policy_class)
    policy = policy_cls.from_pretrained(str(checkpoint), config=cfg)
    if hasattr(policy, "to"):
        policy.to(device)
    policy.eval()

    from lerobot.policies.factory import make_pre_post_processors

    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=cfg,
        pretrained_path=str(checkpoint),
        preprocessor_overrides={
            "device_processor": {"device": device},
            "tokenizer_processor": {"local_files_only": tokenizer_local_files_only},
        },
    )

    logging.info("Loaded policy type=%s class=%s checkpoint=%s", cfg.type, policy_cls.__name__, checkpoint)
    try:
        import lerobot

        logging.info("Using lerobot from: %s", getattr(lerobot, "__file__", "unknown"))
    except Exception:
        pass
    return policy, cfg, preprocessor, postprocessor


def build_adapter_class(base_policy_class):
    class SmolVLABimanualSimWebsocketPolicy(base_policy_class):
        """OpenPI websocket server 需要的 BasePolicy 适配器。"""

        def __init__(
            self,
            policy,
            preprocessor,
            postprocessor,
            *,
            device: str,
            image_key_style: str,
            default_task: str,
            max_action_steps: int | None,
            wrist_visualizer: WristCameraVisualizer | None,
        ) -> None:
            self.policy = policy
            self.preprocessor = preprocessor
            self.postprocessor = postprocessor
            self.device = device
            self.image_key_style = image_key_style
            self.image_keys = IMAGE_KEYS_BY_STYLE[image_key_style]
            self.default_task = default_task
            self.max_action_steps = max_action_steps
            self.wrist_visualizer = wrist_visualizer
            self.metadata = {
                "adapter": "smolvla_bimanual_sim",
                "device": device,
                "image_key_style": image_key_style,
                "state_key": "observation.state",
                "image_keys": self.image_keys,
                "action_key": "actions",
                "action_shape": ["T", 14],
            }

        def reset(self) -> None:
            if hasattr(self.policy, "reset"):
                self.policy.reset()

        @torch.inference_mode()
        def infer(self, obs: dict[str, Any]) -> dict[str, np.ndarray]:
            if not isinstance(obs, dict):
                raise TypeError(f"websocket obs 必须是 dict，实际 {type(obs)}")

            # OpenPI 原始 websocket client 的 reset() 是空实现。为了让 RoboTwin
            # 每个 episode 开始时仍然能清理 policy 内部缓存，我们约定 client 可以
            # 发送 {"__reset__": True} 作为轻量 reset 消息。
            if obs.get("__reset__", False):
                self.reset()
                return {"actions": np.zeros((0, 14), dtype=np.float32)}

            missing = [key for key in self.image_keys.values() if key not in obs]
            if "observation.state" not in obs:
                missing.append("observation.state")
            if missing:
                raise KeyError(
                    "websocket obs 缺少 SmolVLA 双臂仿真输入 key："
                    f"{missing}。当前收到 keys={list(obs.keys())}"
                )

            task = _get_task(obs, self.default_task)
            if self.wrist_visualizer is not None:
                self.wrist_visualizer.update(obs, self.image_keys, task)
            raw_batch = {
                "observation.state": _state_to_tensor(obs["observation.state"]),
                "task": [task],
                self.image_keys["main"]: _image_to_tensor(obs[self.image_keys["main"]]),
                self.image_keys["left_wrist"]: _image_to_tensor(obs[self.image_keys["left_wrist"]]),
                self.image_keys["right_wrist"]: _image_to_tensor(obs[self.image_keys["right_wrist"]]),
            }

            # 官方 LeRobot 推理流程：
            # raw observation/task -> preprocessor(tokenize/normalize/device)
            # -> policy inference -> postprocessor(unnormalize action)
            batch = self.preprocessor(raw_batch)
            if hasattr(self.policy, "predict_action_chunk"):
                action = self.policy.predict_action_chunk(batch)
            else:
                action = self.policy.select_action(batch)
            action = self.postprocessor(action)
            actions = _normalize_actions(action, self.max_action_steps)
            return {"actions": actions}

    return SmolVLABimanualSimWebsocketPolicy


def _warmup(adapter, image_size: int) -> None:
    obs = {
        "observation.state": np.zeros(16, dtype=np.float32),
        adapter.image_keys["main"]: np.zeros((image_size, image_size, 3), dtype=np.uint8),
        adapter.image_keys["left_wrist"]: np.zeros((image_size, image_size, 3), dtype=np.uint8),
        adapter.image_keys["right_wrist"]: np.zeros((image_size, image_size, 3), dtype=np.uint8),
        "task": adapter.default_task,
    }
    try:
        out = adapter.infer(obs)
        logging.info("Warmup ok, actions shape=%s", out["actions"].shape)
    except Exception as exc:
        logging.warning("Warmup failed, continue serving anyway: %s", exc)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Serve SmolVLA bimanual simulation policy over OpenPI websocket.")
    parser.add_argument("--checkpoint-dir", required=True, help="LeRobot/SmolVLA checkpoint 目录。")
    parser.add_argument("--host", default="0.0.0.0", help="websocket server 监听地址。")
    parser.add_argument("--port", type=int, default=8000, help="websocket server 监听端口。")
    parser.add_argument("--device", default=None, help="例如 cuda、cuda:0、cpu；默认自动选择。")
    parser.add_argument("--image-key-style", choices=sorted(IMAGE_KEYS_BY_STYLE), default="generic")
    parser.add_argument("--max-action-steps", type=int, default=8, help="返回 action chunk 的前多少步；<=0 表示不截断。")
    parser.add_argument("--default-task", default="", help="请求里没有 task/prompt 时使用的默认语言指令。")
    parser.add_argument("--openpi-repo-root", default=None, help="OpenPI 源码根目录；默认读取 OPENPI_ROOT 或常用路径。")
    parser.add_argument("--lerobot-repo-root", default=None, help="可选：自定义 LeRobot/SmolVLA 源码根目录。")
    parser.add_argument("--policy-class", default=None, help="可选：显式 policy 类，格式 module.path:ClassName。")
    parser.add_argument(
        "--tokenizer-local-files-only",
        action="store_true",
        help="加载 tokenizer 时只使用本地缓存，避免访问 HuggingFace。默认允许按 checkpoint 配置自动下载。",
    )
    parser.add_argument("--no-warmup", action="store_true", help="启动时跳过一次 dummy forward。")
    parser.add_argument("--image-size", type=int, default=256, help="warmup dummy 图像尺寸，真实请求不在 server 里 resize。")
    parser.add_argument(
        "--visualize-wrist-cameras",
        action="store_true",
        default=_env_flag("SMOLVLA_VISUALIZE_WRIST_CAMERAS"),
        help=(
            "可视化 server 收到的左右腕部图像。也可设置环境变量 "
            "SMOLVLA_VISUALIZE_WRIST_CAMERAS=1。"
        ),
    )
    parser.add_argument(
        "--wrist-visualize-dir",
        default=os.environ.get("SMOLVLA_WRIST_VIS_DIR", DEFAULT_WRIST_VIS_DIR),
        help=(
            "保存左右腕部拼图的目录；默认 debug_outputs/smolvla_wrist_server。"
            "设置为空字符串可只显示不保存。"
        ),
    )
    parser.add_argument(
        "--wrist-visualize-every-n",
        type=int,
        default=_env_int("SMOLVLA_WRIST_VIS_EVERY_N", 1),
        help="每隔多少次 infer 保存/显示一帧；默认 1。也可用 SMOLVLA_WRIST_VIS_EVERY_N 设置。",
    )
    parser.add_argument(
        "--show-wrist-cameras",
        action="store_true",
        default=_env_flag("SMOLVLA_SHOW_WRIST_CAMERAS"),
        help="同时用 cv2.imshow 弹窗显示；远程/headless 环境通常只建议保存图片。",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s: %(message)s", force=True)

    _setup_import_paths(args.openpi_repo_root, args.lerobot_repo_root)
    websocket_policy_server, base_policy = _import_openpi_websocket_modules()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    max_action_steps = args.max_action_steps if args.max_action_steps > 0 else None
    policy, _cfg, preprocessor, postprocessor = _load_lerobot_policy(
        args.checkpoint_dir,
        device=device,
        policy_class=args.policy_class,
        tokenizer_local_files_only=args.tokenizer_local_files_only,
    )
    device = _policy_device(policy, device)

    adapter_cls = build_adapter_class(base_policy.BasePolicy)
    adapter = adapter_cls(
        policy,
        preprocessor,
        postprocessor,
        device=device,
        image_key_style=args.image_key_style,
        default_task=args.default_task,
        max_action_steps=max_action_steps,
        wrist_visualizer=None,
    )

    if not args.no_warmup:
        _warmup(adapter, args.image_size)

    adapter.wrist_visualizer = WristCameraVisualizer(
        enabled=args.visualize_wrist_cameras,
        output_dir=args.wrist_visualize_dir or None,
        every_n=args.wrist_visualize_every_n,
        show=args.show_wrist_cameras,
    )

    hostname = socket.gethostname()
    try:
        local_ip = socket.gethostbyname(hostname)
    except socket.gaierror:
        local_ip = "unknown"

    logging.info("Serving SmolVLA bimanual sim websocket policy")
    logging.info("Host=%s local_ip=%s port=%d", args.host, local_ip, args.port)
    logging.info("Expected image keys=%s", adapter.image_keys)
    logging.info("Expected state key=observation.state shape=(16,), output actions shape=(T,14)")
    if args.visualize_wrist_cameras:
        logging.info(
            "Wrist camera visualization enabled: dir=%s every_n=%d show=%s",
            args.wrist_visualize_dir or "<disabled>",
            max(1, args.wrist_visualize_every_n),
            args.show_wrist_cameras,
        )

    server = websocket_policy_server.WebsocketPolicyServer(
        policy=adapter,
        host=args.host,
        port=args.port,
        metadata=adapter.metadata,
    )
    server.serve_forever()


if __name__ == "__main__":
    main()
