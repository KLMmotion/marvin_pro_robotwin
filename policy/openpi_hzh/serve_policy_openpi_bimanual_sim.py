#!/usr/bin/env python3
"""OpenPI/pi0.5 bimanual simulation WebSocket policy server.

This server loads an OpenPI checkpoint, receives RoboTwin observations through the
OpenPI websocket protocol, and returns 14D bimanual EEF delta action chunks.

By default it patches the selected OpenPI train config at inference time with a
bimanual transform matching:

    observation.images.image
    observation.images.left_wrist_image
    observation.images.right_wrist_image
    observation.state

This matches the dataset produced by:
`/home/tianji/hzh/study/openpi/convert_hdf5_to_new_format_simulation_bimanual_openpi.py`.
"""

from __future__ import annotations

import argparse
import dataclasses
import logging
import os
import pathlib
import socket
import sys
import time
from typing import Any

import numpy as np


IMAGE_KEYS_BY_STYLE = {
    "generic": {
        "main": "observation.images.image",
        "left_wrist": "observation.images.left_wrist_image",
        "right_wrist": "observation.images.right_wrist_image",
    },
    "aloha": {
        "main": "observation.images.cam_high",
        "left_wrist": "observation.images.cam_left_wrist",
        "right_wrist": "observation.images.cam_right_wrist",
    },
}

DEFAULT_OPENPI_ROOT = "/home/tianji/hzh/study/openpi"
DEFAULT_WRIST_VIS_DIR = "debug_outputs/openpi_wrist_server"


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


def _setup_import_paths(openpi_repo_root: str | None) -> None:
    openpi_root = openpi_repo_root or os.environ.get("OPENPI_ROOT")
    if openpi_root is None and pathlib.Path(DEFAULT_OPENPI_ROOT).exists():
        openpi_root = DEFAULT_OPENPI_ROOT

    if openpi_root:
        root = pathlib.Path(openpi_root).expanduser().resolve()
        _insert_path(root / "src")
        _insert_path(root / "packages" / "openpi-client" / "src")
        _insert_path(root)


def _import_openpi_modules():
    try:
        from openpi import transforms
        from openpi.models import model as openpi_model
        from openpi.policies import policy_config
        from openpi.serving import websocket_policy_server
        from openpi.training import config as training_config
        from openpi_client import base_policy
    except ImportError as exc:
        raise ImportError(
            "无法导入 OpenPI 模块。请在 OpenPI 环境中运行，或传入 "
            "--openpi-repo-root /path/to/openpi。"
        ) from exc
    return transforms, openpi_model, policy_config, websocket_policy_server, training_config, base_policy


def _as_uint8_hwc(image: Any) -> np.ndarray:
    image = np.asarray(image)
    if image.ndim == 2:
        image = np.repeat(image[:, :, None], 3, axis=2)
    elif image.ndim == 3 and image.shape[0] in (1, 3, 4) and image.shape[-1] not in (3, 4):
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


def _state_array(state: Any) -> np.ndarray:
    state = np.asarray(state, dtype=np.float32)
    if state.ndim != 1 or state.shape[-1] != 16:
        raise ValueError(f"OpenPI 双臂仿真需要 observation.state shape=(16,)，实际 {state.shape}")
    return np.ascontiguousarray(state)


def _get_first(data: dict[str, Any], keys: tuple[str, ...]) -> Any:
    for key in keys:
        if key in data:
            return data[key]
    raise KeyError(f"缺少任一 key: {keys}，当前 keys={list(data.keys())}")


def _get_task(obs: dict[str, Any], default_task: str) -> str:
    task = obs.get("task", obs.get("prompt", default_task))
    if isinstance(task, (list, tuple)):
        task = task[0] if task else default_task
    if isinstance(task, np.ndarray):
        task = task.item() if task.ndim == 0 else task.tolist()[0]
    return "" if task is None else str(task)


def _normalize_actions(action: Any, max_action_steps: int | None, action_dim: int) -> np.ndarray:
    if isinstance(action, dict):
        if "actions" in action:
            action = action["actions"]
        elif "action" in action:
            action = action["action"]
        else:
            action = next(iter(action.values()))

    action = np.asarray(action, dtype=np.float32)
    if action.ndim == 1:
        actions = action[None, :]
    elif action.ndim == 2:
        actions = action
    elif action.ndim == 3:
        actions = action[0]
    else:
        raise ValueError(f"不支持的 action 输出维度：shape={action.shape}")

    if actions.shape[-1] != action_dim:
        raise ValueError(f"OpenPI 双臂仿真需要 {action_dim} 维 delta action，实际 shape={actions.shape}")
    if max_action_steps is not None and max_action_steps > 0:
        actions = actions[:max_action_steps]
    return np.ascontiguousarray(actions, dtype=np.float32)


def _parse_image_for_openpi(image: Any) -> np.ndarray:
    return _as_uint8_hwc(image)


def make_bimanual_transform_factory(transforms_module, openpi_model_module, wrapped_factory, action_dim: int):
    @dataclasses.dataclass(frozen=True)
    class BimanualOpenPIInputs(transforms_module.DataTransformFn):
        model_type: Any

        def __call__(self, data: dict[str, Any]) -> dict[str, Any]:
            base_image = _parse_image_for_openpi(
                _get_first(data, ("observation.images.image", "observation/image"))
            )
            left_wrist_image = _parse_image_for_openpi(
                _get_first(data, ("observation.images.left_wrist_image", "observation/left_wrist_image"))
            )
            right_wrist_image = _parse_image_for_openpi(
                _get_first(data, ("observation.images.right_wrist_image", "observation/right_wrist_image"))
            )
            state = np.asarray(
                _get_first(data, ("observation.state", "observation/state", "state")),
                dtype=np.float32,
            )

            inputs = {
                "state": state,
                "image": {
                    "base_0_rgb": base_image,
                    "left_wrist_0_rgb": left_wrist_image,
                    "right_wrist_0_rgb": right_wrist_image,
                },
                "image_mask": {
                    "base_0_rgb": np.True_,
                    "left_wrist_0_rgb": np.True_,
                    "right_wrist_0_rgb": np.True_,
                },
            }
            if "actions" in data:
                inputs["actions"] = np.asarray(data["actions"], dtype=np.float32)
            if "prompt" in data:
                inputs["prompt"] = data["prompt"]
            elif "task" in data:
                inputs["prompt"] = data["task"]
            return inputs

    @dataclasses.dataclass(frozen=True)
    class BimanualOpenPIOutputs(transforms_module.DataTransformFn):
        action_dim: int = 14

        def __call__(self, data: dict[str, Any]) -> dict[str, Any]:
            return {"actions": np.asarray(data["actions"][:, : self.action_dim], dtype=np.float32)}

    @dataclasses.dataclass(frozen=True)
    class BimanualDataConfigFactory:
        wrapped: Any
        action_dim: int = 14

        def create(self, assets_dirs: pathlib.Path, model_config: Any) -> Any:
            data_config = self.wrapped.create(assets_dirs, model_config)
            return dataclasses.replace(
                data_config,
                repack_transforms=transforms_module.Group(),
                data_transforms=transforms_module.Group(
                    inputs=[BimanualOpenPIInputs(model_type=model_config.model_type)],
                    outputs=[BimanualOpenPIOutputs(action_dim=self.action_dim)],
                ),
            )

    return BimanualDataConfigFactory(wrapped=wrapped_factory, action_dim=action_dim)


class WristCameraVisualizer:
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
        self.window_name = "OpenPI server wrist cameras"
        self.output_dir = pathlib.Path(output_dir).expanduser() if output_dir else None
        self._cv2 = None

        if not self.enabled:
            return

        try:
            import cv2
        except ImportError as exc:
            raise ImportError(
                "启用腕部图像可视化需要 cv2。请安装 opencv-python，"
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
        for text, pos in (
            (f"left_wrist #{self.counter}", (8, 24)),
            ("right_wrist", (left_w + 8, 24)),
            (stamp, (8, max(48, panel_bgr.shape[0] - 12))),
        ):
            self._cv2.putText(
                panel_bgr,
                text,
                pos,
                self._cv2.FONT_HERSHEY_SIMPLEX,
                0.65 if "wrist" in text else 0.5,
                (255, 255, 255),
                2 if "wrist" in text else 1,
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


def build_adapter_class(base_policy_class):
    class OpenPIBimanualSimWebsocketPolicy(base_policy_class):
        def __init__(
            self,
            policy,
            *,
            image_key_style: str,
            default_task: str,
            max_action_steps: int | None,
            action_dim: int,
            wrist_visualizer: WristCameraVisualizer | None,
        ) -> None:
            self.policy = policy
            self.image_key_style = image_key_style
            self.image_keys = IMAGE_KEYS_BY_STYLE[image_key_style]
            self.default_task = default_task
            self.max_action_steps = max_action_steps
            self.action_dim = action_dim
            self.wrist_visualizer = wrist_visualizer
            self.metadata = {
                "adapter": "openpi_bimanual_sim",
                "image_key_style": image_key_style,
                "state_key": "observation.state",
                "image_keys": self.image_keys,
                "action_key": "actions",
                "action_shape": ["T", action_dim],
            }

        def reset(self) -> None:
            if hasattr(self.policy, "reset"):
                self.policy.reset()

        def infer(self, obs: dict[str, Any]) -> dict[str, Any]:
            if not isinstance(obs, dict):
                raise TypeError(f"websocket obs 必须是 dict，实际 {type(obs)}")

            if obs.get("__reset__", False):
                self.reset()
                return {"actions": np.zeros((0, self.action_dim), dtype=np.float32)}

            missing = [key for key in self.image_keys.values() if key not in obs]
            if "observation.state" not in obs:
                missing.append("observation.state")
            if missing:
                raise KeyError(
                    "websocket obs 缺少 OpenPI 双臂仿真输入 key："
                    f"{missing}。当前收到 keys={list(obs.keys())}"
                )

            task = _get_task(obs, self.default_task)
            if self.wrist_visualizer is not None:
                self.wrist_visualizer.update(obs, self.image_keys, task)

            policy_obs = {
                "observation.images.image": _as_uint8_hwc(obs[self.image_keys["main"]]),
                "observation.images.left_wrist_image": _as_uint8_hwc(obs[self.image_keys["left_wrist"]]),
                "observation.images.right_wrist_image": _as_uint8_hwc(obs[self.image_keys["right_wrist"]]),
                "observation.state": _state_array(obs["observation.state"]),
                "prompt": task,
            }

            result = self.policy.infer(policy_obs)
            actions = _normalize_actions(result, self.max_action_steps, self.action_dim)
            if isinstance(result, dict) and "policy_timing" in result:
                return {"actions": actions, "policy_timing": result["policy_timing"]}
            return {"actions": actions}

    return OpenPIBimanualSimWebsocketPolicy


def _load_openpi_policy(
    *,
    config_name: str,
    checkpoint_dir: str,
    default_prompt: str | None,
    pytorch_device: str | None,
    use_bimanual_transforms: bool,
    action_dim: int,
    modules: tuple[Any, Any, Any, Any, Any, Any],
):
    transforms_module, openpi_model_module, policy_config, _server, training_config, _base_policy = modules

    train_config = training_config.get_config(config_name)
    if use_bimanual_transforms:
        train_config = dataclasses.replace(
            train_config,
            data=make_bimanual_transform_factory(
                transforms_module,
                openpi_model_module,
                train_config.data,
                action_dim=action_dim,
            ),
        )

    policy = policy_config.create_trained_policy(
        train_config,
        checkpoint_dir,
        default_prompt=default_prompt,
        pytorch_device=pytorch_device,
    )
    logging.info("Loaded OpenPI policy config=%s checkpoint=%s", config_name, checkpoint_dir)
    logging.info("Bimanual inference transforms: %s", use_bimanual_transforms)
    try:
        import openpi

        logging.info("Using openpi from: %s", getattr(openpi, "__file__", "unknown"))
    except Exception:
        pass
    return policy


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
    parser = argparse.ArgumentParser(description="Serve OpenPI/pi0.5 bimanual simulation policy over websocket.")
    parser.add_argument("--checkpoint-dir", required=True, help="OpenPI checkpoint 目录，例如 .../run/30000。")
    parser.add_argument("--config-name", required=True, help="训练时使用的 OpenPI config 名，例如 pi05_xxx。")
    parser.add_argument("--host", default="0.0.0.0", help="websocket server 监听地址。")
    parser.add_argument("--port", type=int, default=8000, help="websocket server 监听端口。")
    parser.add_argument("--pytorch-device", default=None, help="PyTorch checkpoint 用的设备，例如 cuda:0；默认自动选择。")
    parser.add_argument("--image-key-style", choices=sorted(IMAGE_KEYS_BY_STYLE), default="generic")
    parser.add_argument("--action-dim", type=int, default=14, help="模型有效 action 维度，双臂 EEF delta 默认 14。")
    parser.add_argument("--max-action-steps", type=int, default=8, help="返回 action chunk 的前多少步；<=0 表示不截断。")
    parser.add_argument("--default-prompt", default="", help="请求里没有 task/prompt 时使用的默认语言指令。")
    parser.add_argument("--openpi-repo-root", default=None, help="OpenPI 源码根目录；默认读取 OPENPI_ROOT 或常用路径。")
    parser.add_argument(
        "--no-bimanual-transforms",
        dest="use_bimanual_transforms",
        action="store_false",
        help="不替换 config 的输入/输出 transform，完全使用 config 原始推理 transform。",
    )
    parser.set_defaults(use_bimanual_transforms=True)
    parser.add_argument("--no-warmup", action="store_true", help="启动时跳过一次 dummy forward。")
    parser.add_argument("--image-size", type=int, default=256, help="warmup dummy 图像尺寸。")
    parser.add_argument(
        "--visualize-wrist-cameras",
        action="store_true",
        default=_env_flag("OPENPI_VISUALIZE_WRIST_CAMERAS"),
        help="保存/显示 server 收到的左右腕部图像；也可设置 OPENPI_VISUALIZE_WRIST_CAMERAS=1。",
    )
    parser.add_argument(
        "--wrist-visualize-dir",
        default=os.environ.get("OPENPI_WRIST_VIS_DIR", DEFAULT_WRIST_VIS_DIR),
        help="保存左右腕部拼图的目录；设置为空字符串可只显示不保存。",
    )
    parser.add_argument(
        "--wrist-visualize-every-n",
        type=int,
        default=_env_int("OPENPI_WRIST_VIS_EVERY_N", 1),
        help="每隔多少次 infer 保存/显示一帧；默认 1。",
    )
    parser.add_argument(
        "--show-wrist-cameras",
        action="store_true",
        default=_env_flag("OPENPI_SHOW_WRIST_CAMERAS"),
        help="同时用 cv2.imshow 弹窗显示；远程/headless 环境通常只建议保存图片。",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s: %(message)s", force=True)

    _setup_import_paths(args.openpi_repo_root)
    modules = _import_openpi_modules()
    _transforms, _openpi_model, _policy_config, websocket_policy_server, _training_config, base_policy = modules

    max_action_steps = args.max_action_steps if args.max_action_steps > 0 else None
    policy = _load_openpi_policy(
        config_name=args.config_name,
        checkpoint_dir=args.checkpoint_dir,
        default_prompt=args.default_prompt,
        pytorch_device=args.pytorch_device,
        use_bimanual_transforms=args.use_bimanual_transforms,
        action_dim=args.action_dim,
        modules=modules,
    )

    adapter_cls = build_adapter_class(base_policy.BasePolicy)
    adapter = adapter_cls(
        policy,
        image_key_style=args.image_key_style,
        default_task=args.default_prompt,
        max_action_steps=max_action_steps,
        action_dim=args.action_dim,
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

    logging.info("Serving OpenPI bimanual sim websocket policy")
    logging.info("Host=%s local_ip=%s port=%d", args.host, local_ip, args.port)
    logging.info("Expected image keys=%s", adapter.image_keys)
    logging.info("Expected state key=observation.state shape=(16,), output actions shape=(T,%d)", args.action_dim)
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
