import os
import pathlib
import sys

import cv2
import numpy as np


# 这些 key 必须和训练数据转换脚本保持一致。
# 参考：
# /home/tianji/hzh/study/openpi/convert_hdf5_to_new_format_simulation_bimanual.py
#
# generic 是该转换脚本的默认命名：
#   observation.images.image
#   observation.images.left_wrist_image
#   observation.images.right_wrist_image
#
# aloha 是可选兼容命名：
#   observation.images.cam_high
#   observation.images.cam_left_wrist
#   observation.images.cam_right_wrist
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


def as_bool(value):
    """把 yml/命令行里常见的布尔写法统一成 bool。"""
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    return str(value).strip().lower() in ("1", "true", "yes", "y", "on")


def add_openpi_client_paths(usr_args):
    """让 RoboTwin 环境能导入 OpenPI websocket client。

    client 侧只需要 openpi_client + websockets/msgpack 这些轻量依赖，不需要
    lerobot/torch。优先使用用户在 yml 里指定的路径；如果没有，就尝试几个
    本机常见位置和 RoboTwin 仓库里自带的 pi0 openpi-client。
    """
    repo_root = pathlib.Path(__file__).resolve().parents[2]
    candidates = []

    explicit_client_path = usr_args.get("openpi_client_path")
    if explicit_client_path:
        candidates.append(pathlib.Path(explicit_client_path).expanduser())

    openpi_root = usr_args.get("openpi_repo_root") or os.environ.get("OPENPI_ROOT")
    if openpi_root:
        openpi_root = pathlib.Path(openpi_root).expanduser()
        candidates.extend(
            [
                openpi_root / "packages" / "openpi-client" / "src",
                openpi_root / "src",
                openpi_root,
            ]
        )

    candidates.extend(
        [
            pathlib.Path("/home/tianji/hzh/study/openpi/packages/openpi-client/src"),
            repo_root / "policy" / "pi0" / "packages" / "openpi-client" / "src",
            repo_root / "policy" / "pi05" / "packages" / "openpi-client" / "src",
        ]
    )

    for path in candidates:
        if path.exists() and str(path) not in sys.path:
            sys.path.insert(0, str(path))


def ensure_uint8_rgb(image):
    """把 RoboTwin 相机图像整理成 uint8 RGB 的 HWC 格式。

    RoboTwin 在线环境里通常已经给出 uint8 RGB 图像，但这里仍然处理几种
    可能情况：
    - 灰度图：复制成 3 通道；
    - RGBA：丢弃 alpha；
    - float 图像：如果范围是 0~1，则放大到 0~255。
    """
    image = np.asarray(image)
    if image.ndim == 2:
        image = np.repeat(image[:, :, None], 3, axis=2)
    elif image.ndim == 3 and image.shape[2] == 4:
        image = image[:, :, :3]
    if image.dtype != np.uint8:
        if np.nanmax(image) <= 1.0:
            image = image * 255.0
        image = np.clip(image, 0, 255).astype(np.uint8)
    return image


def resize_image(image, target_size=(256, 256)):
    """把图像 resize 到训练时使用的 256x256。

    转换脚本写入 LeRobot 数据集时固定 target_size=(256, 256)，部署时
    必须保持同样的输入分辨率，否则和训练分布不一致。
    """
    image = ensure_uint8_rgb(image)
    if image.shape[:2] == target_size:
        return image
    width, height = target_size[1], target_size[0]
    return cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)


def normalize_quat_xyzw(quat):
    """归一化四元数。

    本文件里的辅助函数使用 [qx, qy, qz, qw] 顺序，和转换脚本里的
    scipy Rotation.from_quat(...) 约定一致。RoboTwin endpose 的后 4 维
    在你的转换脚本中也是按这个顺序解释的，所以部署端也沿用同一约定。
    """
    quat = np.asarray(quat, dtype=np.float64)
    norm = np.linalg.norm(quat)
    if norm < 1e-12 or not np.isfinite(norm):
        raise ValueError(f"Invalid quaternion: {quat}")
    return quat / norm


def quat_xyzw_to_axisangle(quat):
    """把四元数转换成轴角 rotvec。

    训练数据里的 observation.state 不是原始 quaternion endpose，而是：
      [x, y, z, rx, ry, rz, gripper, -gripper]
    其中 [rx, ry, rz] 是四元数转换出来的 axis-angle。
    """
    quat = normalize_quat_xyzw(quat)
    xyz = quat[:3]
    w = quat[3]
    xyz_norm = np.linalg.norm(xyz)
    if xyz_norm < 1e-12:
        return np.zeros(3, dtype=np.float32)

    angle = 2.0 * np.arctan2(xyz_norm, w)
    if angle > np.pi:
        angle -= 2.0 * np.pi
    axis = xyz / xyz_norm
    return (axis * angle).astype(np.float32)


def axisangle_to_quat_xyzw(axisangle):
    """把轴角 rotvec 转回四元数 [qx, qy, qz, qw]。

    模型输出的是 delta axis-angle，RoboTwin 的 ee 控制接口需要 absolute
    pose quaternion，所以执行动作前需要做一次反变换。
    """
    axisangle = np.asarray(axisangle, dtype=np.float64)
    angle = np.linalg.norm(axisangle)
    if angle < 1e-12:
        return np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
    axis = axisangle / angle
    half = angle * 0.5
    quat = np.concatenate([axis * np.sin(half), [np.cos(half)]])
    return normalize_quat_xyzw(quat).astype(np.float32)


def quat_xyzw_multiply(q1, q2):
    """四元数乘法，返回 q1 * q2。

    转换脚本中相对旋转定义为：
      relative_rotation = R_next * R_prev.inv()
    因此部署时从当前姿态和模型预测的 delta 恢复下一帧姿态时，需要：
      R_next = R_delta * R_current
    """
    x1, y1, z1, w1 = normalize_quat_xyzw(q1)
    x2, y2, z2, w2 = normalize_quat_xyzw(q2)
    return np.array(
        [
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        ],
        dtype=np.float32,
    )


def apply_delta_axisangle(axisangle, delta_axisangle):
    """在当前 axis-angle 姿态上应用模型预测的 delta axis-angle。"""
    current_quat = axisangle_to_quat_xyzw(axisangle)
    delta_quat = axisangle_to_quat_xyzw(delta_axisangle)
    next_quat = quat_xyzw_multiply(delta_quat, current_quat)
    return quat_xyzw_to_axisangle(next_quat)


def build_arm_state(endpose, gripper):
    """构建单臂 8 维 observation.state。

    输入：
      endpose: RoboTwin 当前末端位姿 [x, y, z, qx, qy, qz, qw]
      gripper: 当前夹爪值

    输出和转换脚本 build_arm_states(...) 保持一致：
      [x, y, z, rx, ry, rz, gripper, -gripper]

    双臂 state 会由 left_state + right_state 拼成 16 维。
    """
    endpose = np.asarray(endpose, dtype=np.float64)
    axisangle = quat_xyzw_to_axisangle(endpose[3:7])
    return np.concatenate(
        [
            endpose[:3].astype(np.float32),
            axisangle,
            np.asarray([gripper, -gripper], dtype=np.float32),
        ]
    )


def split_bimanual(values, arm_order):
    """按训练时的 arm_order 切分双臂向量。

    values 可以是：
    - 16 维 state：[left_state8, right_state8]
    - 14 维 action：[left_action7, right_action7]

    如果训练转换时用了 --arm_order right_left，这里也必须同步设置成
    right_left，否则左右臂动作会互换。
    """
    values = np.asarray(values, dtype=np.float32)
    half = values.shape[-1] // 2
    first, second = values[..., :half], values[..., half:]
    if arm_order == "left_right":
        return first, second
    if arm_order == "right_left":
        return second, first
    raise ValueError(f"Unknown arm_order: {arm_order}")


def concat_env_ee_action(left_pose7, left_gripper, right_pose7, right_gripper):
    """拼成 RoboTwin take_action(..., action_type='ee') 需要的 16 维动作。

    RoboTwin ee 控制格式：
      [left_pose7, left_gripper, right_pose7, right_gripper]
    其中 pose7 = [x, y, z, qx, qy, qz, qw]。
    """
    return np.concatenate(
        [
            np.asarray(left_pose7, dtype=np.float32),
            np.asarray([left_gripper], dtype=np.float32),
            np.asarray(right_pose7, dtype=np.float32),
            np.asarray([right_gripper], dtype=np.float32),
        ]
    )


def apply_arm_delta(current_state8, action7):
    """把单臂模型 delta action 转成下一帧 absolute EEF pose。

    训练时 action 的定义来自转换脚本 compute_arm_actions_from_states(...)：
      action7 = [delta_xyz(3), delta_axisangle(3), next_gripper(1)]

    在线执行时不能直接把 action7 传给 RoboTwin，因为 RoboTwin 的 ee 控制
    需要 absolute pose。因此这里用当前 state8 恢复下一帧 pose7。
    """
    next_xyz = current_state8[:3] + action7[:3]
    next_axisangle = apply_delta_axisangle(current_state8[3:6], action7[3:6])
    next_gripper = float(action7[6])
    next_state8 = np.concatenate(
        [next_xyz, next_axisangle, np.asarray([next_gripper, -next_gripper], dtype=np.float32)]
    ).astype(np.float32)
    # The conversion script interprets/stores quaternion arrays in the same 4-value
    # order as the source endpose. Keep that array order when sending back to RoboTwin.
    next_quat = axisangle_to_quat_xyzw(next_axisangle)
    next_pose7 = np.concatenate([next_xyz, next_quat]).astype(np.float32)
    return next_state8, next_pose7, next_gripper


def delta_actions_to_env_ee_actions(actions, current_state, arm_order):
    """把 SmolVLA 输出的双臂 delta actions 转成 RoboTwin ee actions。

    SmolVLA 输出：
      [left_delta7, right_delta7]，总计 14 维。

    RoboTwin 执行：
      [left_abs_pose7, left_gripper, right_abs_pose7, right_gripper]，
      总计 16 维。

    如果模型一次输出一个 action chunk，这里会从 current_state 开始逐个
    累积 delta，得到每一步要发送给环境的 absolute EEF 动作。
    """
    actions = np.asarray(actions, dtype=np.float32)
    if actions.ndim == 1:
        actions = actions[None, :]
    if actions.shape[-1] != 14:
        raise ValueError(f"Expected SmolVLA delta action dim 14, got {actions.shape}")

    left_state, right_state = split_bimanual(current_state, arm_order)
    env_actions = []
    for action in actions:
        left_action, right_action = split_bimanual(action, arm_order)
        left_state, left_pose7, left_gripper = apply_arm_delta(left_state, left_action)
        right_state, right_pose7, right_gripper = apply_arm_delta(right_state, right_action)
        env_actions.append(concat_env_ee_action(left_pose7, left_gripper, right_pose7, right_gripper))
    return np.asarray(env_actions, dtype=np.float32)


def encode_obs(observation, image_key_style="generic"):  # Post-Process Observation
    """把 RoboTwin 在线 observation 转成 SmolVLA 训练时的输入格式。

    RoboTwin 原始 observation 中使用：
      observation/head_camera/rgb
      observation/left_camera/rgb
      observation/right_camera/rgb
      endpose/left_endpose
      endpose/right_endpose

    SmolVLA/LeRobot 训练数据中使用：
      observation.images.image
      observation.images.left_wrist_image
      observation.images.right_wrist_image
      observation.state

    因此这里负责做 key 映射、图像 resize，以及 EEF state 编码。
    """
    image_keys = IMAGE_KEYS_BY_STYLE[image_key_style]
    left_state = build_arm_state(
        observation["endpose"]["left_endpose"],
        observation["endpose"]["left_gripper"],
    )
    right_state = build_arm_state(
        observation["endpose"]["right_endpose"],
        observation["endpose"]["right_gripper"],
    )

    return {
        image_keys["main"]: resize_image(observation["observation"]["head_camera"]["rgb"]),
        image_keys["left_wrist"]: resize_image(observation["observation"]["left_camera"]["rgb"]),
        image_keys["right_wrist"]: resize_image(observation["observation"]["right_camera"]["rgb"]),
        "observation.state": np.concatenate([left_state, right_state]).astype(np.float32),
    }


class SmolVLAModel:
    """SmolVLA policy 的最薄封装。

    注意：这个类只会在 policy 环境里实例化。
    - direct 模式 eval.sh：它会在 RoboTwin 环境里实例化，因此两个环境需要共用依赖。
    - 分离模式 eval_double_env.sh：它会在 policy_model_server.py 所激活的
      policy_conda_env 中实例化，RoboTwin 环境只通过 socket 调用它。
    """

    def __init__(self, usr_args):
        # 这些 import 放在 __init__ 里面，是为了让分离模式下只有 policy server
        # 需要安装 lerobot；RoboTwin client 侧不会在 import 本文件时立刻加载它们。
        import torch
        from lerobot.configs.policies import PreTrainedConfig
        from lerobot.policies.factory import get_policy_class

        # ckpt_setting 可以直接传 checkpoint 路径；也可以传一个相对名字，
        # 如果 policy/smolvla-hzh/checkpoints/<name> 存在，就会自动使用它。
        policy_path = (
            usr_args.get("policy_path")
            or usr_args.get("checkpoint_path")
            or usr_args.get("model_path")
            or usr_args.get("ckpt_setting")
        )
        if policy_path in (None, "", "null"):
            raise ValueError("Please set policy_path/checkpoint_path/model_path/ckpt_setting for SmolVLA.")

        local_policy_path = os.path.join("policy", "smolvla-hzh", "checkpoints", str(policy_path))
        if not os.path.exists(str(policy_path)) and os.path.exists(local_policy_path):
            policy_path = local_policy_path

        self.torch = torch
        self.device = usr_args.get("device") or ("cuda" if self.torch.cuda.is_available() else "cpu")
        # 模型输出会在 eval() 中转成 RoboTwin absolute ee action，所以这里
        # 默认用 ee 控制。
        self.action_type = usr_args.get("action_type", "ee")
        # 每次从 action chunk 中执行多少步。这个值应该和训练/推理时的
        # action horizon 使用习惯匹配。
        self.smolvla_step = int(usr_args.get("smolvla_step", usr_args.get("n_action_steps", 8)))
        # image_key_style 必须和转换数据时的 --image_key_style 一致。
        self.image_key_style = usr_args.get("image_key_style", "generic")
        # arm_order 必须和转换数据时的 --arm_order 一致。
        self.arm_order = usr_args.get("arm_order", "left_right")
        self.instruction = None

        # 按 checkpoint 中的 config.type 动态找到 policy 类，避免硬编码
        # SmolVLAPolicy 路径导致不同 lerobot 版本不兼容。
        cfg = PreTrainedConfig.from_pretrained(str(policy_path))
        cfg.device = self.device
        policy_cls = get_policy_class(cfg.type)
        self.policy = policy_cls.from_pretrained(str(policy_path), config=cfg)
        self.policy.eval()

    def set_language(self, instruction):
        """设置当前 episode 的语言指令。"""
        self.instruction = "" if instruction is None else str(instruction)

    def _to_image_tensor(self, image):
        """把 HWC uint8/float 图像转成模型 batch 需要的 [1, C, H, W] tensor。"""
        image = np.asarray(image)
        if image.dtype == np.uint8:
            image = image.astype(np.float32) / 255.0
        else:
            image = image.astype(np.float32)
            if image.max(initial=0.0) > 1.0:
                image = image / 255.0
        image = np.transpose(image, (2, 0, 1))
        return self.torch.from_numpy(np.ascontiguousarray(image)).unsqueeze(0).to(self.device)

    def build_batch(self, obs):
        """把 encode_obs 的 numpy 字典组装成 policy.select_action 的 batch。"""
        batch = {
            "observation.state": self.torch.from_numpy(obs["observation.state"][None]).to(self.device),
            "task": [self.instruction or ""],
        }
        image_keys = IMAGE_KEYS_BY_STYLE[self.image_key_style]
        for key in image_keys.values():
            batch[key] = self._to_image_tensor(obs[key])
        return batch

    def get_action(self, obs):
        """运行 SmolVLA 推理，返回模型原始 14 维 delta action chunk。

        注意这里还没有转成 RoboTwin 可执行的 ee action；转换发生在 eval()
        里的 delta_actions_to_env_ee_actions(...)。
        """
        if "task" in obs:
            self.set_language(obs["task"])
        batch = self.build_batch(obs)
        with self.torch.inference_mode():
            actions = self.policy.select_action(batch)

        if isinstance(actions, dict):
            actions = actions.get("action", next(iter(actions.values())))
        if isinstance(actions, self.torch.Tensor):
            actions = actions.detach().cpu().numpy()

        actions = np.asarray(actions, dtype=np.float32)
        if actions.ndim == 1:
            actions = actions[None, :]
        elif actions.ndim == 3:
            actions = actions[0]
        return actions[: self.smolvla_step]

    def reset(self):
        """清空 episode 级缓存。"""
        self.instruction = None
        if hasattr(self.policy, "reset"):
            self.policy.reset()

    def reset_model(self):
        """给 policy_model_server.py 远程调用用的 reset 接口。"""
        self.reset()


class SmolVLAWebsocketClient:
    """RoboTwin 侧的 OpenPI websocket client。

    这个类只运行在 RoboTwin 环境里，不加载 SmolVLA checkpoint，也不 import
    lerobot。真正的模型推理发生在
    serve_policy_smolvla_bimanual_sim.py 启动的 policy server 里。
    """

    def __init__(self, usr_args):
        add_openpi_client_paths(usr_args)
        from openpi_client import websocket_client_policy

        self.host = usr_args.get("remote_host") or usr_args.get("host") or "localhost"
        self.port = int(usr_args.get("remote_port") or usr_args.get("websocket_port") or 8000)
        self.action_type = usr_args.get("action_type", "ee")
        self.smolvla_step = int(usr_args.get("smolvla_step", usr_args.get("n_action_steps", 8)))
        self.image_key_style = usr_args.get("image_key_style", "generic")
        self.arm_order = usr_args.get("arm_order", "left_right")

        self.policy = websocket_client_policy.WebsocketClientPolicy(host=self.host, port=self.port)
        self.metadata = {}
        if hasattr(self.policy, "get_server_metadata"):
            self.metadata = self.policy.get_server_metadata()
            adapter = self.metadata.get("adapter")
            if adapter and adapter != "smolvla_bimanual_sim":
                print(f"[SmolVLA websocket] Warning: connected server adapter={adapter!r}")
        print(f"[SmolVLA websocket] Connected to ws://{self.host}:{self.port}")

    def get_action(self, obs):
        """通过 websocket 请求 server 推理，返回 [T, 14] delta action。"""
        result = self.policy.infer(obs)
        if "error" in result:
            raise RuntimeError(f"SmolVLA websocket server error:\n{result['error']}")
        if "actions" not in result:
            raise KeyError(f"SmolVLA websocket response missing 'actions', got keys={list(result.keys())}")

        actions = np.asarray(result["actions"], dtype=np.float32)
        if actions.ndim == 1:
            actions = actions[None, :]
        elif actions.ndim == 3:
            actions = actions[0]
        if actions.shape[-1] != 14:
            raise ValueError(f"Expected websocket actions shape [T,14], got {actions.shape}")
        return actions[: self.smolvla_step]

    def reset(self):
        """通知 server 清理 episode 级缓存。"""
        try:
            result = self.policy.infer({"__reset__": True})
            if "error" in result:
                raise RuntimeError(result["error"])
        except Exception as exc:
            print(f"[SmolVLA websocket] Warning: reset request failed: {exc}")

    def reset_model(self):
        self.reset()


def get_model(usr_args):  # from deploy_policy.yml and eval.sh (overrides)
    """RoboTwin 官方入口：创建并返回 policy model。"""
    if as_bool(usr_args.get("use_websocket")) or usr_args.get("inference_mode") == "websocket":
        return SmolVLAWebsocketClient(usr_args)
    return SmolVLAModel(usr_args)  # return your policy model


def eval(TASK_ENV, model, observation):
    """
    Keep the RoboTwin policy structure, but match the SmolVLA dataset conversion:
    observation.state is bimanual EEF state, model action is bimanual EEF delta.
    """
    # direct 模式下 model 是 SmolVLAModel，可以读取 model.image_key_style；
    # 分离模式下 model 是 ModelClient，没有这些属性，因此使用 yml 默认值 generic。
    image_key_style = getattr(model, "image_key_style", "generic")
    obs = encode_obs(observation, image_key_style=image_key_style)
    obs["task"] = TASK_ENV.get_instruction()

    # 分离模式：model 是 socket client，通过 model.call(...) 请求 policy server。
    # direct 模式：model 是本地 SmolVLAModel，直接调用 get_action(...)。
    if hasattr(model, "call"):
        delta_actions = model.call(func_name="get_action", obs=obs)
        arm_order = "left_right"
        action_type = "ee"
    else:
        delta_actions = model.get_action(obs)
        arm_order = model.arm_order
        action_type = model.action_type

    # 模型输出是训练格式中的 14 维 delta action；RoboTwin 执行需要 16 维
    # absolute ee action，所以这里做格式转换。
    env_actions = delta_actions_to_env_ee_actions(
        delta_actions,
        obs["observation.state"],
        arm_order=arm_order,
    )

    # 保持 RoboTwin 官方模板逻辑：逐步执行 action chunk。
    for action in env_actions:
        TASK_ENV.take_action(action, action_type=action_type)
        if TASK_ENV.eval_success:
            break
        observation = TASK_ENV.get_obs()


def reset_model(model):
    # Clean the model cache at the beginning of every evaluation episode.
    # eval_policy_client.py 会直接 model.call("reset_model")，但这里也兼容
    # direct eval_policy.py 调用 reset_model(model) 的情况。
    if hasattr(model, "call"):
        model.call(func_name="reset_model")
    else:
        model.reset()
