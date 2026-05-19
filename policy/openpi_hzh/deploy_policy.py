"""RoboTwin client adapter for an OpenPI/pi0.5 bimanual simulation policy.

The training converter
`/home/tianji/hzh/study/openpi/convert_hdf5_to_new_format_simulation_bimanual_openpi.py`
uses the same online state/action convention as the SmolVLA adapter:

- observation.state: 16D bimanual EEF state.
- actions: 14D bimanual EEF delta action.

This file therefore reuses the carefully tested RoboTwin-side observation encoding
and delta-to-absolute-EEF conversion helpers from `policy.smolvla_hzh`.
"""

from __future__ import annotations

import numpy as np

from policy.smolvla_hzh.deploy_policy import (
    add_openpi_client_paths,
    as_bool,
    delta_actions_to_env_ee_actions,
    encode_obs,
)


class OpenPIWebsocketClient:
    """Thin RoboTwin-side OpenPI websocket client.

    The heavy OpenPI/pi0.5 checkpoint stays in the policy server process started by
    `serve_policy_openpi_bimanual_sim.py`. RoboTwin only sends encoded observations
    and receives 14D delta action chunks.
    """

    def __init__(self, usr_args):
        add_openpi_client_paths(usr_args)
        from openpi_client import websocket_client_policy

        self.host = usr_args.get("remote_host") or usr_args.get("host") or "localhost"
        self.port = int(usr_args.get("remote_port") or usr_args.get("websocket_port") or 8000)
        self.action_type = usr_args.get("action_type", "ee")
        self.openpi_step = int(
            usr_args.get("openpi_step", usr_args.get("pi0_step", usr_args.get("n_action_steps", 8)))
        )
        self.image_key_style = usr_args.get("image_key_style", "generic")
        self.arm_order = usr_args.get("arm_order", "left_right")

        self.policy = websocket_client_policy.WebsocketClientPolicy(host=self.host, port=self.port)
        self.metadata = {}
        if hasattr(self.policy, "get_server_metadata"):
            self.metadata = self.policy.get_server_metadata()
            adapter = self.metadata.get("adapter")
            if adapter and adapter != "openpi_bimanual_sim":
                print(f"[OpenPI websocket] Warning: connected server adapter={adapter!r}")
        print(f"[OpenPI websocket] Connected to ws://{self.host}:{self.port}")

    def get_action(self, obs):
        result = self.policy.infer(obs)
        if "error" in result:
            raise RuntimeError(f"OpenPI websocket server error:\n{result['error']}")
        if "actions" not in result:
            raise KeyError(f"OpenPI websocket response missing 'actions', got keys={list(result.keys())}")

        actions = np.asarray(result["actions"], dtype=np.float32)
        if actions.ndim == 1:
            actions = actions[None, :]
        elif actions.ndim == 3:
            actions = actions[0]
        if actions.shape[-1] != 14:
            raise ValueError(f"Expected websocket actions shape [T,14], got {actions.shape}")
        return actions[: self.openpi_step]

    def reset(self):
        try:
            result = self.policy.infer({"__reset__": True})
            if "error" in result:
                raise RuntimeError(result["error"])
        except Exception as exc:
            print(f"[OpenPI websocket] Warning: reset request failed: {exc}")

    def reset_model(self):
        self.reset()


def get_model(usr_args):
    if not (as_bool(usr_args.get("use_websocket")) or usr_args.get("inference_mode") == "websocket"):
        raise ValueError(
            "openpi_hzh currently expects websocket inference. Start "
            "policy/openpi_hzh/serve_policy_openpi_bimanual_sim.py in the OpenPI env "
            "and keep use_websocket=true."
        )
    return OpenPIWebsocketClient(usr_args)


def eval(TASK_ENV, model, observation):
    image_key_style = getattr(model, "image_key_style", "generic")
    obs = encode_obs(observation, image_key_style=image_key_style)
    obs["task"] = TASK_ENV.get_instruction()

    if hasattr(model, "call"):
        delta_actions = model.call(func_name="get_action", obs=obs)
        arm_order = "left_right"
        action_type = "ee"
    else:
        delta_actions = model.get_action(obs)
        arm_order = model.arm_order
        action_type = model.action_type

    env_actions = delta_actions_to_env_ee_actions(
        delta_actions,
        obs["observation.state"],
        arm_order=arm_order,
    )

    for action in env_actions:
        TASK_ENV.take_action(action, action_type=action_type)
        if TASK_ENV.eval_success:
            break
        observation = TASK_ENV.get_obs()


def reset_model(model):
    if hasattr(model, "call"):
        model.call(func_name="reset_model")
    else:
        model.reset()
