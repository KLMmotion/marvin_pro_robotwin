import os
import sys
import traceback
from argparse import ArgumentParser

sys.path.append("./")

import torch.multiprocessing as mp
import yaml

from envs import CONFIGS_PATH
from script.collect_data import class_decorator, get_embodiment_config


def build_args(task_name, task_config, seed):
    config_path = f"./task_config/{task_config}.yml"
    with open(config_path, "r", encoding="utf-8") as f:
        args = yaml.load(f.read(), Loader=yaml.FullLoader)

    args["task_name"] = task_name
    embodiment_config_path = os.path.join(CONFIGS_PATH, "_embodiment_config.yml")
    with open(embodiment_config_path, "r", encoding="utf-8") as f:
        embodiment_types = yaml.load(f.read(), Loader=yaml.FullLoader)

    embodiment_type = args["embodiment"]
    args["left_robot_file"] = embodiment_types[embodiment_type[0]]["file_path"]
    args["right_robot_file"] = embodiment_types[embodiment_type[0]]["file_path"]
    args["dual_arm_embodied"] = True
    args["left_embodiment_config"] = get_embodiment_config(args["left_robot_file"])
    args["right_embodiment_config"] = get_embodiment_config(args["right_robot_file"])
    args["embodiment_name"] = str(embodiment_type[0])
    args["task_config"] = task_config
    args["save_path"] = os.path.join(args["save_path"], task_name, task_config)
    args["need_plan"] = True
    args["render_freq"] = 0
    args["render_sleep"] = 0
    args["save_data"] = False
    args["now_ep_num"] = 0
    args["seed"] = seed
    return args


def main():
    parser = ArgumentParser()
    parser.add_argument("task_name")
    parser.add_argument("task_config")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--render-freq", type=int, default=0)
    parser.add_argument("--slow", type=float, default=0)
    parser.add_argument("--hold", action="store_true")
    args = parser.parse_args()

    task = class_decorator(args.task_name)
    task_args = build_args(args.task_name, args.task_config, args.seed)
    task_args["render_freq"] = args.render_freq
    task_args["render_sleep"] = args.slow

    try:
        task.setup_demo(**task_args)
        print("hammer_pose", task.hammer.get_pose().p.tolist(), task.hammer.get_pose().q.tolist())
        print("block_fp0", task.block.get_functional_point(0, "list"))
        print("block_fp1", task.block.get_functional_point(1, "list"))
        info = task.play_once()
        print("play_once info", info)
        print("plan_success", task.plan_success)
        print("final_hammer_fp0", task.hammer.get_functional_point(0, "list"))
        print("final_block_fp1", task.block.get_functional_point(1, "list"))
        print("hammer_block_contact", task.check_actors_contact(task.hammer.get_name(), task.block.get_name()))
        print("check_success", task.check_success() if task.plan_success else None)
        if args.hold and task_args["render_freq"]:
            input("Press Enter to close the viewer...")
    except Exception:
        traceback.print_exc()
    finally:
        try:
            task.close_env()
        except Exception:
            pass


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()
