import os
import sys
import traceback
from argparse import ArgumentParser

sys.path.append("./")

import torch.multiprocessing as mp

from envs._GLOBAL_CONFIGS import GRASP_DIRECTION_DIC
from script.collect_data import class_decorator
from script.debug_task_once import build_args


def plan_status(task, arm, pose):
    plan_func = task.robot.left_plan_path if arm == "left" else task.robot.right_plan_path
    result = plan_func(pose)
    status = result.get("status") if isinstance(result, dict) else result
    steps = len(result.get("position", [])) if isinstance(result, dict) and result.get("position") is not None else None
    return status, steps


def main():
    parser = ArgumentParser()
    parser.add_argument("task_name")
    parser.add_argument("task_config")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    task = class_decorator(args.task_name)
    task_args = build_args(args.task_name, args.task_config, args.seed)

    try:
        task.setup_demo(**task_args)
        hammer_contact = task.hammer.get_contact_point(0, "list")
        hammer_pose = task.hammer.get_pose().p.tolist()
        probe_positions = {
            "hammer_contact": hammer_contact[:3],
            "manual_pre_x": [hammer_contact[0] - 0.16, hammer_contact[1], hammer_contact[2]],
            "manual_grasp_x": [hammer_contact[0] - 0.13, hammer_contact[1], hammer_contact[2]],
            "pregrasp_0p04": [0.0005479099712794098, -0.09294599202547815, 0.9337052164850569],
            "pregrasp_0p12": [0.0007763283543628748, -0.10946794461122672, 1.0119802144732977],
            "place_failed": [0.19849627073377862, 0.07899469415676245, 1.0953303759866853],
            "table_center_high": [0.0, -0.1, 0.9],
            "robot_center_high": [0.0, -0.25, 0.9],
        }
        probe_quats = {
            "current": [0.5483463278501112, -0.4448468164549634, 0.44592399270923155, 0.5500721837626761],
            "identity": [1, 0, 0, 0],
            "top_down": GRASP_DIRECTION_DIC["top_down"],
            "top_down_little_left": GRASP_DIRECTION_DIC["top_down_little_left"],
            "top_down_little_right": GRASP_DIRECTION_DIC["top_down_little_right"],
            "front": GRASP_DIRECTION_DIC["front"],
            "front_left": GRASP_DIRECTION_DIC["front_left"],
            "front_right": GRASP_DIRECTION_DIC["front_right"],
            "left": GRASP_DIRECTION_DIC["left"],
            "right": GRASP_DIRECTION_DIC["right"],
        }

        print("hammer_pose", hammer_pose)
        print("hammer_contact", hammer_contact)
        for pos_name, pos in probe_positions.items():
            print(f"\n[position] {pos_name}: {pos}")
            for quat_name, quat in probe_quats.items():
                pose = pos + quat
                row = []
                for arm in ["left", "right"]:
                    status, steps = plan_status(task, arm, pose)
                    row.append(f"{arm}={status}:{steps}")
                print(f"  {quat_name}: " + "  ".join(row))
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
