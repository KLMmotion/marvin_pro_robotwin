import os
import sys
import traceback
from argparse import ArgumentParser

sys.path.append("./")

import numpy as np
import torch.multiprocessing as mp
import transforms3d as t3d

from script.collect_data import class_decorator
from script.debug_task_once import build_args


VARIANTS = {
    "current": [[1, 0, 0], [0, -1, 0], [0, 0, -1]],
    "identity": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
    "franka": [[0, 0, 1], [0, -1, 0], [1, 0, 0]],
    "piper_delta": [[0, 0, -1], [0, 1, 0], [1, 0, 0]],
    "x_z_y": [[0, 0, 1], [1, 0, 0], [0, 1, 0]],
    "x_negz_negy": [[0, 0, 1], [-1, 0, 0], [0, -1, 0]],
    "z_negx_negy": [[0, -1, 0], [0, 0, -1], [1, 0, 0]],
}


def set_delta(task, matrix):
    matrix = np.array(matrix)
    task.robot.left_delta_matrix = matrix
    task.robot.right_delta_matrix = matrix
    task.robot.left_inv_delta_matrix = np.linalg.inv(matrix)
    task.robot.right_inv_delta_matrix = np.linalg.inv(matrix)


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
        contact_matrix = task.hammer.get_contact_point(0, "matrix")
        grasp_matrix = contact_matrix @ np.array([[0, 0, 1, 0], [-1, 0, 0, 0], [0, -1, 0, 0], [0, 0, 0, 1]])
        raw_pose = grasp_matrix[:3, 3] + grasp_matrix[:3, :3] @ np.array([-0.16, 0, 0])
        raw_pose = raw_pose.tolist() + t3d.quaternions.mat2quat(grasp_matrix[:3, :3]).tolist()
        print("raw_pregrasp_pose", raw_pose)

        for name, matrix in VARIANTS.items():
            set_delta(task, matrix)
            result = task.robot.right_plan_path(raw_pose)
            status = result.get("status") if isinstance(result, dict) else result
            steps = len(result.get("position", [])) if isinstance(result, dict) and result.get("position") is not None else None
            print(f"{name}: {status}:{steps} matrix={matrix}")
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
