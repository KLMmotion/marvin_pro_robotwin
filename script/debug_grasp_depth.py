import sys
import traceback
from argparse import ArgumentParser

sys.path.append("./")

import torch.multiprocessing as mp

from envs.utils import ArmTag
from script.collect_data import class_decorator
from script.debug_task_once import build_args


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
        arm = ArmTag("right")
        for grasp_dis in [0.02, 0.01, 0.0, -0.01, -0.02, -0.03, -0.04, -0.05, -0.06, -0.08, -0.10]:
            task.plan_success = True
            try:
                pre_pose, grasp_pose = task.choose_grasp_pose(
                    task.hammer,
                    arm_tag=arm,
                    pre_dis=0.04,
                    target_dis=grasp_dis,
                )
                if pre_pose is None or grasp_pose is None:
                    print(f"grasp_dis={grasp_dis}: choose=None")
                    continue
                result = task.robot.right_plan_path(grasp_pose, constraint_pose=[1, 1, 1, 0, 0, 0])
                status = result.get("status") if isinstance(result, dict) else result
                print(
                    f"grasp_dis={grasp_dis}: status={status} "
                    f"pre_xyz={pre_pose[:3]} grasp_xyz={grasp_pose[:3]} quat={grasp_pose[3:]}"
                )
            except Exception as e:
                print(f"grasp_dis={grasp_dis}: error={e}")
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
