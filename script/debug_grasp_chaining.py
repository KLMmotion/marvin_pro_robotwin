import sys
import traceback
from argparse import ArgumentParser

sys.path.append("./")

import numpy as np
import torch.multiprocessing as mp

from envs.utils import ArmTag
from script.collect_data import class_decorator
from script.debug_task_once import build_args


def full_qpos_from_drive_targets(task, arm):
    entity = task.robot.right_entity if arm == "right" else task.robot.left_entity
    joints = task.robot.right_arm_joints if arm == "right" else task.robot.left_arm_joints
    active_joints = entity.get_active_joints()
    qpos = entity.get_qpos().copy()
    for joint in joints:
        qpos[active_joints.index(joint)] = joint.get_drive_target()[0]
    return qpos


def full_qpos_after_arm_path(task, arm, arm_qpos):
    entity = task.robot.right_entity if arm == "right" else task.robot.left_entity
    joints = task.robot.right_arm_joints if arm == "right" else task.robot.left_arm_joints
    active_joints = entity.get_active_joints()
    qpos = entity.get_qpos().copy()
    for value, joint in zip(arm_qpos, joints):
        qpos[active_joints.index(joint)] = value
    return qpos


def arm_qpos(task, arm, qpos):
    entity = task.robot.right_entity if arm == "right" else task.robot.left_entity
    joints = task.robot.right_arm_joints if arm == "right" else task.robot.left_arm_joints
    active_joints = entity.get_active_joints()
    return np.array([qpos[active_joints.index(joint)] for joint in joints])


def status(result):
    return result.get("status"), len(result.get("position", [])) if result.get("position") is not None else None


def main():
    parser = ArgumentParser()
    parser.add_argument("task_name")
    parser.add_argument("task_config")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--arm", choices=["left", "right"], default="right")
    parser.add_argument("--pre-dis", type=float, default=0.04)
    parser.add_argument("--target-dis", type=float, default=0.01)
    parser.add_argument("--settle", type=int, default=0)
    args = parser.parse_args()

    task = class_decorator(args.task_name)
    task_args = build_args(args.task_name, args.task_config, args.seed)

    try:
        task.setup_demo(**task_args)
        arm = ArmTag(args.arm)
        pre_pose, grasp_pose = task.choose_grasp_pose(
            task.hammer,
            arm_tag=arm,
            pre_dis=args.pre_dis,
            target_dis=args.target_dis,
        )
        print("pre_pose", pre_pose)
        print("grasp_pose", grasp_pose)

        plan_path = task.robot.right_plan_path if arm == "right" else task.robot.left_plan_path
        pre_result = plan_path(pre_pose)
        print("pre_plan", status(pre_result))
        ideal_pre_qpos = full_qpos_after_arm_path(task, arm, pre_result["position"][-1])
        ideal_grasp = plan_path(grasp_pose, constraint_pose=[1, 1, 1, 0, 0, 0], last_qpos=ideal_pre_qpos)
        print("grasp_from_ideal_pre_qpos", status(ideal_grasp))

        task.take_dense_action({
            "left_arm": pre_result if arm == "left" else None,
            "left_gripper": None,
            "right_arm": pre_result if arm == "right" else None,
            "right_gripper": None,
        })
        for _ in range(args.settle):
            task.scene.step()

        actual_qpos = task.robot.right_entity.get_qpos() if arm == "right" else task.robot.left_entity.get_qpos()
        drive_qpos = full_qpos_from_drive_targets(task, arm)
        print("actual_arm_qpos", arm_qpos(task, arm, actual_qpos).tolist())
        print("drive_arm_qpos", arm_qpos(task, arm, drive_qpos).tolist())
        print("ideal_arm_qpos", arm_qpos(task, arm, ideal_pre_qpos).tolist())
        print("actual_minus_drive_abs_max", float(np.max(np.abs(arm_qpos(task, arm, actual_qpos) - arm_qpos(task, arm, drive_qpos)))))
        print("actual_minus_ideal_abs_max", float(np.max(np.abs(arm_qpos(task, arm, actual_qpos) - arm_qpos(task, arm, ideal_pre_qpos)))))

        actual_grasp = plan_path(grasp_pose, constraint_pose=[1, 1, 1, 0, 0, 0])
        drive_grasp = plan_path(grasp_pose, constraint_pose=[1, 1, 1, 0, 0, 0], last_qpos=drive_qpos)
        print("grasp_from_actual_qpos", status(actual_grasp))
        print("grasp_from_drive_qpos", status(drive_grasp))
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
