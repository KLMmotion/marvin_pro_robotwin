import sys
import traceback
from argparse import ArgumentParser

sys.path.append("./")

import numpy as np
import torch.multiprocessing as mp
import transforms3d as t3d

from envs._GLOBAL_CONFIGS import GRASP_DIRECTION_DIC
from envs.utils import ArmTag, cal_quat_dis
from script.collect_data import class_decorator
from script.debug_task_once import build_args


CONTACT_TO_GRASP = np.array([
    [0, 0, 1, 0],
    [-1, 0, 0, 0],
    [0, -1, 0, 0],
    [0, 0, 0, 1],
])


def full_qpos_after_arm_path(task, arm, arm_qpos):
    entity = task.robot.right_entity if arm == "right" else task.robot.left_entity
    joints = task.robot.right_arm_joints if arm == "right" else task.robot.left_arm_joints
    active_joints = entity.get_active_joints()
    qpos = entity.get_qpos().copy()
    for value, joint in zip(arm_qpos, joints):
        qpos[active_joints.index(joint)] = value
    return qpos


def plan_status(result):
    status = result.get("status") if isinstance(result, dict) else result
    steps = len(result.get("position", [])) if isinstance(result, dict) and result.get("position") is not None else None
    return status, steps


def make_raw_pre_pose(actor, contact_point_id, pre_dis):
    contact_matrix = actor.get_contact_point(contact_point_id, "matrix")
    grasp_matrix = contact_matrix @ CONTACT_TO_GRASP
    pos = grasp_matrix[:3, 3] + grasp_matrix[:3, :3] @ np.array([-0.12 - pre_dis, 0, 0])
    quat = t3d.quaternions.mat2quat(grasp_matrix[:3, :3])
    return pos.tolist() + quat.tolist(), actor.get_contact_point(contact_point_id, "list")


def make_grasp_pose(pre_pose, pre_dis, target_dis):
    pose = np.array(pre_pose)
    direction_mat = t3d.quaternions.quat2mat(pose[-4:])
    pose[:3] += [pre_dis - target_dis, 0, 0] @ np.linalg.inv(direction_mat)
    return pose.tolist()


def main():
    parser = ArgumentParser()
    parser.add_argument("task_name")
    parser.add_argument("task_config")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--arm", choices=["left", "right"], default="right")
    parser.add_argument("--pre-dis", type=float, default=0.04)
    parser.add_argument("--target-dis", type=float, nargs="+", default=[0.01, 0.02, 0.03, 0.04])
    parser.add_argument("--roll", type=float, nargs="+", default=[0.0])
    args = parser.parse_args()

    task = class_decorator(args.task_name)
    task_args = build_args(args.task_name, args.task_config, args.seed)

    try:
        task.setup_demo(**task_args)
        arm = ArmTag(args.arm)
        plan_multi = task.robot.right_plan_multi_path if arm == "right" else task.robot.left_plan_multi_path
        plan_path = task.robot.right_plan_path if arm == "right" else task.robot.left_plan_path
        pref_direction = task.robot.get_grasp_perfect_direction(arm)

        print("hammer_pose", task.hammer.get_pose().p.tolist(), task.hammer.get_pose().q.tolist())
        print("arm", arm, "pre_dis", args.pre_dis, "target_dis_values", args.target_dis)

        successes = []
        for contact_id, contact_pose in task.hammer.iter_contact_points("list"):
            raw_pre_pose, center_pose = make_raw_pre_pose(task.hammer, contact_id, args.pre_dis)
            base_pre_candidates = task.robot.create_target_pose_list(raw_pre_pose, center_pose, arm)
            pre_candidates = []
            roll_tags = []
            for base_idx, base_pose in enumerate(base_pre_candidates):
                for roll in args.roll:
                    mat = t3d.quaternions.quat2mat(base_pose[-4:])
                    mat = mat @ t3d.axangles.axangle2mat([1, 0, 0], roll)
                    pre_candidates.append(base_pose[:3] + t3d.quaternions.mat2quat(mat).tolist())
                    roll_tags.append((base_idx, roll))
            pre_batch = plan_multi(pre_candidates)
            print(f"\ncontact_id={contact_id} center={center_pose}")
            for idx, pre_pose in enumerate(pre_candidates):
                base_idx, roll = roll_tags[idx]
                pre_status = pre_batch["status"][idx]
                pre_steps = len(pre_batch["position"][idx]) if pre_status == "Success" else None
                if pre_status != "Success":
                    print(f"  cand={idx:02d} base={base_idx:02d} roll={roll:.3f} pre={pre_status}")
                    continue

                pre_final_qpos = full_qpos_after_arm_path(task, arm, pre_batch["position"][idx][-1])
                rows = []
                for target_dis in args.target_dis:
                    grasp_pose = make_grasp_pose(pre_pose, args.pre_dis, target_dis)
                    grasp_result = plan_path(
                        grasp_pose,
                        constraint_pose=[1, 1, 1, 0, 0, 0],
                        last_qpos=pre_final_qpos,
                    )
                    grasp_status, grasp_steps = plan_status(grasp_result)
                    top_down_dis = cal_quat_dis(
                        grasp_pose[-4:],
                        GRASP_DIRECTION_DIC["top_down_little_left" if arm == "right" else "top_down_little_right"],
                    )
                    side_dis = cal_quat_dis(grasp_pose[-4:], GRASP_DIRECTION_DIC[pref_direction])
                    rows.append(f"d={target_dis:.3f}:{grasp_status}:{grasp_steps}")
                    if grasp_status == "Success":
                        successes.append((contact_id, idx, target_dis, pre_steps, grasp_steps, top_down_dis, side_dis,
                                          pre_pose, grasp_pose))
                print(f"  cand={idx:02d} base={base_idx:02d} roll={roll:.3f} pre=Success:{pre_steps} " + " ".join(rows))

        print("\nSUCCESS_COUNT", len(successes))
        for item in sorted(successes, key=lambda x: (x[3] + x[4], x[5] + x[6]))[:20]:
            contact_id, idx, target_dis, pre_steps, grasp_steps, top_down_dis, side_dis, pre_pose, grasp_pose = item
            base_idx, roll = roll_tags[idx]
            print(
                f"success contact={contact_id} cand={idx} base={base_idx} roll={roll:.3f} target_dis={target_dis:.3f} "
                f"steps={pre_steps}+{grasp_steps} top_down_dis={top_down_dis:.3f} side_dis={side_dis:.3f}"
            )
            print("  pre_pose", pre_pose)
            print("  grasp_pose", grasp_pose)
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
