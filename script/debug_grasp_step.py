import sys
import traceback
from argparse import ArgumentParser

sys.path.append("./")

import torch.multiprocessing as mp
import numpy as np
import transforms3d as t3d

from envs.utils import Action, ArmTag
from script.collect_data import class_decorator
from script.debug_task_once import build_args


DELTA_VARIANTS = {
    "identity": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
    "current_old": [[1, 0, 0], [0, -1, 0], [0, 0, -1]],
    "franka": [[0, 0, 1], [0, -1, 0], [1, 0, 0]],
    "piper_delta": [[0, 0, -1], [0, 1, 0], [1, 0, 0]],
    "z_negx_negy": [[0, -1, 0], [0, 0, -1], [1, 0, 0]],
}


ROBOT_LINK_PREFIXES = (
    "base_link",
    "Base_",
    "Link",
    "flange_",
    "left_gripper",
    "right_gripper",
    "left_tool",
    "right_tool",
)


def _is_robot_link(name):
    return name.startswith(ROBOT_LINK_PREFIXES)


def _joint_stats(entity):
    qpos = entity.get_qpos()
    qvel = entity.get_qvel()
    return {
        "qpos_min": float(qpos.min()),
        "qpos_max": float(qpos.max()),
        "qvel_abs_max": float(abs(qvel).max()),
        "qvel_abs_mean": float(abs(qvel).mean()),
    }


def disable_robot_self_collision(task):
    for link in task.robot.left_entity.get_links():
        if not _is_robot_link(link.get_name()):
            continue
        for shape in link.get_collision_shapes():
            groups = list(shape.get_collision_groups())
            groups[2] = groups[2] | 1
            groups[3] = (groups[3] & 0xFFFF0000) | 1
            shape.set_collision_groups(groups)


def print_contacts(task, label):
    interesting_pairs = []
    robot_self_pairs = []
    all_pairs = []
    for contact in task.scene.get_contacts():
        a = contact.bodies[0].entity.name
        b = contact.bodies[1].entity.name
        all_pairs.append((a, b))
        if "hammer" in a or "hammer" in b or "gripper" in a or "gripper" in b:
            interesting_pairs.append((a, b))
        if _is_robot_link(a) and _is_robot_link(b):
            robot_self_pairs.append((a, b))

    unique_self = sorted(set(robot_self_pairs))
    unique_interesting = sorted(set(interesting_pairs))
    print(label, "contact_count", len(all_pairs))
    print(label, "robot_self_contact_count", len(robot_self_pairs), "unique", unique_self[:40])
    print(label, "interesting", unique_interesting[:60])
    print(label, "joint_stats", _joint_stats(task.robot.left_entity))


def _shape_world_vertices(body_pose, shape):
    vertices = np.asarray(shape.get_vertices())
    if vertices.size == 0:
        return vertices.reshape(0, 3)
    scale = np.asarray(shape.get_scale())
    local_pose = shape.get_local_pose()
    local_mat = local_pose.to_transformation_matrix()
    body_mat = body_pose.to_transformation_matrix()
    vertices = vertices * scale
    vertices = (local_mat[:3, :3] @ vertices.T + local_mat[:3, 3:4]).T
    return (body_mat[:3, :3] @ vertices.T + body_mat[:3, 3:4]).T


def _aabb(vertices):
    return vertices.min(axis=0).tolist(), vertices.max(axis=0).tolist()


def print_aabb(task, label):
    names = ["right_gripper_base", "right_gripper_left_finger", "right_gripper_right_finger"]
    print(label, "hammer_pose", task.hammer.get_pose().p.tolist(), task.hammer.get_pose().q.tolist())
    hammer_vertices = []
    for comp in task.hammer.actor.get_components():
        if hasattr(comp, "get_collision_shapes"):
            for shape in comp.get_collision_shapes():
                hammer_vertices.append(_shape_world_vertices(task.hammer.get_pose(), shape))
    if hammer_vertices:
        print(label, "hammer_aabb", _aabb(np.vstack(hammer_vertices)))
    for name in names:
        link = task.robot.right_entity.find_link_by_name(name)
        if link is None:
            continue
        vertices = []
        for shape in link.get_collision_shapes():
            vertices.append(_shape_world_vertices(link.get_pose(), shape))
        if vertices:
            print(label, name, "pose", link.get_pose().p.tolist(), link.get_pose().q.tolist())
            print(label, name, "aabb", _aabb(np.vstack(vertices)))


def main():
    parser = ArgumentParser()
    parser.add_argument("task_name")
    parser.add_argument("task_config")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--grasp-dis", type=float, default=0.01)
    parser.add_argument("--gripper-pos", type=float, default=0.0)
    parser.add_argument("--no-constraint", action="store_true")
    parser.add_argument("--identity-contact", action="store_true")
    parser.add_argument("--settle-steps", type=int, default=100)
    parser.add_argument("--no-plan", action="store_true")
    parser.add_argument("--setup-only", action="store_true")
    parser.add_argument("--robot-z-offset", type=float, default=0.0)
    parser.add_argument("--ignore-robot-self-collision", action="store_true")
    parser.add_argument("--gripper-bias", type=float, default=None)
    parser.add_argument("--print-aabb", action="store_true")
    parser.add_argument("--delta-variant", choices=sorted(DELTA_VARIANTS), default=None)
    parser.add_argument("--roll", type=float, default=None)
    parser.add_argument("--pose-offset-z", type=float, default=0.0)
    parser.add_argument("--attach-after-grasp", action="store_true")
    args = parser.parse_args()

    task = class_decorator(args.task_name)
    task_args = build_args(args.task_name, args.task_config, args.seed)
    if args.no_plan:
        task_args["need_plan"] = False
    if args.robot_z_offset:
        for key in ("left_embodiment_config", "right_embodiment_config"):
            robot_pose = task_args[key].get("robot_pose", [])
            for pose in robot_pose:
                pose[2] += args.robot_z_offset

    try:
        task.setup_demo(**task_args)
        if args.gripper_bias is not None:
            task.robot.left_gripper_bias = args.gripper_bias
            task.robot.right_gripper_bias = args.gripper_bias
        if args.delta_variant is not None:
            delta_matrix = np.array(DELTA_VARIANTS[args.delta_variant])
            task.robot.left_delta_matrix = delta_matrix
            task.robot.right_delta_matrix = delta_matrix
            task.robot.left_inv_delta_matrix = np.linalg.inv(delta_matrix)
            task.robot.right_inv_delta_matrix = np.linalg.inv(delta_matrix)
        if args.ignore_robot_self_collision:
            disable_robot_self_collision(task)
        arm = ArmTag("right")
        print("initial_hammer_pose", task.hammer.get_pose().p.tolist(), task.hammer.get_pose().q.tolist())
        print("initial_hammer_contact", task.hammer.get_contact_point(0, "list"))
        print("initial_left_ee", task.robot.get_left_ee_pose())
        print("initial_right_ee", task.robot.get_right_ee_pose())
        print_contacts(task, "contacts_initial")
        for _ in range(args.settle_steps):
            task.scene.step()
        print_contacts(task, "contacts_after_settle")
        if args.setup_only:
            return

        if args.identity_contact:
            contact_pose = task.hammer.get_contact_point(0, "list")[:3] + [1, 0, 0, 0]
            task.move((
                arm,
                [
                    Action(arm, "move", target_pose=contact_pose),
                    Action(arm, "close", target_gripper_pos=0.0),
                ],
            ))
        elif args.no_constraint:
            pre_pose, grasp_pose = task.choose_grasp_pose(
                task.hammer,
                arm_tag=arm,
                pre_dis=0.04,
                target_dis=args.grasp_dis,
            )
            task.move((
                arm,
                [
                    Action(arm, "move", target_pose=pre_pose),
                    Action(arm, "move", target_pose=grasp_pose),
                    Action(arm, "close", target_gripper_pos=0.0),
                ],
            ))
        else:
            if args.roll is None and args.pose_offset_z == 0.0:
                task.move(
                    task.grasp_actor(
                        task.hammer,
                        arm_tag=arm,
                        pre_grasp_dis=0.04,
                        grasp_dis=args.grasp_dis,
                        gripper_pos=args.gripper_pos,
                    ))
            else:
                pre_pose, grasp_pose = task.choose_grasp_pose(
                    task.hammer,
                    arm_tag=arm,
                    pre_dis=0.04,
                    target_dis=args.grasp_dis,
                )

                def adjust_pose(pose):
                    pose = pose.copy()
                    pose[2] += args.pose_offset_z
                    if args.roll is None:
                        return pose
                    mat = t3d.quaternions.quat2mat(pose[-4:])
                    mat = mat @ t3d.axangles.axangle2mat([1, 0, 0], args.roll)
                    return pose[:3] + t3d.quaternions.mat2quat(mat).tolist()

                task.move((
                    arm,
                    [
                        Action(arm, "move", target_pose=adjust_pose(pre_pose)),
                        Action(
                            arm,
                            "move",
                            target_pose=adjust_pose(grasp_pose),
                            constraint_pose=[1, 1, 1, 0, 0, 0],
                        ),
                        Action(arm, "close", target_gripper_pos=args.gripper_pos),
                    ],
                ))
        print("after_grasp_plan_success", task.plan_success)
        print("after_grasp_hammer_pose", task.hammer.get_pose().p.tolist(), task.hammer.get_pose().q.tolist())
        print("after_grasp_hammer_contact", task.hammer.get_contact_point(0, "list"))
        print("after_grasp_right_ee", task.robot.get_right_ee_pose())
        print("right_gripper_close", task.is_right_gripper_close())
        print_contacts(task, "contacts_after_grasp")
        if args.print_aabb:
            print_aabb(task, "aabb_after_grasp")
        if args.attach_after_grasp:
            parent = task.robot.right_entity.find_link_by_name("right_gripper_base")
            anchor = task.hammer.get_pose()
            pose0 = parent.get_pose().inv() * anchor
            pose1 = task.hammer.get_pose().inv() * anchor
            drive = task.scene.create_drive(parent, pose0, task.hammer.actor, pose1)
            for setter in (
                drive.set_limit_x,
                drive.set_limit_y,
                drive.set_limit_z,
                drive.set_limit_twist,
            ):
                setter(0, 0)
            drive.set_drive_property_x(1e6, 1e4)
            drive.set_drive_property_y(1e6, 1e4)
            drive.set_drive_property_z(1e6, 1e4)
            drive.set_drive_property_slerp(1e6, 1e4)
            print("attached_hammer_to_right_gripper_base")

        task.move(task.move_by_displacement(arm, z=0.07, move_axis="arm"))
        print("after_lift_plan_success", task.plan_success)
        print("after_lift_hammer_pose", task.hammer.get_pose().p.tolist(), task.hammer.get_pose().q.tolist())
        print("after_lift_hammer_contact", task.hammer.get_contact_point(0, "list"))
        print("after_lift_right_ee", task.robot.get_right_ee_pose())
        print_contacts(task, "contacts_after_lift")
        if args.print_aabb:
            print_aabb(task, "aabb_after_lift")
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
