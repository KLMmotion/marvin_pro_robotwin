import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import transforms3d as t3d


REPO_ROOT = Path(__file__).resolve().parents[2]
URDF_PATH = REPO_ROOT / "assets" / "embodiments" / "tianji" / "marvin_robot.urdf"


def rpy_to_mat(rpy):
    return t3d.euler.euler2mat(*rpy, axes="sxyz")


def load_urdf_joints(urdf_path: Path):
    root = ET.parse(urdf_path).getroot()
    joints = {}
    child_to_joint = {}
    for joint in root.findall("joint"):
        name = joint.attrib["name"]
        origin = joint.find("origin")
        xyz = np.fromstring(origin.attrib.get("xyz", "0 0 0"), sep=" ")
        rpy = np.fromstring(origin.attrib.get("rpy", "0 0 0"), sep=" ")
        axis_node = joint.find("axis")
        axis = (np.fromstring(axis_node.attrib.get("xyz", "0 0 1"), sep=" ")
                if axis_node is not None else np.array([0.0, 0.0, 1.0]))
        joints[name] = {
            "type": joint.attrib["type"],
            "parent": joint.find("parent").attrib["link"],
            "child": joint.find("child").attrib["link"],
            "xyz": xyz,
            "rpy": rpy,
            "axis": axis,
        }
        child_to_joint[joint.find("child").attrib["link"]] = name
    return joints, child_to_joint


def path_to_link(link_name: str, child_to_joint, joints, stop_link=None):
    path = []
    current = link_name
    while current in child_to_joint and current != stop_link:
        joint_name = child_to_joint[current]
        path.append(joint_name)
        current = joints[joint_name]["parent"]
        if current == stop_link:
            break
    return list(reversed(path))


def joint_transform(joint_cfg, q=0.0):
    transform = np.eye(4)
    transform[:3, :3] = rpy_to_mat(joint_cfg["rpy"])
    transform[:3, 3] = joint_cfg["xyz"]
    if joint_cfg["type"] in ("revolute", "continuous"):
        axis = joint_cfg["axis"] / np.linalg.norm(joint_cfg["axis"])
        rot = np.eye(4)
        rot[:3, :3] = t3d.axangles.axangle2mat(axis, q)
        transform = transform @ rot
    return transform


def fk_relative(link_name: str, stop_link: str, child_to_joint, joints):
    transform = np.eye(4)
    for joint_name in path_to_link(link_name, child_to_joint, joints, stop_link=stop_link):
        transform = transform @ joint_transform(joints[joint_name], q=0.0)
    return transform


def nearest_axis(vec):
    vec = vec / np.linalg.norm(vec)
    candidates = []
    axis_labels = [
        ("+x", np.array([1.0, 0.0, 0.0])),
        ("-x", np.array([-1.0, 0.0, 0.0])),
        ("+y", np.array([0.0, 1.0, 0.0])),
        ("-y", np.array([0.0, -1.0, 0.0])),
        ("+z", np.array([0.0, 0.0, 1.0])),
        ("-z", np.array([0.0, 0.0, -1.0])),
    ]
    for label, axis in axis_labels:
        candidates.append((float(np.dot(vec, axis)), label, axis))
    _, label, axis = max(candidates, key=lambda item: item[0])
    return label, axis


def main():
    joints, child_to_joint = load_urdf_joints(URDF_PATH)

    ee_in_arm = fk_relative("right_tool", "Base_R", child_to_joint, joints)
    ee_rot = ee_in_arm[:3, :3]

    tool_offset = joints["right_gripper_base_to_right_tool"]["xyz"]
    tool_dir = tool_offset / np.linalg.norm(tool_offset)

    finger_joint = joints["right_gripper_left_finger_joint"]
    finger_axis = rpy_to_mat(finger_joint["rpy"]) @ finger_joint["axis"]
    finger_axis = finger_axis / np.linalg.norm(finger_axis)

    tool_axis_name, tool_axis = nearest_axis(tool_dir)
    finger_axis_name, finger_axis_aligned = nearest_axis(finger_axis)

    # RoboTwin reference frame convention from the official tutorial:
    # x_ref: gripper forward direction
    # y_ref: parallel to gripper opening direction
    # z_ref: right-hand rule
    #
    # For Tianji, the URDF geometry gives:
    # - gripper forward ~ +z_ee
    # - opening direction ~ -x_ee / +x_ee (sign is physically symmetric)
    x_ref_in_ee = np.array([0, 0, 1])
    y_ref_in_ee = np.array([-1, 0, 0])
    z_ref_in_ee = np.cross(x_ref_in_ee, y_ref_in_ee)
    delta_matrix = np.column_stack([x_ref_in_ee, y_ref_in_ee, z_ref_in_ee]).astype(int)

    alt_y_ref_in_ee = np.array([1, 0, 0])
    alt_z_ref_in_ee = np.cross(x_ref_in_ee, alt_y_ref_in_ee)
    delta_matrix_alt = np.column_stack([x_ref_in_ee, alt_y_ref_in_ee, alt_z_ref_in_ee]).astype(int)

    print("URDF:", URDF_PATH)
    print("ee rotation in each arm base frame (right arm, q=0):")
    print(np.round(ee_rot).astype(int))
    print()
    print("tool center direction in ee frame:", np.round(tool_dir, 6), "->", tool_axis_name)
    print("finger slide axis in ee frame:", np.round(finger_axis, 6), "->", finger_axis_name)
    print()
    print("Recommended delta_matrix for Tianji:")
    print(delta_matrix.tolist())
    print()
    print("Sign-flipped alternative if desktop visualization shows the ee Y-axis")
    print("parallel to the opposite finger motion direction:")
    print(delta_matrix_alt.tolist())


if __name__ == "__main__":
    main()
