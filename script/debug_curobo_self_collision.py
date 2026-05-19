#!/usr/bin/env python3
"""Report cuRobo-style self-collision sphere overlaps for a robot config.

This is intentionally CPU-only: it mirrors the collision-sphere pair test from
cuRobo's YAML inputs so we can debug link pairs even when CUDA/cuRobo is not
available in the current shell.
"""

from __future__ import annotations

import argparse
import math
import os
import xml.etree.ElementTree as ET
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import yaml


@dataclass
class Joint:
    name: str
    joint_type: str
    parent: str
    child: str
    origin_xyz: np.ndarray
    origin_rpy: np.ndarray
    axis: np.ndarray
    mimic: Tuple[str, float, float] | None


def _parse_vec(text: str | None, default: Iterable[float]) -> np.ndarray:
    if text is None:
        return np.array(list(default), dtype=float)
    return np.array([float(x) for x in text.split()], dtype=float)


def _rpy_matrix(rpy: np.ndarray) -> np.ndarray:
    roll, pitch, yaw = rpy
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]], dtype=float)
    ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]], dtype=float)
    rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]], dtype=float)
    return rz @ ry @ rx


def _translation_matrix(xyz: np.ndarray) -> np.ndarray:
    out = np.eye(4)
    out[:3, 3] = xyz
    return out


def _rotation_axis_matrix(axis: np.ndarray, angle: float) -> np.ndarray:
    norm = np.linalg.norm(axis)
    if norm == 0.0:
        return np.eye(4)
    x, y, z = axis / norm
    c, s = math.cos(angle), math.sin(angle)
    c1 = 1.0 - c
    rot = np.array(
        [
            [c + x * x * c1, x * y * c1 - z * s, x * z * c1 + y * s],
            [y * x * c1 + z * s, c + y * y * c1, y * z * c1 - x * s],
            [z * x * c1 - y * s, z * y * c1 + x * s, c + z * z * c1],
        ],
        dtype=float,
    )
    out = np.eye(4)
    out[:3, :3] = rot
    return out


def _origin_matrix(xyz: np.ndarray, rpy: np.ndarray) -> np.ndarray:
    out = _translation_matrix(xyz)
    out[:3, :3] = _rpy_matrix(rpy)
    return out


def _joint_motion_matrix(joint: Joint, value: float) -> np.ndarray:
    if joint.joint_type in {"revolute", "continuous"}:
        return _rotation_axis_matrix(joint.axis, value)
    if joint.joint_type == "prismatic":
        return _translation_matrix(joint.axis * value)
    return np.eye(4)


def _load_yaml(path: Path) -> dict:
    with path.open("r") as f:
        return yaml.safe_load(f)


def _resolve_path(path_text: str, repo_root: Path) -> Path:
    path_text = path_text.replace("${ASSETS_PATH}", str(repo_root))
    path = Path(os.path.expandvars(path_text)).expanduser()
    if not path.is_absolute():
        path = repo_root / path
    return path


def _parse_urdf(urdf_path: Path) -> Tuple[Dict[str, Joint], Dict[str, List[Joint]]]:
    root = ET.parse(urdf_path).getroot()
    joints: Dict[str, Joint] = {}
    children: Dict[str, List[Joint]] = defaultdict(list)
    for node in root.findall("joint"):
        name = node.attrib["name"]
        origin = node.find("origin")
        axis = node.find("axis")
        mimic = node.find("mimic")
        mimic_data = None
        if mimic is not None:
            mimic_data = (
                mimic.attrib["joint"],
                float(mimic.attrib.get("multiplier", "1.0")),
                float(mimic.attrib.get("offset", "0.0")),
            )
        joint = Joint(
            name=name,
            joint_type=node.attrib.get("type", "fixed"),
            parent=node.find("parent").attrib["link"],
            child=node.find("child").attrib["link"],
            origin_xyz=_parse_vec(origin.attrib.get("xyz") if origin is not None else None, [0, 0, 0]),
            origin_rpy=_parse_vec(origin.attrib.get("rpy") if origin is not None else None, [0, 0, 0]),
            axis=_parse_vec(axis.attrib.get("xyz") if axis is not None else None, [0, 0, 0]),
            mimic=mimic_data,
        )
        joints[name] = joint
        children[joint.parent].append(joint)
    return joints, children


def _make_joint_values(kin_cfg: dict, joints: Dict[str, Joint], overrides: List[str]) -> Dict[str, float]:
    q: Dict[str, float] = {}
    cspace = kin_cfg.get("cspace", {})
    for name, value in zip(cspace.get("joint_names", []), cspace.get("retract_config", [])):
        q[name] = float(value)
    q.update({k: float(v) for k, v in (kin_cfg.get("lock_joints") or {}).items()})
    for item in overrides:
        name, value = item.split("=", 1)
        q[name] = float(value)

    _apply_mimic_values(q, joints)
    return q


def _make_trajectory_joint_values(
    kin_cfg: dict,
    joints: Dict[str, Joint],
    joint_names: Iterable[str],
    joint_values: Iterable[float],
) -> Dict[str, float]:
    q = _make_joint_values(kin_cfg, joints, [])
    for name, value in zip(joint_names, joint_values):
        q[str(name)] = float(value)
    _apply_mimic_values(q, joints)
    return q


def _apply_mimic_values(q: Dict[str, float], joints: Dict[str, Joint]) -> None:
    changed = True
    while changed:
        changed = False
        for joint in joints.values():
            if joint.mimic is None:
                continue
            src, multiplier, offset = joint.mimic
            if src in q and q.get(joint.name) != q[src] * multiplier + offset:
                q[joint.name] = q[src] * multiplier + offset
                changed = True


def _compute_link_poses(base_link: str, children: Dict[str, List[Joint]], q: Dict[str, float]) -> Dict[str, np.ndarray]:
    poses = {base_link: np.eye(4)}
    stack = [base_link]
    while stack:
        parent = stack.pop()
        for joint in children.get(parent, []):
            value = q.get(joint.name, 0.0)
            poses[joint.child] = poses[parent] @ _origin_matrix(joint.origin_xyz, joint.origin_rpy) @ _joint_motion_matrix(joint, value)
            stack.append(joint.child)
    return poses


def _collision_sphere_offset(collision_sphere_buffer, link: str) -> float:
    if isinstance(collision_sphere_buffer, dict):
        return float(collision_sphere_buffer.get(link, 0.0))
    if collision_sphere_buffer is None:
        return 0.0
    return float(collision_sphere_buffer)


def _self_collision_buffer(kin_cfg: dict, link: str) -> float:
    return float((kin_cfg.get("self_collision_buffer") or {}).get(link, 0.0))


def _is_ignored(ignore: dict, link_a: str, link_b: str) -> bool:
    return link_b in (ignore or {}).get(link_a, []) or link_a in (ignore or {}).get(link_b, [])


def _format_pair(row: dict) -> str:
    ignored = " ignored" if row["ignored"] else ""
    return (
        f"{row['link_a']}[{row['sphere_a']}] <-> {row['link_b']}[{row['sphere_b']}]: "
        f"overlap={row['overlap'] * 1000:.2f} mm, "
        f"center_dist={row['center_dist'] * 1000:.2f} mm, "
        f"threshold={row['threshold'] * 1000:.2f} mm{ignored}"
    )


def _merged_ignore(ignore: dict, rows: List[dict], collision_links: List[str]) -> Dict[str, List[str]]:
    link_order = {name: i for i, name in enumerate(collision_links)}
    merged = {k: list(v or []) for k, v in (ignore or {}).items()}
    for row in rows:
        link_a, link_b = row["link_a"], row["link_b"]
        if _is_ignored(merged, link_a, link_b):
            continue
        first, second = sorted([link_a, link_b], key=lambda name: link_order.get(name, 10**9))
        merged.setdefault(first, []).append(second)
    for key in list(merged.keys()):
        merged[key] = sorted(set(merged[key]), key=lambda name: link_order.get(name, 10**9))
        if not merged[key]:
            del merged[key]
    return {k: merged[k] for k in sorted(merged, key=lambda name: link_order.get(name, 10**9))}


def _format_ignore_block(ignore: Dict[str, List[str]], indent: int = 6) -> str:
    """Format ignore pairs in the same style as cuRobo YAML configs in assets."""
    pad = " " * indent
    key_pad = " " * (indent + 2)
    item_pad = " " * (indent + 4)
    lines = [f"{pad}{{"]
    items = list(ignore.items())
    for key_idx, (link, ignored_links) in enumerate(items):
        lines.append(f'{key_pad}"{link}": [')
        for value_idx, ignored_link in enumerate(ignored_links):
            comma = "," if value_idx < len(ignored_links) - 1 else ""
            lines.append(f'{item_pad}"{ignored_link}"{comma}')
        comma = "," if key_idx < len(items) - 1 else ""
        lines.append(f"{key_pad}]{comma}")
    lines.append(f"{pad}}}")
    return "\n".join(lines)


def _find_collision_rows(
    q: Dict[str, float],
    kin_cfg: dict,
    joints: Dict[str, Joint],
    children: Dict[str, List[Joint]],
    sphere_cfg: dict,
    collision_links: List[str],
    ignore: dict,
    include_ignored: bool,
    min_overlap_mm: float,
) -> List[dict]:
    poses = _compute_link_poses(kin_cfg.get("base_link", "base_link"), children, q)
    spheres = []
    for link in collision_links:
        if link not in poses:
            raise RuntimeError(f"Missing FK pose for collision link: {link}")
        sphere_offset = _collision_sphere_offset(kin_cfg.get("collision_sphere_buffer"), link)
        self_buffer = _self_collision_buffer(kin_cfg, link) - sphere_offset
        for sphere_idx, sphere in enumerate(sphere_cfg[link]):
            center_local = np.array([*sphere["center"], 1.0], dtype=float)
            center_world = (poses[link] @ center_local)[:3]
            padded_radius = float(sphere["radius"]) + sphere_offset
            spheres.append(
                {
                    "link": link,
                    "sphere": sphere_idx,
                    "center": center_world,
                    "radius": padded_radius,
                    "self_offset": self_buffer,
                }
            )

    rows = []
    for i, a in enumerate(spheres):
        for b in spheres[i + 1 :]:
            if a["link"] == b["link"]:
                continue
            ignored = _is_ignored(ignore, a["link"], b["link"])
            if ignored and not include_ignored:
                continue
            center_dist = float(np.linalg.norm(a["center"] - b["center"]))
            threshold = a["radius"] + b["radius"] + a["self_offset"] + b["self_offset"]
            overlap = threshold - center_dist
            if overlap * 1000.0 <= min_overlap_mm:
                continue
            rows.append(
                {
                    "link_a": a["link"],
                    "sphere_a": a["sphere"],
                    "link_b": b["link"],
                    "sphere_b": b["sphere"],
                    "overlap": overlap,
                    "center_dist": center_dist,
                    "threshold": threshold,
                    "ignored": ignored,
                }
            )
    rows.sort(key=lambda row: row["overlap"], reverse=True)
    return rows


def _summarize_rows(rows: List[dict]) -> Dict[Tuple[str, str], dict]:
    summary = {}
    for row in rows:
        key = (row["link_a"], row["link_b"])
        summary.setdefault(key, row)
    return summary


def _flatten_trajectory_positions(positions: np.ndarray) -> np.ndarray:
    positions = np.asarray(positions)
    if positions.ndim == 1:
        return positions.reshape(1, -1)
    if positions.ndim == 2:
        return positions
    return positions.reshape(-1, positions.shape[-1])


def main() -> int:
    repo_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--yml",
        default=str(repo_root / "assets/embodiments/tianji/curobo_left.yml"),
        help="cuRobo robot config yml.",
    )
    parser.add_argument("--joint", action="append", default=[], help="Override a joint value, e.g. Joint1_L=0.0")
    parser.add_argument("--top", type=int, default=40, help="Maximum sphere pairs to print.")
    parser.add_argument("--min-overlap-mm", type=float, default=0.0, help="Only print overlaps above this value.")
    parser.add_argument("--include-ignored", action="store_true", help="Also report pairs masked by self_collision_ignore.")
    parser.add_argument("--summary-only", action="store_true", help="Only print the worst overlap per link pair.")
    parser.add_argument("--suggest-ignore", action="store_true", help="Print a merged self_collision_ignore candidate.")
    parser.add_argument("--trajectory", help="NPZ exported by planner.py with failed cuRobo trajectory.")
    parser.add_argument(
        "--trajectory-key",
        default="trajopt_result_solution_position",
        help="NPZ array key containing waypoint joint positions.",
    )
    args = parser.parse_args()

    yml_path = Path(args.yml).expanduser().resolve()
    yml = _load_yaml(yml_path)
    kin_cfg = yml["robot_cfg"]["kinematics"]
    urdf_path = _resolve_path(kin_cfg["urdf_path"], repo_root)
    sphere_path = _resolve_path(kin_cfg["collision_spheres"], repo_root)
    sphere_cfg = _load_yaml(sphere_path)["collision_spheres"]
    collision_links = kin_cfg["collision_link_names"]
    ignore = kin_cfg.get("self_collision_ignore") or {}

    joints, children = _parse_urdf(urdf_path)

    print(f"config: {yml_path}")
    print(f"urdf:   {urdf_path}")

    if args.trajectory:
        traj_path = Path(args.trajectory).expanduser().resolve()
        data = np.load(traj_path)
        if args.trajectory_key not in data:
            raise KeyError(f"{args.trajectory_key} not found. Available keys: {list(data.keys())}")
        joint_names = [str(name) for name in data["joint_names"].tolist()]
        trajectory = _flatten_trajectory_positions(data[args.trajectory_key])
        all_rows = []
        worst_by_waypoint = []
        for waypoint_idx, joint_values in enumerate(trajectory):
            q = _make_trajectory_joint_values(kin_cfg, joints, joint_names, joint_values)
            rows = _find_collision_rows(
                q,
                kin_cfg,
                joints,
                children,
                sphere_cfg,
                collision_links,
                ignore,
                args.include_ignored,
                args.min_overlap_mm,
            )
            for row in rows:
                row["waypoint"] = waypoint_idx
            all_rows.extend(rows)
            if rows:
                worst_by_waypoint.append((waypoint_idx, rows[0]))

        all_rows.sort(key=lambda row: row["overlap"], reverse=True)
        summary = _summarize_rows(all_rows)
        print(f"trajectory: {traj_path}")
        print(f"trajectory_key: {args.trajectory_key}")
        print(f"joint_names: {joint_names}")
        print(f"waypoints: {len(trajectory)}")
        print(f"colliding waypoints: {len(worst_by_waypoint)}")
        print(f"colliding sphere pairs across trajectory: {len(all_rows)}")
        print(f"colliding link pairs across trajectory:   {len(summary)}")
        if worst_by_waypoint:
            print("\nworst waypoint collisions:")
            for waypoint_idx, row in worst_by_waypoint[: args.top]:
                print(f"  waypoint {waypoint_idx}: " + _format_pair(row))
        if all_rows:
            print("\nworst sphere pairs across trajectory:")
            for row in all_rows[: args.top]:
                print(f"  waypoint {row['waypoint']}: " + _format_pair(row))
        if args.suggest_ignore and all_rows:
            print("\nmerged self_collision_ignore candidate:")
            print(_format_ignore_block(_merged_ignore(ignore, all_rows, collision_links)))
        return 1 if all_rows else 0

    q = _make_joint_values(kin_cfg, joints, args.joint)
    rows = _find_collision_rows(
        q,
        kin_cfg,
        joints,
        children,
        sphere_cfg,
        collision_links,
        ignore,
        args.include_ignored,
        args.min_overlap_mm,
    )
    summary = _summarize_rows(rows)
    print("joint state:")
    for name in kin_cfg.get("cspace", {}).get("joint_names", []):
        print(f"  {name}: {q.get(name, 0.0): .6f}")
    for name in (kin_cfg.get("lock_joints") or {}):
        print(f"  {name}: {q.get(name, 0.0): .6f}  locked")

    print(f"\ncolliding sphere pairs: {len(rows)}")
    print(f"colliding link pairs:   {len(summary)}")
    if not rows:
        return 0

    print("\nworst per link pair:")
    for row in list(summary.values())[: args.top]:
        print("  " + _format_pair(row))

    if not args.summary_only:
        print("\nworst sphere pairs:")
        for row in rows[: args.top]:
            print("  " + _format_pair(row))
    if args.suggest_ignore:
        print("\nmerged self_collision_ignore candidate:")
        print(_format_ignore_block(_merged_ignore(ignore, rows, collision_links)))
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
