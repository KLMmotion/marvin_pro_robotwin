import torch
from curobo.types.base import TensorDeviceType
from curobo.types.math import Pose
from curobo.types.robot import RobotConfig
from curobo.util_file import get_robot_configs_path, join_path, load_yaml
from curobo.wrap.reacher.ik_solver import IKSolver, IKSolverConfig
from pathlib import Path

tensor_args = TensorDeviceType()
# Choose which arm to calibrate: "left" or "right".
arm = "right"

repo_root = Path(__file__).resolve().parents[2]
config_path = repo_root / "assets" / "embodiments" / "tianji" / f"curobo_{arm}.yml"
config_file = load_yaml(str(config_path))
urdf_file = config_file["robot_cfg"]["kinematics"]["urdf_path"]
base_link = config_file["robot_cfg"]["kinematics"]["base_link"]
ee_link = config_file["robot_cfg"]["kinematics"]["ee_link"]
robot_cfg = RobotConfig.from_basic(urdf_file, base_link, ee_link, tensor_args)

ik_config = IKSolverConfig.load_from_robot_config(
    robot_cfg,
    None,
    num_seeds=20,
    self_collision_check=False,
    self_collision_opt=False,
    tensor_args=tensor_args,
    use_cuda_graph=True,
)
ik_solver = IKSolver(ik_config)
x_values = torch.linspace(0.35, 0.0, 25).tolist() + torch.linspace(0.35, 0.7, 25).tolist()
y_values = torch.linspace(0.25, 0.0, 25).tolist() + torch.linspace(0.25, 0.5, 25).tolist()
z_values = torch.linspace(1.0, 0.4, 25).tolist()
quaternion = torch.tensor([[1.0, 0.0, 0.0, 0.0]], device='cuda:0')

print("Testing IK solutions for different positions:")
print("x, y, z, success")
for x in x_values:
    for y in y_values:
        for z in z_values:
            goal = Pose(
                position=torch.tensor([[float(x), float(y), float(z)]], device='cuda:0'),
                quaternion=quaternion
            )
            result = ik_solver.solve_single(goal)
            if result.success.item() == True:
                print(f"{x:.2f}, {y:.2f}, {z:.2f}, {result.success}")