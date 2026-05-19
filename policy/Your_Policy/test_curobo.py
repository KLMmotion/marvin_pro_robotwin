import torch
from curobo.cuda_robot_model.cuda_robot_model import CudaRobotModel, CudaRobotModelConfig
from curobo.types.base import TensorDeviceType
from curobo.types.robot import RobotConfig
from curobo.util_file import get_robot_path, join_path, load_yaml

tensor_args = TensorDeviceType()

# Modify to the absolute path of `curobo.yml`
config_file = load_yaml("/home/tianji//hzh/RoboTwin/assets/embodiments/tianji/curobo_right.yml")

urdf_file = config_file["robot_cfg"]["kinematics"]["urdf_path"]
base_link = config_file["robot_cfg"]["kinematics"]["base_link"]
ee_link = config_file["robot_cfg"]["kinematics"]["ee_link"]
robot_cfg = RobotConfig.from_basic(urdf_file, base_link, ee_link, tensor_args)
kin_model = CudaRobotModel(robot_cfg.kinematics)
q = torch.rand((10, kin_model.get_dof()), **(tensor_args.as_torch_dict()))
out = kin_model.get_state(q)