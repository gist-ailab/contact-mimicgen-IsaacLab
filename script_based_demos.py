# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
This script demonstrates how to use the operational space controller (OSC) with the simulator.

The OSC controller can be configured in different modes. It uses the dynamical quantities such as Jacobians and
mass matricescomputed by PhysX.

.. code-block:: bash

    # Usage
    ./isaaclab.sh -p scripts/tutorials/05_controllers/run_osc.py

"""

"""Launch Isaac Sim Simulator first."""

import argparse

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Tutorial on using the operational space controller.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to spawn.")
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, AssetBaseCfg, RigidObjectCfg, ArticulationCfg
from isaaclab.controllers import OperationalSpaceController, OperationalSpaceControllerCfg
from isaaclab.markers import VisualizationMarkers
from isaaclab.markers.config import FRAME_MARKER_CFG
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import UsdFileCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR, ISAACLAB_NUCLEUS_DIR
from isaaclab.utils.math import (
    combine_frame_transforms,
    matrix_from_quat,
    quat_apply_inverse,
    quat_inv,
    subtract_frame_transforms,
)

##
# Pre-defined configs
##
from isaaclab_assets.robots.franka import FRANKA_PANDA_CFG  # isort:skip

ASSET_DIR = f"{ISAACLAB_NUCLEUS_DIR}/AutoMate"

@configclass
class SceneCfg(InteractiveSceneCfg):
    """Configuration for a simple scene with table, socket, and plug."""

    # ground plane
    ground = AssetBaseCfg(
        prim_path="/World/defaultGroundPlane",
        spawn=sim_utils.GroundPlaneCfg(),
    )

    # lights
    dome_light = AssetBaseCfg(
        prim_path="/World/Light", spawn=sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75))
    )

    # # Table - using the same configuration as boltnut_script_env_cfg.py
    # table = AssetBaseCfg(
    #     prim_path="{ENV_REGEX_NS}/Table",
    #     init_state=AssetBaseCfg.InitialStateCfg(pos=[0.5, 0, 0], rot=[0.707, 0, 0, 0.707]),
    #     spawn=UsdFileCfg(usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Mounts/SeattleLabTable/table_instanceable.usd"),
    # )
    # Table (kinematic cuboid)
    table = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Table",
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.5, 0.0, 0.4), rot=(1.0, 0.0, 0.0, 0.0)),
        spawn=sim_utils.CuboidCfg(
            size=(1.0, 0.6, 0.8),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.05, 0.05, 0.05), opacity=1),  # Red color for testing
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
        ),
    )
    
    # Socket
    socket = ArticulationCfg(
        prim_path="{ENV_REGEX_NS}/Socket",
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.6, 0.0, 0.8),  # On table surface
            rot=(1.0, 0.0, 0.0, 0.0),
            joint_pos={},
            joint_vel={},
        ),
        actuators={},
        spawn=UsdFileCfg(
            usd_path=f"{ASSET_DIR}/00186/socket.usd",
            activate_contact_sensors=True,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False,
                max_depenetration_velocity=5.0,
                linear_damping=0.0,
                angular_damping=0.0,
                max_linear_velocity=1000.0,
                max_angular_velocity=3666.0,
                enable_gyroscopic_forces=True,
                solver_position_iteration_count=192,
                solver_velocity_iteration_count=1,
                max_contact_impulse=1e32,
            ),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=True,
                fix_root_link=True,
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.05),
            collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.005, rest_offset=0.0),
            semantic_tags=[("class", "socket")],
        ),
    )

    # Plug
    plug = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Plug",
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(0.5, 0.0, 0.85),  # On table surface
            rot=(1.0, 0.0, 0.0, 0.0),
        ),
        spawn=UsdFileCfg(
            usd_path=f"{ASSET_DIR}/00186/plug.usd",
            activate_contact_sensors=True,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                kinematic_enabled=False,     # ★ 동적(Rigid)로
                disable_gravity=False,
                max_depenetration_velocity=5.0,
                linear_damping=0.0,
                angular_damping=0.0,
                max_linear_velocity=1000.0,
                max_angular_velocity=3666.0,
                enable_gyroscopic_forces=True,
                solver_position_iteration_count=100,
                solver_velocity_iteration_count=1,
                max_contact_impulse=1e32,
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.05),
            collision_props=sim_utils.CollisionPropertiesCfg(
                contact_offset=0.005,
                rest_offset=0.0,
                torsional_patch_radius=0.005,
            ),
            semantic_tags=[("class", "plug")],
        ),
    )

    # Robot - positioned on top of the table
    robot = FRANKA_PANDA_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    robot.init_state = ArticulationCfg.InitialStateCfg(
        pos=(0.2, 0.0, 0.8),  # On table surface
        rot=(1.0, 0.0, 0.0, 0.0),
        joint_pos={
            "panda_joint1": 0.0,
            "panda_joint2": -0.569,
            "panda_joint3": 0.0,
            "panda_joint4": -2.810,
            "panda_joint5": 0.0,
            "panda_joint6": 3.037,
            "panda_joint7": 0.741,
            "panda_finger_joint.*": 0.04,
        },
    )
    robot.actuators["panda_shoulder"].stiffness = 0.0
    robot.actuators["panda_shoulder"].damping = 0.0
    robot.actuators["panda_forearm"].stiffness = 0.0
    robot.actuators["panda_forearm"].damping = 0.0
    robot.spawn.rigid_props.disable_gravity = True

def generate_path_pose(pose1, pose2, current_step, total_steps, direction=1):
    """Generate interpolated pose along the path between pose1 and pose2.
    
    Args:
        pose1: Starting pose (7,)
        pose2: Ending pose (7,)
        current_step: Current step in the path
        total_steps: Total number of steps for the path
        direction: 1 for pose1->pose2, -1 for pose2->pose1
        
    Returns:
        interpolated_pose: Current pose along the path
    """
    # Calculate interpolation parameter (0 to 1)
    t = current_step / total_steps
    
    # Apply smooth interpolation (sine function for smooth acceleration/deceleration)
    smooth_t = 0.5 * (1 - torch.cos(torch.tensor(t * torch.pi, device=pose1.device)))
    
    # Interpolate position
    interpolated_pos = pose1[:3] * (1 - smooth_t) + pose2[:3] * smooth_t
    
    # Interpolate quaternion (slerp-like)
    interpolated_quat = pose1[3:] * (1 - smooth_t) + pose2[3:] * smooth_t
    interpolated_quat = interpolated_quat / torch.norm(interpolated_quat)
    
    return torch.cat([interpolated_pos, interpolated_quat])

def interpolate_target(start_pose, target_pose, smooth_t):
    """Interpolate the target pose between the start and target pose."""
    interpolated_pos = start_pose[:, :3] * (1 - smooth_t) + target_pose[:, :3] * smooth_t
    interpolated_quat = start_pose[:, 3:] * (1 - smooth_t) + target_pose[:, 3:] * smooth_t
    interpolated_quat = interpolated_quat / torch.norm(interpolated_quat, dim=1, keepdim=True)
    return torch.cat([interpolated_pos, interpolated_quat], dim=-1)

# State machine for pick and place task
# States: 1) Approach position, 2) Pick position, 3) Lift position, 4) Place position

def run_simulator(sim: sim_utils.SimulationContext, scene: InteractiveScene):
    """Runs the simulation loop.

    Args:
        sim: (SimulationContext) Simulation context.
        scene: (InteractiveScene) Interactive scene.
    """

    # Extract scene entities for readability.
    robot = scene["robot"]
    plug = scene["plug"]
    socket = scene["socket"]

    # Obtain indices for the end-effector and arm joints
    ee_frame_name = "panda_hand"  # Use hand center instead of finger tip
    arm_joint_names = ["panda_joint.*"]
    gripper_joint_names = ["panda_finger_joint.*"]
    ee_frame_idx = robot.find_bodies(ee_frame_name)[0][0]
    arm_joint_ids = robot.find_joints(arm_joint_names)[0]
    gripper_joint_ids = robot.find_joints(gripper_joint_names)[0]
    
    # Manual offset for end-effector position (adjust z height)
    ee_offset = torch.tensor([0.0, 0.0, -0.1], device=sim.device)  # Adjust z offset as needed

    # Create the OSC
    osc_cfg = OperationalSpaceControllerCfg(
        target_types=["pose_abs"],
        impedance_mode="variable_kp",
        inertial_dynamics_decoupling=True,
        partial_inertial_dynamics_decoupling=False,
        gravity_compensation=False,
        motion_damping_ratio_task=1.0,  # Reduced damping for smoother motion
        motion_control_axes_task=[1, 1, 1, 1, 1, 1],
        nullspace_control="position",
    )
    osc = OperationalSpaceController(osc_cfg, num_envs=scene.num_envs, device=sim.device)

    # Markers
    frame_marker_cfg = FRAME_MARKER_CFG.copy()
    frame_marker_cfg.markers["frame"].scale = (0.1, 0.1, 0.1)
    ee_marker = VisualizationMarkers(frame_marker_cfg.replace(prim_path="/Visuals/ee_current"))
    goal_marker = VisualizationMarkers(frame_marker_cfg.replace(prim_path="/Visuals/ee_goal"))

    # Get plug and socket positions dynamically
    # world → base 변환
    robot_pos_w = robot.data.root_pos_w[0]
    robot_quat_w = robot.data.root_quat_w[0]
    identity_quat = torch.tensor([0.0, 0.0, 0.0, 1.0], device=sim.device)

    plug_pos_w = plug.data.root_pos_w[0]
    socket_pos_w = socket.data.root_pos_w[0]

    plug_pos_b, _ = subtract_frame_transforms(
        robot_pos_w.unsqueeze(0), robot_quat_w.unsqueeze(0),
        plug_pos_w.unsqueeze(0), identity_quat.unsqueeze(0)
    )
    socket_pos_b, _ = subtract_frame_transforms(
        robot_pos_w.unsqueeze(0), robot_quat_w.unsqueeze(0),
        socket_pos_w.unsqueeze(0), identity_quat.unsqueeze(0)
    )
    plug_pos_b = plug_pos_b[0]
    socket_pos_b = socket_pos_b[0]

    # Define target poses for pick and place task based on actual positions
    # 1) Approach position (above the plug)
    base_quat = torch.tensor([0.0, 1.0, 0.0, 0.0], device=sim.device)
    target_pose_1 = torch.cat([plug_pos_b + torch.tensor([0.0, 0.0, 0.00], device=sim.device), base_quat])
    # 2) Pick position (at the plug)
    target_pose_2 = torch.cat([plug_pos_b + torch.tensor([0.0, 0.0, 0.00], device=sim.device), base_quat])
    # 3) Lift position (lifted plug)
    target_pose_3 = torch.cat([plug_pos_b + torch.tensor([0.0, 0.0, 0.05], device=sim.device), base_quat])
    # 4) Place position (above socket)
    target_pose_4 = torch.cat([socket_pos_b + torch.tensor([0.0, 0.0, 0.05], device=sim.device), base_quat])
    
    # Define simulation stepping
    sim_dt = sim.get_physics_dt()

    # State machine parameters
    path_duration = 2.0  # seconds per transition
    path_steps = int(path_duration / sim_dt)  # number of steps for each transition
    current_path_step = 0
    current_state = 0  # 0: approach, 1: pick, 2: lift, 3: place
    gripper_closed = False  # Start with gripper open

    # Update existing buffers
    # Note: We need to update buffers before the first step for the controller.
    robot.update(dt=sim_dt)

    # Get the center of the robot soft joint limits
    joint_centers = torch.mean(robot.data.soft_joint_pos_limits[:, arm_joint_ids, :], dim=-1)

    # get the updated states
    (
        jacobian_b,
        mass_matrix,
        gravity,
        ee_pose_b,
        ee_vel_b,
        root_pose_w,
        ee_pose_w,
        joint_pos,
        joint_vel,
    ) = update_states(sim, scene, robot, ee_frame_idx, arm_joint_ids, ee_offset)

    # Track the given target command
    command = torch.zeros(
        scene.num_envs, osc.action_dim, device=sim.device
    )  # Generic target command, which can be pose, position, force, etc.
    ee_target_pose_b = torch.zeros(scene.num_envs, 7, device=sim.device)  # Target pose in the body frame
    ee_target_pose_w = torch.zeros(scene.num_envs, 7, device=sim.device)  # Target pose in the world frame (for marker)

    # Set joint efforts to zero
    zero_joint_efforts = torch.zeros(scene.num_envs, robot.num_joints, device=sim.device)
    joint_efforts = torch.zeros(scene.num_envs, len(arm_joint_ids), device=sim.device)

    count = 0
    
    # reset joint state to default
    default_joint_pos = robot.data.default_joint_pos.clone()
    default_joint_vel = robot.data.default_joint_vel.clone()
    robot.write_joint_state_to_sim(default_joint_pos, default_joint_vel)
    robot.set_joint_effort_target(zero_joint_efforts)  # Set zero torques in the initial step
    robot.write_data_to_sim()
    robot.reset()
    
    osc.reset()
    osc.set_command(command=command, current_ee_pose_b=ee_pose_b, current_task_frame_pose_b=ee_pose_b)
    
    # Simulation loop
    while simulation_app.is_running():
        # 1) 최신 상태 먼저 읽기
        (
            jacobian_b,
            mass_matrix,
            gravity,
            ee_pose_b,
            ee_vel_b,
            root_pose_w,
            ee_pose_w,
            joint_pos,
            joint_vel,
        ) = update_states(sim, scene, robot, ee_frame_idx, arm_joint_ids, ee_offset)

        # Recompute plug/socket poses and targets each step to follow moving objects
        robot_pos_w = robot.data.root_pos_w[0]
        robot_quat_w = robot.data.root_quat_w[0]

        plug_pos_w = plug.data.root_pos_w[0]
        socket_pos_w = socket.data.root_pos_w[0]

        plug_pos_b, _ = subtract_frame_transforms(
            robot_pos_w.unsqueeze(0), robot_quat_w.unsqueeze(0),
            plug_pos_w.unsqueeze(0), identity_quat.unsqueeze(0)
        )
        socket_pos_b, _ = subtract_frame_transforms(
            robot_pos_w.unsqueeze(0), robot_quat_w.unsqueeze(0),
            socket_pos_w.unsqueeze(0), identity_quat.unsqueeze(0)
        )
        plug_pos_b = plug_pos_b[0]
        socket_pos_b = socket_pos_b[0]

        base_quat = torch.tensor([0.0, 1.0, 0.0, 0.0], device=sim.device)
        target_pose_1 = torch.cat([plug_pos_b + torch.tensor([0.0, 0.0, 0.03], device=sim.device), base_quat])
        target_pose_2 = torch.cat([plug_pos_b + torch.tensor([0.0, 0.0, 0.01], device=sim.device), base_quat])
        target_pose_3 = torch.cat([plug_pos_b + torch.tensor([0.0, 0.0, 0.05], device=sim.device), base_quat])
        target_pose_4 = torch.cat([socket_pos_b + torch.tensor([0.0, 0.0, 0.05], device=sim.device), base_quat])
        
        # 2) State machine logic for pick and place
        if current_state == 0:  # Approach to pick position
            current_target_pose = generate_path_pose(target_pose_1, target_pose_2, current_path_step, path_steps)
            if current_path_step >= path_steps:
                current_state = 1
                current_path_step = 0
                gripper_closed = True  # Close gripper at pick position
                
        elif current_state == 1:  # Pick position (wait for gripper to close)
            current_target_pose = target_pose_2
            if current_path_step >= path_steps // 2:  # Wait half the time
                current_state = 2
                current_path_step = 0
                
        elif current_state == 2:  # Lift the plug (slower and more careful)
            # Use longer duration for lifting
            lift_path_steps = int(3.0 / sim_dt)  # 3 seconds for lifting
            current_target_pose = generate_path_pose(target_pose_2, target_pose_3, current_path_step, lift_path_steps)
            if current_path_step >= lift_path_steps:
                current_state = 3
                current_path_step = 0
                
        elif current_state == 3:  # Move to place position (slower for careful placement)
            # Use longer duration for careful placement
            place_path_steps = int(2.5 / sim_dt)  # 2.5 seconds for placement
            current_target_pose = generate_path_pose(target_pose_3, target_pose_4, current_path_step, place_path_steps)
            if current_path_step >= place_path_steps:
                current_state = 4
                current_path_step = 0
                gripper_closed = False  # Open gripper at place position
                
        elif current_state == 4:  # Place position (wait for gripper to open)
            current_target_pose = target_pose_4
            if current_path_step >= path_steps // 2:  # Wait half the time for gripper to open
                current_state = 5
                current_path_step = 0
                
        elif current_state == 5:  # Return to approach position
            current_target_pose = generate_path_pose(target_pose_4, target_pose_1, current_path_step, path_steps)
            if current_path_step >= path_steps:
                current_state = 0  # Reset to start
                current_path_step = 0
                gripper_closed = False  # Open gripper for next cycle
        
        # Update path step counter
        current_path_step += 1
        
        # target_pose: shape (7,) -> (num_envs, 7)
        ee_target_pose_b = current_target_pose.unsqueeze(0).repeat(scene.num_envs, 1)

        # 3) 현재 EE 기준의 **상대 오차**로 커맨드 만들기
        rel_pos_b, rel_quat_b = subtract_frame_transforms(
            ee_pose_b[:, :3], ee_pose_b[:, 3:7],
            ee_target_pose_b[:, :3], ee_target_pose_b[:, 3:7]
        )
        command = torch.zeros(scene.num_envs, osc.action_dim, device=sim.device)
        command[:, 0:3] = rel_pos_b
        command[:, 3:7] = rel_quat_b
        command[:, 7:13] = torch.tensor([320., 320., 320., 320., 320., 320.], device=sim.device)

        # 4) 매 스텝 **반드시** set_command 호출 (task frame=현재 EE)
        osc.set_command(
            command=command,
            current_ee_pose_b=ee_pose_b,
            current_task_frame_pose_b=ee_pose_b
        )

        # 5) 제어 입력 계산/적용
        joint_efforts = osc.compute(
            jacobian_b=jacobian_b,
            current_ee_pose_b=ee_pose_b,
            current_ee_vel_b=ee_vel_b,
            current_ee_force_b=torch.zeros(scene.num_envs, 6, device=sim.device),
            mass_matrix=mass_matrix,
            gravity=gravity,
            current_joint_pos=joint_pos,
            current_joint_vel=joint_vel,
            nullspace_joint_pos_target=joint_centers,
        )
        robot.set_joint_effort_target(joint_efforts, joint_ids=arm_joint_ids)
        
        # Control gripper
        if gripper_closed:
            gripper_target = torch.tensor([0.0, 0.0], device=sim.device)  # Closed position
        else:
            gripper_target = torch.tensor([0.04, 0.04], device=sim.device)  # Open position
        
        # Apply gripper control (position control)
        robot.set_joint_position_target(gripper_target, joint_ids=gripper_joint_ids)
        
        robot.write_data_to_sim()

        # 마커 업데이트 - 같은 좌표계 사용
        # 현재 위치: world frame
        # ee_marker.visualize(ee_pose_w[:, 0:3], ee_pose_w[:, 3:7])
        
        # # 목표 위치: current_target_pose를 world frame으로 변환
        # target_pos_w, target_quat_w = combine_frame_transforms(
        #     root_pose_w[:, 0:3], root_pose_w[:, 3:7],
        #     current_target_pose[:3].unsqueeze(0), current_target_pose[3:7].unsqueeze(0)
        # )
        # target_pose_w = torch.cat([target_pos_w, target_quat_w], dim=-1)
        # goal_marker.visualize(target_pose_w[:, 0:3], target_pose_w[:, 3:7])


        # Debug: Print current state
        if count % 100 == 0:  # Print every 100 steps
            state_names = ["Approach", "Pick", "Lift", "Move", "Place", "Return"]
            print(f"State: {state_names[current_state]}, Gripper: {'Closed' if gripper_closed else 'Open'}")
        
        # 스텝/업데이트
        sim.step(render=True)
        robot.update(sim.get_physics_dt())
        scene.update(sim.get_physics_dt())
        count += 1

# Update robot states
def update_states(
    sim: sim_utils.SimulationContext,
    scene: InteractiveScene,
    robot: Articulation,
    ee_frame_idx: int,
    arm_joint_ids: list[int],
    ee_offset: torch.Tensor,
):
    """Update the robot states.

    Args:
        sim: (SimulationContext) Simulation context.
        scene: (InteractiveScene) Interactive scene.
        robot: (Articulation) Robot articulation.
        ee_frame_idx: (int) End-effector frame index.
        arm_joint_ids: (list[int]) Arm joint indices.
        ee_offset: (torch.Tensor) Manual offset for end-effector position.

    Returns:
        jacobian_b (torch.tensor): Jacobian in the body frame.
        mass_matrix (torch.tensor): Mass matrix.
        gravity (torch.tensor): Gravity vector.
        ee_pose_b (torch.tensor): End-effector pose in the body frame.
        ee_vel_b (torch.tensor): End-effector velocity in the body frame.
        root_pose_w (torch.tensor): Root pose in the world frame.
        ee_pose_w (torch.tensor): End-effector pose in the world frame.
        joint_pos (torch.tensor): The joint positions.
        joint_vel (torch.tensor): The joint velocities.
    """
    # obtain dynamics related quantities from simulation
    ee_jacobi_idx = ee_frame_idx - 1
    jacobian_w = robot.root_physx_view.get_jacobians()[:, ee_jacobi_idx, :, arm_joint_ids]
    mass_matrix = robot.root_physx_view.get_generalized_mass_matrices()[:, arm_joint_ids, :][:, :, arm_joint_ids]
    gravity = robot.root_physx_view.get_gravity_compensation_forces()[:, arm_joint_ids]
    # Convert the Jacobian from world to root frame
    jacobian_b = jacobian_w.clone()
    root_rot_matrix = matrix_from_quat(quat_inv(robot.data.root_quat_w))
    jacobian_b[:, :3, :] = torch.bmm(root_rot_matrix, jacobian_b[:, :3, :])
    jacobian_b[:, 3:, :] = torch.bmm(root_rot_matrix, jacobian_b[:, 3:, :])

    # Compute current pose of the end-effector
    root_pos_w = robot.data.root_pos_w
    root_quat_w = robot.data.root_quat_w
    ee_pos_w = robot.data.body_pos_w[:, ee_frame_idx]
    ee_quat_w = robot.data.body_quat_w[:, ee_frame_idx]
    
    # Apply manual offset to end-effector position
    ee_pos_w = ee_pos_w + ee_offset.unsqueeze(0).repeat(scene.num_envs, 1)
    
    ee_pos_b, ee_quat_b = subtract_frame_transforms(root_pos_w, root_quat_w, ee_pos_w, ee_quat_w)
    root_pose_w = torch.cat([root_pos_w, root_quat_w], dim=-1)
    ee_pose_w = torch.cat([ee_pos_w, ee_quat_w], dim=-1)
    ee_pose_b = torch.cat([ee_pos_b, ee_quat_b], dim=-1)

    # Compute the current velocity of the end-effector
    ee_vel_w = robot.data.body_vel_w[:, ee_frame_idx, :]  # Extract end-effector velocity in the world frame
    root_vel_w = robot.data.root_vel_w  # Extract root velocity in the world frame
    relative_vel_w = ee_vel_w - root_vel_w  # Compute the relative velocity in the world frame
    ee_lin_vel_b = quat_apply_inverse(robot.data.root_quat_w, relative_vel_w[:, 0:3])  # From world to root frame
    ee_ang_vel_b = quat_apply_inverse(robot.data.root_quat_w, relative_vel_w[:, 3:6])
    ee_vel_b = torch.cat([ee_lin_vel_b, ee_ang_vel_b], dim=-1)

    # Get joint positions and velocities
    joint_pos = robot.data.joint_pos[:, arm_joint_ids]
    joint_vel = robot.data.joint_vel[:, arm_joint_ids]

    return (
        jacobian_b,
        mass_matrix,
        gravity,
        ee_pose_b,
        ee_vel_b,
        root_pose_w,
        ee_pose_w,
        joint_pos,
        joint_vel,
    )


# Update the target commands
def update_target(
    sim: sim_utils.SimulationContext,
    scene: InteractiveScene,
    osc: OperationalSpaceController,
    root_pose_w: torch.tensor,
    target_pose: torch.tensor,
):
    """Update the targets for the operational space controller.

    Args:
        sim: (SimulationContext) Simulation context.
        scene: (InteractiveScene) Interactive scene.
        osc: (OperationalSpaceController) Operational space controller.
        root_pose_w: (torch.tensor) Root pose in the world frame.
        ee_target_set: (torch.tensor) End-effector target set.
        current_goal_idx: (int) Current goal index.

    Returns:
        command (torch.tensor): Updated target command.
        ee_target_pose_b (torch.tensor): Updated target pose in the body frame.
        ee_target_pose_w (torch.tensor): Updated target pose in the world frame.
        next_goal_idx (int): Next goal index.
    """

    # update the ee desired command
    command = torch.zeros(scene.num_envs, osc.action_dim, device=sim.device)
    # Set pose (first 7 elements)
    command[:, :7] = target_pose
    # Set Kp gains (last 6 elements) - using lower values for smoother motion
    command[:, 7:] = torch.tensor([260.0, 260.0, 260.0, 260.0, 260.0, 260.0], device=sim.device)
    # update the ee desired pose
    ee_target_pose_b = torch.zeros(scene.num_envs, 7, device=sim.device)
    for target_type in osc.cfg.target_types:
        if target_type == "pose_abs":
            ee_target_pose_b[:] = command[:, :7]
        else:
            raise ValueError("Undefined target_type within update_target().")

    # update the target desired pose in world frame (for marker)
    ee_target_pos_w, ee_target_quat_w = combine_frame_transforms(
        root_pose_w[:, 0:3], root_pose_w[:, 3:7], ee_target_pose_b[:, 0:3], ee_target_pose_b[:, 3:7]
    )
    ee_target_pose_w = torch.cat([ee_target_pos_w, ee_target_quat_w], dim=-1)

    return command, ee_target_pose_b, ee_target_pose_w


# Convert the target commands to the task frame
def convert_to_task_frame(osc: OperationalSpaceController, command: torch.tensor, ee_target_pose_b: torch.tensor):
    """Converts the target commands to the task frame.

    Args:
        osc: OperationalSpaceController object.
        command: Command to be converted.
        ee_target_pose_b: Target pose in the body frame.

    Returns:
        command (torch.tensor): Target command in the task frame.
        task_frame_pose_b (torch.tensor): Target pose in the task frame.
    """
    command = command.clone()
    task_frame_pose_b = ee_target_pose_b.clone()

    cmd_idx = 0
    for target_type in osc.cfg.target_types:
        if target_type == "pose_abs":
            command[:, :3], command[:, 3:7] = subtract_frame_transforms(
                task_frame_pose_b[:, :3], task_frame_pose_b[:, 3:], command[:, :3], command[:, 3:7]
            )
            cmd_idx += 7
        else:
            raise ValueError("Undefined target_type within _convert_to_task_frame().")

    return command, task_frame_pose_b


def main():
    """Main function."""
    # Load kit helper
    sim_cfg = sim_utils.SimulationCfg(dt=0.01, device=args_cli.device)
    sim = sim_utils.SimulationContext(sim_cfg)
    # Set main camera - adjusted to see the table
    sim.set_camera_view([1.5, 1.5, 1.5], [0.0, 0.0, 0.5])  # Look at table center
    # Design scene
    scene_cfg = SceneCfg(num_envs=args_cli.num_envs, env_spacing=2.0)
    scene = InteractiveScene(scene_cfg)
    # Play the simulator
    sim.reset()
    # Now we are ready!
    # Run the simulator
    run_simulator(sim, scene)


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
