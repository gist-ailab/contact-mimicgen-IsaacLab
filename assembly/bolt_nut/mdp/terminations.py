# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Common functions that can be used to activate certain terminations for the lift task.

The functions can be passed to the :class:`isaaclab.managers.TerminationTermCfg` object to enable
the termination introduced by the function.
"""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

import isaacsim.core.utils.torch as torch_utils

from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def cubes_stacked(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    socket_cfg: SceneEntityCfg = SceneEntityCfg("socket"),
    plug_cfg: SceneEntityCfg = SceneEntityCfg("plug"),
    # cube_3_cfg: SceneEntityCfg = SceneEntityCfg("cube_3"),
    xy_threshold: float = 0.05,
    height_threshold: float = 0.005,
    height_diff: float = 0.0468,
    gripper_open_val: torch.tensor = torch.tensor([0.04]),
    atol=0.0001,
    rtol=0.0001,
):
    robot: Articulation = env.scene[robot_cfg.name]
    socket: RigidObject = env.scene[socket_cfg.name]
    plug: RigidObject = env.scene[plug_cfg.name]
    # cube_3: RigidObject = env.scene[cube_3_cfg.name]

    pos_diff_c12 = socket.data.root_pos_w - plug.data.root_pos_w
    # pos_diff_c23 = cube_2.data.root_pos_w - cube_3.data.root_pos_w

    # Compute cube position difference in x-y plane
    xy_dist_c12 = torch.norm(pos_diff_c12[:, :2], dim=1)
    # xy_dist_c23 = torch.norm(pos_diff_c23[:, :2], dim=1)

    # Compute cube height difference
    h_dist_c12 = torch.norm(pos_diff_c12[:, 2:], dim=1)
    # h_dist_c23 = torch.norm(pos_diff_c23[:, 2:], dim=1)

    # Check cube positions
    # stacked = torch.logical_and(xy_dist_c12 < xy_threshold)
    stacked = xy_dist_c12 < xy_threshold
    stacked = torch.logical_and(h_dist_c12 - height_diff < height_threshold, stacked)
    # stacked = torch.logical_and(h_dist_c23 - height_diff < height_threshold, stacked)

    # Check gripper positions
    stacked = torch.logical_and(
        torch.isclose(robot.data.joint_pos[:, -1], gripper_open_val.to(env.device), atol=atol, rtol=rtol), stacked
    )
    stacked = torch.logical_and(
        torch.isclose(robot.data.joint_pos[:, -2], gripper_open_val.to(env.device), atol=atol, rtol=rtol), stacked
    )
    return stacked



def boltnut_assembled(
        env: ManagerBasedRLEnv,
        robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
        socket_cfg: SceneEntityCfg = SceneEntityCfg("socket"),
        plug_cfg: SceneEntityCfg = SceneEntityCfg("plug"),
        xy_threshold: float = 0.05,
        height_threshold: float = 0.005,
        height_diff: float = 0.0468,
        gripper_open_val: torch.tensor = torch.tensor([0.04]),
        atol=0.0001,
        rtol=0.0001,
):
    socket: RigidObject = env.scene[socket_cfg.name]
    plug: RigidObject = env.scene[plug_cfg.name]

    identity_quat = torch.tensor([1.0, 0.0, 0.0, 0.0], device=env.device).repeat((env.num_envs, 1))
    offsets = _get_keypoint_offsets(env, 4)
    keypoint_offsets = offsets * 0.15
    keypoints_plug = torch.zeros((env.num_envs, 4, 3), device=env.device)
    keypoints_socket = torch.zeros_like(keypoints_plug, device=env.device)
    
    # get socket, plug position and orientation in world frame
    socket_pos = socket.data.root_pos_w
    plug_pos = plug.data.root_pos_w

    socket_quat = socket.data.root_quat_w
    plug_quat = plug.data.root_quat_w

    # get socket, plug local pos and quat
    plug_base_x_offset = 0.0
    plug_base_z_offset = 0.0
    plug_base_pos_local = torch.tensor([0.0, 0.0, 0.0], device=env.device).repeat((env.num_envs, 1))
    plug_base_pos_local[:, 0] = plug_base_x_offset
    plug_base_pos_local[:, 2] = plug_base_z_offset
    plug_base_quat_local = identity_quat.clone().detach()

    socket_success_pos_local = torch.zeros((env.num_envs, 3), device=env.device)
    socket_success_pos_local[:, 2] = 0.0

    # get keypoint of socket and plug
    plug_base_quat, plug_base_pos = torch_utils.tf_combine(
        plug_quat, plug_pos, plug_base_quat_local, plug_base_pos_local
    )

    socket_base_quat, socket_base_pos = torch_utils.tf_combine(
        socket_quat, socket_pos, identity_quat, socket_success_pos_local
    )

    for idx, keypoint_offset in enumerate(keypoint_offsets):
        keypoints_plug[:, idx] = torch_utils.tf_combine(
            plug_base_quat, plug_base_pos, identity_quat, keypoint_offset.repeat(env.num_envs, 1)
        )[1]
        keypoints_socket[:, idx] = torch_utils.tf_combine(
            socket_base_quat, socket_base_pos, identity_quat, keypoint_offset.repeat(env.num_envs, 1)
        )[1]

    is_assembled = _check_plug_inserted_in_socket(
        env=env,
        plug_pos=plug_pos,
        socket_pos=socket_pos,
        disassembly_dist=0.05,
        keypoints_plug=keypoints_plug,
        keypoints_socket=keypoints_socket,
        close_error_thresh=0.005,
    )

    return is_assembled

def _get_keypoint_offsets(env, num_keypoints):
    """Get uniformly-spaced keypoints along a line of unit length, centered at 0."""
    keypoint_offsets = torch.zeros((num_keypoints, 3), device=env.device)
    keypoint_offsets[:, -1] = torch.linspace(0.0, 1.0, num_keypoints, device=env.device) - 0.5

    return keypoint_offsets

def _check_plug_inserted_in_socket(
        env, plug_pos, socket_pos, disassembly_dist, keypoints_plug, keypoints_socket, close_error_thresh
):
    """Check if plug is inserted in socket."""
    is_plug_below_insertion_height = plug_pos[:, 2] < socket_pos[:, 2] + disassembly_dist
    is_plug_above_table_height = plug_pos[:, 2] > socket_pos[:, 2] + 0.02
    
    is_plug_height_success = torch.logical_and(is_plug_below_insertion_height, is_plug_above_table_height)
    # check is plug close to socket
    # compute if keypiont distance is below threshold
    keypoint_dist = torch.norm(keypoints_socket - keypoints_plug, p=2, dim=-1)
    # check if keypoint dist is below threshold
    is_plug_close_to_socket = torch.where(torch.mean(keypoint_dist, dim=-1) < close_error_thresh,
                                          torch.tensor([True], device=env.device),
                                          torch.tensor([False], device=env.device))
    
    # return torch.logical_and(is_plug_height_success, is_plug_close_to_socket)
    return is_plug_close_to_socket
    