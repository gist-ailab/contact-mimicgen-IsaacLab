# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
import gymnasium as gym
import os

from . import (
    agents,
    boltnut_ik_rel_env_cfg,
    boltnut_osc_env_cfg,
)
gym.register(
    id="Isaac-BoltNut-Franka-IK-Rel-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": boltnut_osc_env_cfg.FrankaBoltNutEnvCfg,
        "robomimic_bc_cfg_entry_point": os.path.join(list(agents.__path__)[0], "robomimic/bc_rnn_low_dim.json"),
    },
    disable_env_checker=True,
)