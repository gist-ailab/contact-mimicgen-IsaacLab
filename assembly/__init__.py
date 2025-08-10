# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Assembly package for Isaac Lab."""

import os
import importlib
import pkgutil

def import_packages(package_name: str, blacklist: list[str] | None = None):
    """Import all sub-packages in the given package.
    
    Args:
        package_name: The name of the package to import sub-packages from.
        blacklist: List of package names to skip during import.
    """
    if blacklist is None:
        blacklist = []
    
    package = importlib.import_module(package_name)
    if package.__file__ is None:
        return
    package_path = os.path.dirname(package.__file__)
    
    for _, name, is_pkg in pkgutil.iter_modules([package_path]):
        if is_pkg and name not in blacklist:
            importlib.import_module(f"{package_name}.{name}")

# Import all sub-packages to register environments
import_packages(__name__) 