# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Spacemouse controller for SE(3) control."""

import hid
import numpy as np
import threading
import time
from collections.abc import Callable
from scipy.spatial.transform import Rotation

from ..device_base import DeviceBase
from .utils import convert_buffer

import pyspacemouse

class Se3PySpaceMouse(DeviceBase):
    """A space-mouse controller for sending SE(3) commands as delta poses.

    This class implements a space-mouse controller to provide commands to a robotic arm with a gripper.
    It uses the `HID-API`_ which interfaces with USD and Bluetooth HID-class devices across multiple platforms [1].

    The command comprises of two parts:

    * delta pose: a 6D vector of (x, y, z, roll, pitch, yaw) in meters and radians.
    * gripper: a binary command to open or close the gripper.

    Note:
        The interface finds and uses the first supported device connected to the computer.

    Currently tested for following devices:

    - SpaceMouse Compact: https://3dconnexion.com/de/product/spacemouse-compact/

    .. _HID-API: https://github.com/libusb/hidapi

    """

    def __init__(self, pos_sensitivity: float = 0.4, rot_sensitivity: float = 0.8):
        """Initialize the space-mouse layer.

        Args:
            pos_sensitivity: Magnitude of input position command scaling. Defaults to 0.4.
            rot_sensitivity: Magnitude of scale input rotation commands scaling. Defaults to 0.8.
        """
        # store inputs
        self.pos_sensitivity = pos_sensitivity
        self.rot_sensitivity = rot_sensitivity
        # acquire device interface
        # self._device = hid.device()
        try:
            self._device = pyspacemouse.open()
        except Exception as e:
            print(f"Error opening spacemouse: {e}")
            raise

        # self._find_device()
        # read rotations
        self._read_rotation = False

        # command buffers
        self._close_gripper = False
        self._delta_pos = np.zeros(3)  # (x, y, z)
        self._delta_rot = np.zeros(3)  # (roll, pitch, yaw)
        # dictionary for additional callbacks
        self._additional_callbacks = dict()
        
        # button state tracking for debouncing
        self._prev_buttons = [0, 0]  # previous button states
        self._button_debounce_time = 0.1  # debounce time in seconds
        self._last_button_press_time = [0.0, 0.0]  # last press time for each button
        
        # run a thread for listening to device updates
        self._thread = threading.Thread(target=self._run_device)
        self._thread.daemon = True
        self._thread.start()

    def __del__(self):
        """Destructor for the class."""
        self._thread.join()

    def __str__(self) -> str:
        """Returns: A string containing the information of joystick."""
        msg = f"Spacemouse Controller for SE(3): {self.__class__.__name__}\n"
        # msg += f"\tManufacturer: {self._device.get_manufacturer_string()}\n"
        # msg += f"\tProduct: {self._device.get_product_string()}\n"
        msg += "\t----------------------------------------------\n"
        msg += "\tRight button: reset command\n"
        msg += "\tLeft button: toggle gripper command (open/close)\n"
        msg += "\tMove mouse laterally: move arm horizontally in x-y plane\n"
        msg += "\tMove mouse vertically: move arm vertically\n"
        msg += "\tTwist mouse about an axis: rotate arm about a corresponding axis"
        return msg

    """
    Operations
    """

    def reset(self):
        # default flags
        self._close_gripper = False
        self._delta_pos = np.zeros(3)  # (x, y, z)
        self._delta_rot = np.zeros(3)  # (roll, pitch, yaw)

    def add_callback(self, key: str, func: Callable):
        # check keys supported by callback
        if key not in ["L", "R"]:
            raise ValueError(f"Only left (L) and right (R) buttons supported. Provided: {key}.")
        # TODO: Improve this to allow multiple buttons on same key.
        self._additional_callbacks[key] = func

    def advance(self) -> tuple[np.ndarray, bool]:
        """Provides the result from spacemouse event state.

        Returns:
            A tuple containing the delta pose command and gripper commands.
        """
        rot_vec = Rotation.from_euler("XYZ", self._delta_rot).as_rotvec()
        # if new command received, reset event flag to False until keyboard updated.
        return np.concatenate([self._delta_pos, rot_vec]), self._close_gripper

    """
    Internal helpers.
    """


    def _start_timer(self, interval_sec: float = 0.001):
        """This method starts a timer with the given interval to run the callback function for updates."""

        self._timer = threading.Timer(interval_sec, self._run_device)
        self._timer.start()

    def _find_device(self):
        """Find the device connected to computer."""
        found = False
        # implement a timeout for device search
        for _ in range(5):
            for device in hid.enumerate():
                if (
                    device["product_string"] == "SpaceMouse Compact"
                    or device["product_string"] == "SpaceMouse Wireless"
                ):
                    # set found flag
                    found = True
                    vendor_id = device["vendor_id"]
                    product_id = device["product_id"]
                    # connect to the device
                    self._device.close()
                    self._device.open(vendor_id, product_id)
            # check if device found
            if not found:
                time.sleep(1.0)
            else:
                break
        # no device found: return false
        if not found:
            raise OSError("No device found by SpaceMouse. Is the device connected?")

    def _run_device(self):
        """Listener thread that keeps pulling new messages."""
        # keep running
        while True:
            # read the device data
            # data = self._device.read(7)
            # self._start_timer()
            data = self._device.read() # x,y,z,roll,pitch,yaw,button
            # print('data: ', data)
            # print('origin_data: ', self._device.read(14))
            if data is not None:

                
                self._delta_pos[1] = self.pos_sensitivity * data.x 
                self._delta_pos[0] = self.pos_sensitivity * -data.y 
                self._delta_pos[2] = self.pos_sensitivity * data.z
                self._delta_rot[0] = self.rot_sensitivity * -data.roll
                self._delta_rot[1] = self.rot_sensitivity * -data.pitch
                self._delta_rot[2] = self.rot_sensitivity * -data.yaw
                
                # Handle button presses with debouncing
                current_time = time.time()
                
                # Left button (index 0) - gripper toggle
                if data.buttons[0] == 1 and self._prev_buttons[0] == 0:
                    # Button just pressed (rising edge)
                    if current_time - self._last_button_press_time[0] > self._button_debounce_time:
                        # close gripper
                        self._close_gripper = not self._close_gripper
                        # additional callbacks
                        if "L" in self._additional_callbacks:
                            self._additional_callbacks["L"]()
                        self._last_button_press_time[0] = current_time
                
                # Right button (index 1) - reset
                if data.buttons[1] == 1 and self._prev_buttons[1] == 0:
                    # Button just pressed (rising edge)
                    if current_time - self._last_button_press_time[1] > self._button_debounce_time:
                        # reset layer
                        self.reset()
                        # additional callbacks
                        if "R" in self._additional_callbacks:
                            self._additional_callbacks["R"]()
                        self._last_button_press_time[1] = current_time
                
                # Update previous button states
                self._prev_buttons = data.buttons.copy()
            
            # Add a small delay to prevent reading too fast
            time.sleep(0.001)  # 1ms delay