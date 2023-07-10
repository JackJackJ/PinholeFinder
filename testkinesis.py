"""An example that uses the .NET Kinesis Libraries to connect to a KDC."""
import os
import time
import sys
import clr

# Add References to .NET libraries
clr.AddReference("C:\\Program Files\\Thorlabs\\Kinesis\\Thorlabs.MotionControl.DeviceManagerCLI.dll.")
clr.AddReference("C:\\Program Files\\Thorlabs\\Kinesis\\Thorlabs.MotionControl.GenericMotorCLI.dll.")
clr.AddReference("C:\\Program Files\\Thorlabs\\Kinesis\\Thorlabs.MotionControl.KCube.BrushlessMotorCLI.dll.")


from Thorlabs.MotionControl.DeviceManagerCLI import *
from Thorlabs.MotionControl.GenericMotorCLI import *
from Thorlabs.MotionControl.KCube.BrushlessMotorCLI import *
from System import Decimal

print(DeviceManagerCLI.BuildDeviceList())
        # create new device
serial_no = str("27502723")  # Replace this line with your device's serial number
kcube = KCubeBrushlessMotor.CreateKCubeBrushlessMotor(serial_no)

        # Connect, begin polling, and enable
kcube.Connect(serial_no)