from pylablib.devices import Thorlabs
print(Thorlabs.list_kinesis_devices())
print(Thorlabs.list_cameras_tlcam())
#stage = Thorlabs.KinesisMotor("27502629")