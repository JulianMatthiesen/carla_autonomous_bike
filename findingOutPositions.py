import random
import carla
import time
import numpy as np
import cv2

client = carla.Client('localhost', 2000)
client.set_timeout(5.0)

world = client.get_world()
spectator = world.get_spectator()
bp_lib = world.get_blueprint_library()

bike_bp = bp_lib.find("vehicle.diamondback.century")
transform = carla.Transform(carla.Location(x=-65, y=-87, z = 2), carla.Rotation()) 
bike = world.spawn_actor(bike_bp, transform)

spectator.set_transform(transform)
spectator_transform = spectator.get_transform()
spectator_position = spectator_transform.location
print(spectator_position)


# x=-27, y=-54
# x=-27, y=-122
# x=-16, y=-122
# x=-16, y=-54