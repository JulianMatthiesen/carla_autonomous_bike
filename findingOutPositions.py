import random
import carla
import time
import numpy as np
import cv2

client = carla.Client('localhost', 2000)
client.set_timeout(5.0)

world = client.get_world()
spectator = world.get_spectator()
spectator_transform = spectator.get_transform()
spectator_position = spectator_transform.location
print(spectator_position)
print()