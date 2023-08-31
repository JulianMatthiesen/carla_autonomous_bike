import math
import random
import carla
import time
import numpy as np
import cv2

client = carla.Client('localhost', 2000)
client.set_timeout(5.0)
#world = client.load_world('Town03_Opt', carla.MapLayer.Buildings | carla.MapLayer.ParkedVehicles)
world = client.get_world()
spectator = world.get_spectator()
# bp_lib = world.get_blueprint_library()

# bike_bp = bp_lib.find("vehicle.diamondback.century")
# transform = world.get_map().get_spawn_points()
# bike = world.try_spawn_actor(bike_bp, transform)

# transform = carla.Transform(bike.get_transform().transform(carla.Location(x=-4, z=2)),bike.get_transform().rotation) 
# spectator.set_transform(transform)
# time.sleep(2)

# bike.apply_control(carla.VehicleControl(throttle=0.05))
spectator_transform = spectator.get_transform()
spectator_position = spectator_transform.location
print("specpos:" + str(spectator_transform))
info = {
    "target_locations": []
}

# Annahme: Sie haben einige Koordinaten zu info["target_locations"] hinzugefügt
# info["target_locations"].append([1, 2])
# info["target_locations"].append([2, 3])
# info["target_locations"].append([3, 3])
# Weitere Koordinaten...



xTarget = random.uniform(-8.75, 14)
yTarget = random.uniform(-144, -129)

target = np.array([int(xTarget), int(yTarget)])

info["target_locations"].append(target.tolist())
for location in info["target_locations"]:
    x, y = location
    print(f"X: {x}, Y: {y}")
