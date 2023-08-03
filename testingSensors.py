import random
import carla
import time
import numpy as np
import cv2

client = carla.Client('localhost', 2000)
client.set_timeout(5.0)
world = client.get_world()
time.sleep(1)
#world.unload_map_layer(carla.MapLayer.All)

bp_lib = world.get_blueprint_library()
spawn_points = world.get_map().get_spawn_points()

#alle vorhandenen Fahrräder löschen
actor_list = world.get_actors()
if actor_list is not None:
    for actor in actor_list:
        if 'vehicle' in actor.type_id:
            actor.destroy()


#spawn vehicle
vehicle_bp = bp_lib.find("vehicle.diamondback.century")
transform = carla.Transform(carla.Location(x=-8.75, y=-131, z=2), carla.Rotation())
vehicle = world.try_spawn_actor(vehicle_bp, random.choice(spawn_points))



#position spectator
spectator = world.get_spectator()
transform = carla.Transform(vehicle.get_transform().transform(carla.Location(x=-4,z=2.5)),vehicle.get_transform().rotation) 
spectator.set_transform(transform)

vehicle.apply_control(carla.VehicleControl(throttle=0.2))

camera_bp = bp_lib.find('sensor.camera.depth') 
camera_bp.set_attribute("fov", "130")
depth_camera_1_init_trans = carla.Transform(carla.Location(x=0.5, z=0.95))
depth_camera_2_init_trans = carla.Transform(carla.Location(x=0.5, y=0.2, z=0.95), carla.Rotation(yaw=90))
depth_camera_3_init_trans = carla.Transform(carla.Location(x=0.5, y=-0.2, z=0.95), carla.Rotation(yaw=-90))

depth_camera_1 = world.spawn_actor(camera_bp, depth_camera_1_init_trans, attach_to = vehicle)
depth_camera_2 = world.spawn_actor(camera_bp, depth_camera_2_init_trans, attach_to = vehicle)
depth_camera_3 = world.spawn_actor(camera_bp, depth_camera_3_init_trans, attach_to = vehicle)



# Callback stores camera data in a dictionary for use outside callback              
# 
# das flattened Array raw_data wird zu einem 3D-Array mit den Abmessungen (image.height, image.width, 4)          
# bspw. wird image.raw_data = [r1, g1, b1, a1, r2, g2, b2, a2, r3, g3, b3, a3, r4, g4, b4, a4, r5, ...]
# zu reshaped_array = [
#    [[r1, g1, b1, a1], [r2, g2, b2, a2], [r3, g3, b3, a3], [r4, g4, b4, a4]],
#    [[r5, g5, b5, a5], [r6, g6, b6, a6], [r7, g7, b7, a7], [r8, g8, b8, a8]],
#    [[r9, g9, b9, a9], [r10, g10, b10, a10], [r11, g11, b11, a11], [r12, g12, b12, a12]]
#]

def depth_callback_1(image, data_dict):
    image.convert(carla.ColorConverter.LogarithmicDepth)
    data_dict["depth_image_1"] = np.reshape(np.copy(image.raw_data), (image.height, image.width, 4)) 

def depth_callback_2(image, data_dict):
    image.convert(carla.ColorConverter.LogarithmicDepth)
    data_dict["depth_image_2"] = np.reshape(np.copy(image.raw_data), (image.height, image.width, 4))

def depth_callback_3(image, data_dict):
    image.convert(carla.ColorConverter.LogarithmicDepth)
    data_dict["depth_image_3"] = np.reshape(np.copy(image.raw_data), (image.height, image.width, 4))

# Get camera dimensions and initialise dictionary                       
image_w = camera_bp.get_attribute("image_size_x").as_int()
image_h = camera_bp.get_attribute("image_size_y").as_int()
camera_data = {'depth_image_1': np.zeros((image_h, image_w, 4)),
                'depth_image_2': np.zeros((image_h, image_w, 4)),
                'depth_image_3': np.zeros((image_h, image_w, 4))}

# für observation space
print(image_h)
print(image_w)

# Start camera recording
depth_camera_1.listen(lambda image: depth_callback_1(image, camera_data))
depth_camera_2.listen(lambda image: depth_callback_2(image, camera_data))
depth_camera_3.listen(lambda image: depth_callback_3(image, camera_data))

cv2.namedWindow('Depth Camera', cv2.WINDOW_AUTOSIZE)
top_row = np.concatenate((camera_data['depth_image_3'], camera_data['depth_image_1'], camera_data['depth_image_2']), axis=1)

cv2.imshow('Depth Camera', top_row)
cv2.waitKey(1)

# Game loop
while True:
    top_row = np.concatenate((camera_data['depth_image_3'], camera_data['depth_image_1'], camera_data['depth_image_2']), axis=1)

    # Imshow renders camera data to display
    cv2.imshow('Depth Camera', top_row)
    
    # Quit if user presses 'q'
    if cv2.waitKey(1) == ord('q'):
        break

# Close OpenCV window when finished
depth_camera_1.stop()
depth_camera_2.stop()
depth_camera_3.stop()

depth_camera_1.destroy()
depth_camera_2.destroy()
depth_camera_3.destroy()

cv2.destroyAllWindows()
