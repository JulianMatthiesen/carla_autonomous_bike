#!/usr/bin/env python3

import datetime
import glob
import math
import os
import sys
import time
try:
    sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass
import gym
import numpy as np
from gym import spaces
import carla
import random


class BikeEnv(gym.Env):
    """Custom Environment that follows gym interface."""
    
    XMAX=-10
    XMIN=-70
    YMAX=-5
    YMIN=-130
    DISCOUNT = 0.99

    metadata = {"render_modes": ["human"], "render_fps": 30}

    def __init__(self):
        super(BikeEnv, self).__init__()
        # Define action and observation space

        high = np.array([
            1.0,   # throttle bike
            1.0    # steer bike
        ])

        low = np.array([
            -1.0,   # throttle bike
            -1.0   # steer bike
        ])

        self.action_space = spaces.Box(low=low, high=high, shape=(2,), dtype=np.float32)

        image_width = 32
        image_height = 32

        # neural Networks prefer Inputs between 0 and 1 or -1 and 1 or sth like that -> sentdex
        # 0 ist tatsächlich nicht der niedrigste Wert, sondern 1/255
        self.observation_space = spaces.Box(low=0, high=1, shape=(image_height, image_width), dtype=np.float32)



        self.client = carla.Client('localhost', 2000)
        self.client.set_timeout(20.0)
        self.world = self.client.load_world('Town03_Opt', carla.MapLayer.Buildings | carla.MapLayer.ParkedVehicles)
        time.sleep(1)
        self.bp_lib = self.world.get_blueprint_library()

        self.sensor_data = None
        self.bike, self.depth_sensor, self.collision_sensor = self.spawn_bike()

        # synchronous mode und Fixed time-step später wichtig für synchrone Sensoren
        settings = self.world.get_settings()
        settings.synchronous_mode = True  
        settings.fixed_delta_seconds = 0.05
        self.world.apply_settings(settings)

        #self.client.reload_world(False)

        #position spectator
        spectator = self.world.get_spectator()
        transform = carla.Transform(self.bike.get_transform().transform(carla.Location(x=-4, z=2)), self.bike.get_transform().rotation) 
        spectator.set_transform(transform)

        self.front_camera=None
        self.bike_location = self.bike.get_transform().location
        self.target_location = self.set_new_target()
        self.done = False
        self.reward = 0
        self.tick_count = 0
        self.max_time_steps = 4000
        self.world.tick()
        
        self.info = {"actions": []}

    def step(self, action):
        throttle = float((action[0] + 1)/ 2)
        steer=float(action[1])
        self.bike.apply_control(carla.VehicleControl(throttle=throttle, steer=steer))

        # update bike_location
        self.bike_location = self.bike.get_transform().location
        
        self.info["actions"].append(action.tolist())
        observation = self.get_observation()
        self.reward = self.calculate_reward()
        return observation, self.reward, self.done, self.info

    def reset(self):
        if not len(self.world.get_actors()) == 0:
            self.bike.destroy()
            self.depth_sensor.destroy()
            self.collision_sensor.destroy()


        self.bike, self.depth_sensor, self.collision_sensor = self.spawn_bike()
        
        while self.front_camera is None: 
            time.sleep(0.01) # warten bis die front camera das erste Bild liefert

      
        self.done = False
        self.reward = 0
        self.tick_count = 0
        self.info = {"actions": []}
        self.world.tick()
        self.tick_count += 1
        return self.get_observation() #info

    def close(self):
        self.bike.destroy()
        settings = self.world.get_settings()
        settings.synchronous_mode = False
        self.world.apply_settings(settings)
         
        self.world.tick()

    """
    def render(self):
        ...
    """
# ================== Hilfsmethoden ==================

    def spawn_bike(self):
        # spawn bike
        bike_bp = self.bp_lib.find("vehicle.diamondback.century")
        spawn_point = carla.Transform(carla.Location(x=-25.661824, y=-55.620903, z=0.1), carla.Rotation(yaw=0))
        bike = self.world.spawn_actor(bike_bp, spawn_point)
        self.bike_location = spawn_point.location

        # spawn depth sensor
        depth_sensor_bp = self.bp_lib.find('sensor.camera.depth') 
        depth_sensor_bp.set_attribute("fov", "130") 
        depth_sensor_bp.set_attribute("image_size_x", "32")
        depth_sensor_bp.set_attribute("image_size_y", "32")
        depth_camera_init_trans = carla.Transform(carla.Location(x=0.5, z=0.95))
        depth_sensor = self.world.spawn_actor(depth_sensor_bp, depth_camera_init_trans, attach_to=bike)
        
        # initialize sensor_data
        image_w = depth_sensor_bp.get_attribute("image_size_x").as_int()
        image_h = depth_sensor_bp.get_attribute("image_size_y").as_int()
        self.sensor_data = {'depth_image': np.zeros((image_h, image_w, 4)),
                            'collision': False}
        
        depth_sensor.listen(lambda image: self.depth_callback(image, self.sensor_data))

        # spawn collision sensor
        collision_sensor = self.world.spawn_actor(
            self.world.get_blueprint_library().find('sensor.other.collision'),
            carla.Transform(), attach_to=bike)
        collision_sensor.listen(lambda event: self.collision_callback(event, self.sensor_data))

        return bike, depth_sensor, collision_sensor
    
    def depth_callback(self, image, data_dict):
        image.convert(carla.ColorConverter.LogarithmicDepth)
        data_dict["depth_image"] = np.reshape(np.copy(image.raw_data), (image.height, image.width, 4))
        self.front_camera = data_dict["depth_image"]

    def collision_callback(event, data_dict):
        data_dict["collision"] = True

    def get_random_spawn_point(self):
        # spawn vehicle at random location within square
        xSpawn = random.uniform(self.XMIN, self.XMAX)
        ySpawn = random.uniform(self.YMIN, self.YMAX)
        location = carla.Location(x=xSpawn, y=ySpawn, z=0.05)
        phiSpawn = random.uniform(-180, 180)
        rotation = carla.Rotation(pitch=0.0, yaw=phiSpawn, roll=0.0)
        random_point = carla.Transform(location, rotation)
        return random_point
    
    def get_observation(self):
        # normiert auf Werte zw. 0 und 1
        # Observation besteht für jeden Pixel nur aus dem ersten Wert, da sowieso alle rgb Werte gleich sind
        observation = np.array(self.front_camera[:, :, 0], dtype=np.float32) / 255 
        return observation

    def get_distance_to_target(self):
        return self.bike_location.distance(self.target_location)
    
    def set_new_target(self):
        # set new target at random location within square

        # target wird in der mitte gesetzt, um leichter erreichbar 
        # zu sein, ohne das Viereck zu verlassen
        xTarget = random.uniform(self.XMIN + 2, self.XMAX - 2)
        yTarget = random.uniform(self.YMIN + 2, self.YMAX - 2)
        return carla.Location(x=xTarget, y=yTarget, z=0.0)
        
    def calculate_reward(self):
        # reward in a range from -0.996 to 0.996 (depending on how dark/bright the depth image is)
        depth_reward = (np.mean(self.front_camera[:, :, 0]) - 128) / 255 * 2
                        
        # penalty for speeds below 5kmh
        v = self.bike.get_velocity()
        kmh = int(3.6 * math.sqrt(v.x**2 + v.y**2 + v.z**2))
        speed_penalty = -1 if kmh < 5 else 0

        reward = (depth_reward + speed_penalty) 

        # negative reward and stop episode, when leaving the square or colliding
        if not self.is_within_boundary() or self.sensor_data["collision"] == True:
            self.done = True
            reward = -100

        # end episode and negative reward for taking too long (rausgenommen)
        self.world.tick()
        self.tick_count += 1
        
        """
        if self.tick_count >= self.max_time_steps:
            self.done = True
            reward = -100
        """
        return reward
    
    def is_within_boundary(self):
        return self.XMIN <= self.bike_location.x <= self.XMAX and self.YMIN <= self.bike_location.y <= self.YMAX

    

