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
    
    XMAX=14
    XMIN=-8.75
    YMAX=-129
    YMIN=-144
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

        image_width = 800
        image_height = 600

        # neural Networks prefer Inputs between 0 and 1 or -1 and 1 or sth like that -> sentdex
        self.observation_space = spaces.Box(low=0, high=1, shape=(image_height, image_width), dtype=np.float32)

        self.client = carla.Client('localhost', 2000)
        self.client.set_timeout(20.0)
        self.world = self.client.load_world('Town03_Opt', carla.MapLayer.Buildings | carla.MapLayer.ParkedVehicles)
        time.sleep(1)
        self.world.unload_map_layer(carla.MapLayer.All)
        self.bp_lib = self.world.get_blueprint_library()


        self.bike, self.depth_sensor, self.collision_sensor = self.spawn_bike()

        # synchronous mode und Fixed time-step später wichtig für synchrone Sensoren
        settings = self.world.get_settings()
        settings.synchronous_mode = True  
        settings.fixed_delta_seconds = 0.05
        self.world.apply_settings(settings)

        #self.client.reload_world(False)

        #position spectator
        self.spectator = self.world.get_spectator()
        self.spectator.set_transform(carla.Transform(carla.Location(x=7.727516, y=-117.762421, z=8.201375), carla.Rotation(pitch=-19.756622, yaw=-100.927879, roll=0.000024)))

        self.front_camera=None
        self.bike_location = self.bike.get_transform().location
        self.target_location = self.set_new_target()
        self.done = False
        self.reward = 0
        self.tick_count = 0
        self.max_time_steps = 4000
        self.world.tick()
        self.sensor_data = None
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

        # set target at random location within square
        self.target_location = self.set_new_target()
        self.world.debug.draw_string(self.target_location, "X", draw_shadow=False,
                                     color=carla.Color(r=255, g=0, b=0), life_time=2,
                                     persistent_lines=True)
        self.prev_distance = self.get_distance_to_target()
        self.done = False
        self.reward = 0
        print("tick_count: " + str(self.tick_count))
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
        bike_bp = self.bp_lib.find("vehicle.diamondback.century")
        spawn_point = self.get_random_spawn_point()
        bike = self.world.spawn_actor(bike_bp, spawn_point)
        self.bike_location = spawn_point.location

        depth_sensor_bp = self.bp_lib.find('sensor.camera.depth') 
        depth_sensor_bp.set_attribute("fov", "130")
        depth_camera_init_trans = carla.Transform(carla.Location(x=0.5, z=0.95))
        depth_sensor = self.world.spawn_actor(depth_sensor_bp, depth_camera_init_trans, attach_to=bike)
                    
        image_w = depth_sensor_bp.get_attribute("image_size_x").as_int()
        image_h = depth_sensor_bp.get_attribute("image_size_y").as_int()
        self.sensor_data = {'depth_image': np.zeros((image_h, image_w, 4)),
                       'collision': False}
        
        depth_sensor.listen(lambda image: self.depth_callback(image, self.sensor_data))

        collision_sensor = self.world.spawn_actor(
            self.world.get_blueprint_library().find('sensor.other.collision'),
            carla.Transform(), attach_to=bike)
        collision_sensor.listen(lambda event: self.collision_callback(event, self.sensor_data))

        return bike, depth_sensor, collision_sensor
    
    def collision_callback(event, data_dict):
        data_dict["collision"] = True

    def depth_callback(self, image, data_dict):
        image.convert(carla.ColorConverter.LogarithmicDepth)
        data_dict["depth_image"] = np.reshape(np.copy(image.raw_data), (image.height, image.width, 4))
        self.front_camera = data_dict["depth_image"]

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
        observation = np.array(self.front_camera, dtype=np.float32) / 255 #normiert auf Werte zw. 0 und 1
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
        current_distance = self.get_distance_to_target()

        reward_for_target = 0
        time_penalty = -1       #negative reward for every step the target was not reached

        #penalty for speeds below 10kmh
        v = self.bike.get_velocity()
        kmh = int(3.6 * math.sqrt(v.x**2 + v.y**2 + v.z**2))
        speed_penalty = -5 if kmh < 5 else 0

        # if target reached -> reward for finding
        # and calculate new target 
        if current_distance < 1.0:
            self.target_location = self.set_new_target()
            reward_for_target = 100
            time_penalty = 0
            self.tick_count = 0
            self.world.debug.draw_string(self.target_location, "X", draw_shadow=False,
                                        color=carla.Color(r=255, g=0, b=0), life_time=2,
                                        persistent_lines=True)
            print("target reached")

        if self.sensor_data["collision"] == True: 
            self.done = True
            collision_reward = -100

        reward = (self.DISCOUNT**self.tick_count) * (collision_reward + time_penalty + speed_penalty) 
        self.prev_distance = current_distance
        
        # negative reward and stop episode, when leaving the square
        if not self.is_within_boundary():
            self.done = True
            reward = -100

        self.world.tick()
        self.tick_count += 1
        if self.tick_count >= self.max_time_steps:
            self.done = True
            reward = -100
        return reward
    
    def is_within_boundary(self):
        return self.XMIN <= self.bike_location.x <= self.XMAX and self.YMIN <= self.bike_location.y <= self.YMAX

    

