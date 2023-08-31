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

        # The minimal resolution for an image is 36x36 for the default `CnnPolicy`
        # -> otherwise custom features extractor
        image_width = 36
        image_height = 36
        num_channels_depth = 4

        # neural Networks prefer Inputs between 0 and 1 or -1 and 1 or sth like that -> sentdex
        # CNN policy normalizes the observation automatically
        self.observation_space = spaces.Box(low=0, high=255, shape=(image_height, image_width, num_channels_depth), dtype=np.uint8)


        self.client = carla.Client('localhost', 2000)
        self.client.set_timeout(20.0)
        self.world = self.client.load_world('Town03_Opt', carla.MapLayer.Buildings | carla.MapLayer.ParkedVehicles)
        time.sleep(1)
        self.bp_lib = self.world.get_blueprint_library()

        self.sensor_data = {}
        self.bike, self.depth_camera, self.collision_sensor = self.spawn_bike()

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

        self.front_camera_depth=None
        self.done = False
        self.reward = 0
        self.tick_count = 0
        self.max_time_steps = 1000
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
            self.depth_camera.destroy()
            self.collision_sensor.destroy()


        self.bike, self.depth_camera, self.collision_sensor = self.spawn_bike()
        
        while self.front_camera_depth is None: 
            time.sleep(0.01) # warten bis die Kameras das erste Bild liefern

      
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
        spawn_point = self.get_random_spawn_point()
        bike = self.world.spawn_actor(bike_bp, spawn_point)
        self.bike_location = spawn_point.location

        camera_init_trans = carla.Transform(carla.Location(x=0.5, z=0.95))

        # spawn depth sensor
        depth_camera_bp = self.bp_lib.find('sensor.camera.depth') 
        depth_camera_bp.set_attribute("fov", "130") 
        depth_camera_bp.set_attribute("image_size_x", "36")
        depth_camera_bp.set_attribute("image_size_y", "36")
        depth_camera = self.world.spawn_actor(depth_camera_bp, camera_init_trans, attach_to=bike)
        

        # initialize sensor_data
        image_w_depth = depth_camera_bp.get_attribute("image_size_x").as_int()
        image_h_depth = depth_camera_bp.get_attribute("image_size_y").as_int()
        
        self.sensor_data = {"depth_image": np.zeros((image_h_depth, image_w_depth, 4)),
                            "collision": False}
        
        depth_camera.listen(lambda image: self.depth_callback(image, self.sensor_data))

        # spawn collision sensor
        collision_sensor = self.world.spawn_actor(self.world.get_blueprint_library().find('sensor.other.collision'), carla.Transform(), attach_to=bike)
        collision_sensor.listen(lambda event: self.collision_callback(event))

        return bike, depth_camera, collision_sensor
    

    def depth_callback(self, image, data_dict):
        image.convert(carla.ColorConverter.LogarithmicDepth)
        data_dict["depth_image"] = np.reshape(np.copy(image.raw_data), (image.height, image.width, 4))
        self.front_camera_depth = data_dict["depth_image"]

    def collision_callback(self, event):
        self.sensor_data["collision"] = True


    def get_random_spawn_point(self):
        # spawn vehicle at random location 
        spawn_place = random.randint(1,4)
        if spawn_place == 1:
            xSpawn = random.uniform(-27, -16)
            ySpawn = random.uniform(-122, -54)
        elif spawn_place == 2:
            xSpawn = random.uniform(-64, -16)
            ySpawn = random.uniform(-125, -120)
        elif spawn_place == 3:
            xSpawn = random.uniform(-63, -16)
            ySpawn = random.uniform(-72, -65)
        elif spawn_place == 4:
            xSpawn = random.uniform(-65, -53)
            ySpawn = random.uniform(-97, -87)
        location = carla.Location(x=xSpawn, y=ySpawn, z=0.05)
        phiSpawn = random.uniform(-180, 180)
        rotation = carla.Rotation(pitch=0.0, yaw=phiSpawn, roll=0.0)
        random_point = carla.Transform(location, rotation)
        return random_point
    
    def get_observation(self):
        
        observation = self.front_camera_depth
        observation = np.array(observation)
        return observation
        
    def calculate_reward(self):
        # reward in a range from ~(-1) to ~1 (depending on how dark/bright the depth image is)
        # so the agent wont maximize reward by driving in a circle
        depth_reward = ((np.mean(self.front_camera_depth[:, :, 0])) - 127) / 255 * 2 
                        
        # penalty for speeds below 5kmh and above 15kmh
        v = self.bike.get_velocity()
        kmh = int(3.6 * math.sqrt(v.x**2 + v.y**2 + v.z**2))
        speed_penalty = -1 if kmh < 5 or kmh > 15 else 0

        reward = (depth_reward + speed_penalty) 

        # negative reward and stop episode, when leaving the square or colliding
        if not self.is_within_boundary() or self.sensor_data["collision"] == True:
            self.done = True
            reward = -100

        # end episode and negative reward for taking too long (rausgenommen)
        self.world.tick()
        self.tick_count += 1
        
        
        if self.tick_count >= self.max_time_steps:
            self.done = True
            #reward = -100
        
        return reward
    
    def is_within_boundary(self):
        return self.XMIN <= self.bike_location.x <= self.XMAX and self.YMIN <= self.bike_location.y <= self.YMAX

