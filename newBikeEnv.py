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

    metadata = {"render_modes": ["human"], "render_fps": 30}

    def __init__(self):
        super(BikeEnv, self).__init__()
        # Define action and observation space

        high = np.array([
            1.0,   # throttle bike
            1.0    # steer bike
        ])

        low = np.array([
            0.0,   # throttle bike
            -1.0   # steer bike
        ])

        self.action_space = spaces.Box(low=low, high=high, shape=(2,), dtype=np.float32)

        high = np.array([
            14, -129,    # position bike(x,y)
            14, -129,    # position target(x,y)
            180          # rotation bike 
        ])

        low = np.array([
            -8.75, -144, # position bike(x,y)
            -8.75, -144, # position target(x,y)
            -180         # rotation bike 
        ])

        self.observation_space = spaces.Box(low=low, high=high, shape=(5,), dtype=np.float32)


        self.client = carla.Client('localhost', 2000)
        self.client.set_timeout(20.0)
        self.world = self.client.load_world('Town03_Opt', carla.MapLayer.Buildings | carla.MapLayer.ParkedVehicles)
        time.sleep(1)
        self.world.unload_map_layer(carla.MapLayer.All)
        blueprint_library = self.world.get_blueprint_library()
        
        self.bike_bp = blueprint_library.find("vehicle.diamondback.century")

        # synchonous mode und Fixed time-step später wichtig für synchrone Sensoren
        settings = self.world.get_settings()
        settings.synchronous_mode = True  
        settings.fixed_delta_seconds = 0.05
        self.world.apply_settings(settings)
        self.client.reload_world(False)

        spawn_point = self.get_random_point()
        self.bike = self.world.spawn_actor(self.bike_bp, spawn_point)
        self.bike_location = spawn_point
        self.target_location = self.set_new_target()

        self.world.tick()

        self.info = {"actions": []}



    def step(self, action):
        throttle = float(action[0])
        steer=float(action[1])
        self.bike.apply_control(carla.VehicleControl(throttle=throttle, steer=steer))

        # update bike_location
        self.bike_location = self.bike.get_transform().location
        distance = self.get_distance_to_target()

        # if target reached -> reward and calculate new target 
        reward_for_target = 0
        if distance < 0.5:
            self.target_location = self.set_new_target()
            reward_for_target = 100

        self.terminated, reward_for_distance = self.get_reward(distance)      
        self.reward = reward_for_distance + reward_for_target

        self.world.tick()
        
        self.info["actions"].append(action.tolist())
        return self.get_observation(), self.reward, self.terminated, self.info

    def reset(self):
        
        spawn_point = self.get_random_point()
        self.bike = self.world.try_spawn_actor(self.bike_bp, spawn_point)
        self.bike_location = spawn_point

        # set target at random location within square
        self.target_location = self.set_new_target()
        
        self.info = {"actions": []}
        self.world.tick()
        return self.get_observation() #info


# ================== Hilfsmethoden ==================

    
    def get_random_point(self):
        # spawn vehicle at random location within square
        xSpawn = random.uniform(self.XMIN, self.XMAX)
        ySpawn = random.uniform(self.YMIN, self.YMAX)
        location = carla.Location(x=xSpawn, y=ySpawn, z=5)
        random_point = carla.Transform(location, carla.Rotation())
        return random_point
    
    def get_reward(self, distance):
        # negative reward and stop episode, when leaving the square
        if not self.XMIN <= self.bike_location.x <= self.XMAX or not self.YMIN <= self.bike_location.y <= self.YMAX:
            terminated = True
            reward = -50
        else:
            terminated = False
            reward = (27.25 - distance) # maximum minus actual distance
        return terminated, reward
    
    def get_observation(self):
        bike_transform = self.bike.get_transform()
        get_current_location = bike_transform.location
        current_location = [get_current_location.x, get_current_location.y]
        current_rotation = [bike_transform.rotation.yaw]
        target_location = [self.target_location.x, self.target_location.y]
        observation = current_location + target_location + current_rotation
        observation = np.array(observation, dtype=np.float32)
        return observation

    def get_distance_to_target(self):
        return self.bike_location.distance(self.target_location)
    
    def set_new_target(self):
        # set new target at random location within square

        # target wird in der mitte gesetzt, um leichter erreichbar 
        # zu sein, ohne das Viereck zu verlassen
        xTarget = random.uniform(self.XMIN + 2, self.XMAX - 2)
        yTarget = random.uniform(self.YMIN + 2, self.YMAX - 2)
        return carla.Location(x=xTarget, y=yTarget, z=2)
        

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
