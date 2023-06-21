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

        high = np.array([
            14, -129,    # position bike(x,y)
            14, -129,    # position target(x,y)
            180,         # rotation bike 
            27.5         # distance 
        ])

        low = np.array([
            -8.75, -144, # position bike(x,y)
            -8.75, -144, # position target(x,y)
            -180,        # rotation bike 
            0            # distance 
        ])

        self.observation_space = spaces.Box(low=low, high=high, shape=(6,), dtype=np.float32)


        self.client = carla.Client('localhost', 2000)
        self.client.set_timeout(20.0)
        self.world = self.client.load_world('Town03_Opt', carla.MapLayer.Buildings | carla.MapLayer.ParkedVehicles)
        time.sleep(1)
        self.world.unload_map_layer(carla.MapLayer.All)
        blueprint_library = self.world.get_blueprint_library()
        
        self.bike_bp = blueprint_library.find("vehicle.diamondback.century")

        # synchronous mode und Fixed time-step später wichtig für synchrone Sensoren
        settings = self.world.get_settings()
        settings.synchronous_mode = True  
        settings.fixed_delta_seconds = 0.05
        self.world.apply_settings(settings)

        #self.client.reload_world(False)

        #position spectator
        self.spectator = self.world.get_spectator()
        self.spectator.set_transform(carla.Transform(carla.Location(x=7.727516, y=-117.762421, z=8.201375), carla.Rotation(pitch=-19.756622, yaw=-100.927879, roll=0.000024)))

        spawn_point = self.get_random_spawn_point()
        self.bike = self.world.spawn_actor(self.bike_bp, spawn_point)
        self.bike_location = spawn_point.location
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
        spawn_point = self.get_random_spawn_point()
        self.bike = self.world.try_spawn_actor(self.bike_bp, spawn_point)
        self.bike_location = spawn_point.location

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
        bike_transform = self.bike.get_transform()
        get_current_location = bike_transform.location
        current_location = [get_current_location.x, get_current_location.y]
        current_rotation = [bike_transform.rotation.yaw]
        target_location = [self.target_location.x, self.target_location.y]
        dist = [self.get_distance_to_target()]
        observation = current_location + target_location + current_rotation + dist
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

        reward = (self.DISCOUNT**self.tick_count) * ((self.prev_distance - current_distance) * 1000 + reward_for_target + time_penalty + speed_penalty) 
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

    

