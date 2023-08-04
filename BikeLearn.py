import gym
from stable_baselines3 import  PPO
import os
from SensorsBikeEnv import BikeEnv
import time


models_dir = f"models/{int(time.time())}"
logdir = f"logs/{int(time.time())}"

if not os.path.exists(models_dir):
    os.makedirs(models_dir)

if not os.path.exists(logdir):
    os.makedirs(logdir)

env = BikeEnv()
obs = env.reset()


model = PPO("MlpPolicy", env, verbose=1, tensorboard_log=logdir)

TIMESTEPS=10000

#check model performance:


for i in range(1, 100):
    model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False, tb_log_name="PPO")
    model.save(f"{models_dir}/{TIMESTEPS*i}")

env.close()
        

