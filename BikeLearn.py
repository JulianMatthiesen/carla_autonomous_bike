import gym
from stable_baselines3 import  PPO
import os
from SensorsBikeEnvWithSemSeg import BikeEnv
import time
from gym.wrappers import FlattenObservation


models_dir = f"models/WithoutPositionOrSemSeg"
logdir = f"logs/WithoutPositionOrSemSeg"

if not os.path.exists(models_dir):
    os.makedirs(models_dir)

if not os.path.exists(logdir):
    os.makedirs(logdir)

env = BikeEnv()

obs = env.reset()

#model_path = f"{models_dir}/320000.zip"
#model = PPO.load(model_path, env=env)

model = PPO("MlpPolicy", env, verbose=1, tensorboard_log=logdir)

TIMESTEPS=10000
i = 0


while i < 200:
    i+=1
    model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False, tb_log_name="PPO")
    model.save(f"{models_dir}/{TIMESTEPS*i}")

env.close()
        

