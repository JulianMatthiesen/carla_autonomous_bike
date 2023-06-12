import gym
from stable_baselines3 import  PPO
import os
from newBikeEnv import BikeEnv
import time

models_dir = f"models/{int(time.time())}"
logdir = f"logs/{int(time.time())}"

if not os.path.exists(models_dir):
    os.makedirs(models_dir)

if not os.path.exists(logdir):
    os.makedirs(logdir)


env = BikeEnv()



model = PPO("MlpPolicy", env, verbose=1)

TIMESTEPS=10000
model.learn(total_timesteps=TIMESTEPS)

obs = env.reset()

for _ in range(100):
    action, _ = model.predict(obs)
    obs, reward, done, info = env.step(action)
    if done:
        obs = env.reset()
        env.close()
        

