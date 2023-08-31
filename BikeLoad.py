import math
import gym
from stable_baselines3 import PPO   
from SensorsBikeEnv import BikeEnv
import carla


models_dir = "models/thirdTry"

env = BikeEnv()
env.reset()

model_path = f"{models_dir}/1200000.zip"
model = PPO.load(model_path, env=env)

episodes = 15

#check model performance:

spectator = env.world.get_spectator()
transform = carla.Transform(carla.Location(x=-51, y=-126, z=62), carla.Rotation(pitch=-60, yaw=84, roll=0))
spectator.set_transform(transform)


for ep in range(episodes):
    obs = env.reset()
    done = False
    while not done:
        action, _states = model.predict(obs)
        obs, rewards, done, info = env.step(action)
        v = env.bike.get_velocity()
        kmh = int(3.6 * math.sqrt(v.x**2 + v.y**2 + v.z**2))

