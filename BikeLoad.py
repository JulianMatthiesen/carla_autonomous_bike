import gym
from stable_baselines3 import PPO
from SensorsBikeEnv import BikeEnv


models_dir = "models/1691160335"

env = BikeEnv()
env.reset()

model_path = f"{models_dir}/430000.zip"
model = PPO.load(model_path, env=env)

episodes = 5

for ep in range(episodes):
    obs = env.reset()
    done = False
    while not done:
        action, _states = model.predict(obs)
        obs, rewards, done, info = env.step(action)
        print(rewards)