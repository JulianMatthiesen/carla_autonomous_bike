from stable_baselines3.common.env_checker import check_env
from SensorsBikeEnv import BikeEnv

env = BikeEnv()
check_env(env)