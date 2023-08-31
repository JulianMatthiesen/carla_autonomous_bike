from stable_baselines3.common.env_checker import check_env
from SensorsBikeEnvWithoutPositionOrSemSeg import BikeEnv

env = BikeEnv()
check_env(env)