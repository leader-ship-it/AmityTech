from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor
from environment import delivery  

# Wrap  environment
env = DummyVecEnv([lambda: delivery()])
env = VecMonitor(env)  # Automatically tracks rewards and episode lengths

# Create and train model
model = PPO.load("ppo_delivery_model")
model.set_env(env)
model.learn(total_timesteps=1000000)
model.save("ppo_delivery_model")
