import gym
from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env

import warnings
warnings.filterwarnings('ignore')

class CartPolePPOAgent:
    def __init__(self):
        self.vec_env = make_vec_env("CartPole-v1")

    def train(self, total_timesteps=2500):
        # Train using PPO
        self.model = PPO("MlpPolicy", self.vec_env, verbose=1)
        self.model.learn(total_timesteps=total_timesteps)
        self.model.save("ppo_cartpole")
        self.vec_env.close()  # Close the environment after training

    def test(self):
        # Test the trained PPO model
        self.model = PPO.load("ppo_cartpole")
        obs = self.vec_env.reset()
        done = False
        scores = 0
        while not done:
            action, _states = self.model.predict(obs)
            obs, rewards, done, info = self.vec_env.step(action)
            scores += rewards
            self.vec_env.render("human")
        print('Score: {}'.format(scores[0]))

# Create an instance of the CartPolePPOAgent
agent = CartPolePPOAgent()

# Train the agent
agent.train(total_timesteps=2500)

# Test the agent
agent.test()

# Evaluate the policy using DQN
model_dqn = DQN("MlpPolicy", env="CartPole-v1")
mean_reward, std_reward = evaluate_policy(model_dqn, model_dqn.get_env(), n_eval_episodes=10)
print(f"Mean reward: {mean_reward}, Std reward: {std_reward}")



"""
The "Standard deviation" of 1.136 gives us an idea of how consistent the agent's performance is. 
A lower standard deviation would mean that the agent's performance is more consistent, 
while a higher standard deviation means its performance varies more from game to game. 
In this case, with a standard deviation of 1.136, the agent's performance varies somewhat from game to game, 
but it's not extremely variable."""
