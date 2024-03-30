import gym
from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy

import warnings
warnings.filterwarnings('ignore')
class MsPacmanDQNAgent:
    def __init__(self):
        self.env = gym.make("MsPacman-v4")
        self.env = DummyVecEnv([lambda: self.env])
        self.model = self._build_model()
        self.env.close()

    def _build_model(self):
        return DQN("CnnPolicy", self.env, verbose=1)

    def train(self, total_timesteps=200000):
        self.model.learn(total_timesteps=total_timesteps)

    def test(self, num_episodes=100):
        mean_reward, _ = evaluate_policy(self.model, self.env, n_eval_episodes=num_episodes)
        return mean_reward

# Usage example
pacman_agent = MsPacmanDQNAgent()
pacman_agent.train(total_timesteps=300 * 60)
mean_reward = pacman_agent.test(num_episodes=100)
print("Mean reward:", mean_reward)
