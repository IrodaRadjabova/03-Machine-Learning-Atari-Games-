import gym
from stable_baselines3 import DQN
from stable_baselines3.common.atari_wrappers import AtariWrapper
from stable_baselines3.common.vec_env import VecFrameStack
from stable_baselines3.common.env_util import make_atari_env
import warnings
warnings.filterwarnings('ignore')

class SpaceInvadersAgent:
    def __init__(self):
        self.env_id = "SpaceInvadersNoFrameskip-v4"
        self.model = None

    def setup_environment(self):
        env = make_atari_env(self.env_id, n_envs=1)
        env = VecFrameStack(env, n_stack=4)
        return env

    def train(self, total_timesteps=2500):
        env = self.setup_environment()
        model = DQN("CnnPolicy", env, verbose=1)
        model.learn(total_timesteps=total_timesteps)
        model.save("dqn_space_invaders")
        self.model = model
        env.close()

    def evaluate_policy(self, num_episodes=10):
        if self.model is None:
            print("Model not trained yet. Please train the model first.")
            return

        env = make_atari_env(self.env_id, n_envs=1)
        env = VecFrameStack(env, n_stack=4)

        total_rewards = []
        for _ in range(num_episodes):
            obs = env.reset()
            done = False
            episode_reward = 0
            while not done:
                action, _states = self.model.predict(obs, deterministic=True)
                obs, reward, done, _ = env.step(action)
                episode_reward += reward
            total_rewards.append(episode_reward)
        
        mean_reward = sum(total_rewards) / len(total_rewards)
        print(f"Mean reward over {num_episodes} episodes: {mean_reward}")

# Usage example
agent = SpaceInvadersAgent()
agent.train(total_timesteps=2500)
agent.evaluate_policy(num_episodes=10)

"""ep_len_mean: Average number of steps the agent takes in an episode. Higher values mean the agent is taking longer to complete episodes.

ep_rew_mean: Average reward obtained per episode. Higher values indicate better performance.

exploration_rate: Probability of the agent taking a random action instead of following its learned policy. A higher rate means more exploration.

episodes: Number of episodes completed during training.

fps: Speed at which the agent is interacting with the environment.

time_elapsed: Time elapsed since the start of training.

total_timesteps: Total number of actions taken by the agent during training.

Looking at the output, increasing ep_len_mean and ep_rew_mean suggest the agent is learning to play the game better. The constant exploration_rate indicates consistent exploration. The Mean reward over 10 episodes of [0.] might be due to specific episodes or randomness in the environment.

Monitoring these statistics helps understand the agent's learning progress and performance, guiding further training adjustments for better results.

"""