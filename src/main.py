import os
import gym
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import EvalCallback

# Training parameters
timesteps = 10_000_000
log_frequency = 10_000

# Evaluation parameters
run_eval = True
render_active = False
episodes = 20

# Model parameters
model_type = "PPO"
policy = 'MlpPolicy'

# Environment parameters
environment = 'LunarLander-v2'

# Directory for load and saving the model/training results
log_dir = os.path.join('../output/logs', environment)
models_dir = os.path.join(
    '../output/models', environment, f'{model_type}-extra')
saved_model_path = os.path.join(models_dir, 'best_model.zip')

# Create directories if they don't exist
for dir in [log_dir, models_dir]:
    os.makedirs(dir, exist_ok=True)

print("### Reinforcement Learning Runner ###")
print('Executing {} model on {} environment\n'.format(model_type, environment))

env = gym.make(environment)
model = PPO(policy, env, verbose=0, tensorboard_log=log_dir)

if os.path.exists(saved_model_path):
    print('Loading model from {}'.format(saved_model_path))
    model = PPO.load(saved_model_path, env=env, tensorboard_log=log_dir)
    if run_eval:
        print("Evaluating loaded model in {} episodes".format(episodes))
        evaluate_policy(model, env, n_eval_episodes=episodes,
                        render=render_active)
        env.reset()

if not run_eval:
    eval_callback = EvalCallback(env, best_model_save_path=models_dir,
                                 log_path=log_dir, eval_freq=log_frequency,
                                 deterministic=True, render=False)
    print("learning for {} timesteps".format(timesteps))
    model.learn(total_timesteps=timesteps,
                callback=eval_callback, tb_log_name=model_type)
    env.close()
