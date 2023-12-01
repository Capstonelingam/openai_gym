# %%
import gym
from stable_baselines3 import PPO,DQN
from stable_baselines3.common.env_util import make_vec_env
# from stable_baselines3.common.envs import DummyVecEnv
import time
import os
import numpy as np
from stable_baselines3.common.vec_env import VecFrameStack

# %%
vec_env= make_vec_env("FrozenLake-v1", n_envs = 8)


# %%
from stable_baselines3.common.callbacks import EvalCallback,StopTrainingOnRewardThreshold
#Evalcallback and Stop Training Stopping training
modelpath=os.path.join("Training","Models")
logpath=os.path.join("Training","Logs")

# %%
stop_callback=StopTrainingOnRewardThreshold(reward_threshold=200,verbose=1)
eval_callback=EvalCallback(vec_env,
                           callback_on_new_best=stop_callback,
                           eval_freq=5000,
                           best_model_save_path=modelpath,
                           verbose=1)

# %%
vec_env.action_space

# %%
vec_env.observation_space

# %%
model = DQN("MlpPolicy", vec_env, verbose = 1,tensorboard_log=logpath)

model.learn(total_timesteps = 2000000,callback=eval_callback)
model.save("Training/Models/1M_4X4_DQN")
model.load('Training/Models/best_model.zip')

# %%
#model = PPO.load("PPO_frozenlake_part1")

# %%
vec_env= make_vec_env("FrozenLake-v1", n_envs = 16)

# %%
obs=vec_env.reset()
obs

# %%
vec_env.render(mode="rgb_array")

# %%
action,_info=model.predict(obs)
action

# %%


# %%
type(np.int64(model.predict(vec_env.observation_space.sample())[0].item()))

# %%
vec_env.observation_space.sample()

# %%
vec_env.step(action)

# %%
from render_browser import render_browser
@render_browser
def test():
    obs = vec_env.reset() # reset the  envirnoment
    done=[False]
    score=0
    while True not in done:
        action,_info = model.predict(obs)
        obs, rewards,done, info = vec_env.step(action)
        #done=terminated or truncated # for human render
        yield vec_env.render(mode="rgb_array")
        score+=sum(rewards)
        print(score,end="   ") # for human render
        # vec_env.render()

# %%
test()

# %%



