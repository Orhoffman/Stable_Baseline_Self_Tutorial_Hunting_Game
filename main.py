import numpy as np
from numpy import linalg as LA
from matplotlib import pyplot as plt
from enum import Enum
from gym import Env
from gym.spaces import Box
from gym.utils import seeding

from stable_baselines3 import PPO
from stable_baselines3.ppo import MlpPolicy
# from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
import os
from stable_baselines3.common.results_plotter import load_results, ts2xy
from stable_baselines3.common.callbacks import BaseCallback

DT = 1
MIN_RR = 10
MAX_TIME = 400


class SaveOnBestTrainingRewardCallback(BaseCallback):
    """
    Callback for saving a model (the check is done every ``check_freq`` steps)
    based on the training reward (in practice, we recommend using ``EvalCallback``).

    :param check_freq: (int)
    :param log_dir: (str) Path to the folder where the model will be saved.
      It must contains the file created by the ``Monitor`` wrapper.
    :param verbose: (int)
    """
    def __init__(self, check_freq: int, log_dir: str, verbose=1):
        super(SaveOnBestTrainingRewardCallback, self).__init__(verbose)
        self.check_freq = check_freq
        self.log_dir = log_dir
        self.save_path = os.path.join(log_dir, 'best_model')
        self.best_mean_reward = -np.inf

    def _init_callback(self) -> None:
        # Create folder if needed
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)

    def _on_step(self) -> bool:
        if self.n_calls % self.check_freq == 0:

          # Retrieve training reward
          x, y = ts2xy(load_results(self.log_dir), 'timesteps')
          if len(x) > 0:
              # Mean training reward over the last 100 episodes
              mean_reward = np.mean(y[-100:])
              if self.verbose > 0:
                print(f"Num timesteps: {self.num_timesteps}")
                print(f"Best mean reward: {self.best_mean_reward:.2f} - Last mean reward per episode: {mean_reward:.2f}")

              # New best model, you could save the agent here
              if mean_reward > self.best_mean_reward:
                  self.best_mean_reward = mean_reward
                  # Example for saving best model
                  if self.verbose > 0:
                    print(f"Saving new best model to {self.save_path}.zip")
                  self.model.save(self.save_path)

        return True


class ObjTypes(Enum):
    RUNNER = 1
    HUNTER_PN = 2
    RUNNER_RL = 3
    HUNTER_RL = 4


class Animal:
    def __init__(self, pos, vel, obj_type, guid_gain=3):
        self.pos = pos
        self.vel = vel
        self.obj_type = obj_type
        self.acc = np.array([0, 0, 0])

        self.clock = 0
        self.guid_gain = guid_gain

    def reset(self, pos, vel, obj_type, guid_gain=3):
        self.pos = pos
        self.vel = vel
        self.obj_type = obj_type
        self.acc = np.array([0, 0, 0])
        self.clock = 0
        self.guid_gain = guid_gain

    def guidance(self, other_obj, acc):
        if self.obj_type == ObjTypes.HUNTER_PN:
            self.pn_guid(other_obj)
        elif self.obj_type == ObjTypes.HUNTER_RL:
            self.acc = acc
        elif self.obj_type == ObjTypes.RUNNER:
            self.acc = np.array([0, 0, 0])
        elif self.obj_type == ObjTypes.RUNNER_RL:
            self.acc = acc

    def pn_guid(self, other_obj):
        rr = other_obj.pos - self.pos
        vr = other_obj.vel - self.vel
        guid_range = LA.norm(rr)
        closing_vel = LA.norm(vr)
        los = rr / guid_range
        lambda_dot = np.cross(rr, vr) / (guid_range**2)
        self.acc = self.guid_gain * closing_vel * np.cross(lambda_dot, los)

    def dynamics_and_control(self):
        self.acc = self.acc

    def kinematics(self):
        self.vel = self.vel + self.acc*DT
        self.pos = self.pos + self.vel*DT
        self.clock = self.clock+DT


class HuntingEnvironment(Env):
    def __init__(self, cheetah_pos, cheetah_vel, cheetah_type, impala_pos, impala_vel, impala_type, cheetah_guid_gain):
        # initialize predator (cheetah) and prey (impala):
        self.init_cheetah_pos = cheetah_pos
        self.init_cheetah_vel = cheetah_vel
        self.init_cheetah_type = cheetah_type
        self.init_impala_pos = impala_pos
        self.init_impala_vel = impala_vel
        self.init_impala_type = impala_type
        self.cheetah = Animal(cheetah_pos, cheetah_vel, cheetah_type, cheetah_guid_gain)
        self.impala = Animal(impala_pos, impala_vel, impala_type)

        # action is the predator acceleration:
        self.action_space = Box(np.array([-10, -10, 0]), np.array([10, 10, 0]))

        # observations are the predator and prey position and velocity:
        self.observation_space = Box(np.array([-1000, -1000, 0,    # predator pos min
                                               -100, -100, 0,      # predator vel min
                                               -1000, -1000, 0,    # prey pos min
                                               -1000, -1000, 0]),  # prey vel min
                                     np.array([-1000, -1000, 0,    # predator pos max
                                               -1000, -1000, 0,    # predator vel max
                                               -1000, -1000, 0,    # prey pos max
                                               -1000, -1000, 0]))  # prey vel
        self.seed()
        self.viewer = None
        self.state = np.concatenate((self.cheetah.pos, self.cheetah.vel, self.impala.pos, self.impala.vel))

    def reset(self):
        self.cheetah.reset(self.init_cheetah_pos, self.init_cheetah_vel, self.init_cheetah_type)
        self.impala.reset(self.init_impala_pos, self.init_impala_vel, self.init_impala_type)
        self.state = np.concatenate((self.cheetah.pos, self.cheetah.vel, self.impala.pos, self.impala.vel))
        return self._get_obs()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, acc):
        done = False
        info = {}
        self.cheetah.guidance(self.impala, acc)
        self.cheetah.dynamics_and_control()
        self.impala.guidance(self.cheetah, None)
        self.impala.dynamics_and_control()
        self.cheetah.kinematics()
        self.impala.kinematics()
        self.state = np.concatenate((self.cheetah.pos, self.cheetah.vel, self.impala.pos, self.impala.vel))
        rr = LA.norm(self.impala.pos - self.cheetah.pos)
        reward = -rr
        if rr < MIN_RR:
            reward = 10000000000
            done = True
        if self.cheetah.clock > MAX_TIME:
            done = True

        return self._get_obs(), reward, done, info

    def render(self, mode='human'):
        screen_width = 600
        screen_height = 600
        scale = 0.2

        cartwidth = 10.0
        cartheight = 10.0

        if self.viewer is None:
            from gym.envs.classic_control import rendering
            self.viewer = rendering.Viewer(screen_width, screen_height)
            l, r, t, b = -cartwidth / 2, cartwidth / 2, cartheight / 2, -cartheight / 2

            cheetah = rendering.FilledPolygon([(l, b), (l, t), (r, t), (r, b)])
            cheetah.set_color(1, 0, 0)
            self.cheetahtrans = rendering.Transform()
            cheetah.add_attr(self.cheetahtrans)
            self.viewer.add_geom(cheetah)

            impala = rendering.FilledPolygon([(l, b), (l, t), (r, t), (r, b)])
            impala.set_color(0, 0, 1)
            self.impalatrans = rendering.Transform()
            impala.add_attr(self.impalatrans)
            self.viewer.add_geom(impala)

        if self.state is None:
            return None

        x = self.state
        cheetah_x = x[0]* scale + screen_width / 2.0
        cheetah_y = x[1]* scale
        impala_x = x[6] * scale + screen_width / 2.0
        impala_y = x[7] * scale
        self.cheetahtrans.set_translation(cheetah_x, cheetah_y)
        self.impalatrans.set_translation(impala_x, impala_y)
        return self.viewer.render(return_rgb_array=mode == 'rgb_array')

    def _get_obs(self):
        return self.state

    def run(self):
        cheetah_pos_x = []
        cheetah_pos_y = []
        cheetah_pos_x.append(self.cheetah.pos[0])
        cheetah_pos_y.append(self.cheetah.pos[1])
        impala_pos_x = []
        impala_pos_y = []
        impala_pos_x.append(self.impala.pos[0])
        impala_pos_y.append(self.impala.pos[1])

        rr = LA.norm(self.impala.pos-self.cheetah.pos)


        done = False
        i = 0
        while done is False:
            if i % 100 == 0:
                self.render()
            obs, reward, done, info = self.step(acc=None)
            cheetah_pos_x.append(self.cheetah.pos[0])
            cheetah_pos_y.append(self.cheetah.pos[1])
            impala_pos_x.append(self.impala.pos[0])
            impala_pos_y.append(self.impala.pos[1])
            rr = LA.norm(self.impala.pos - self.cheetah.pos)
            i = i+1

        plt.figure()
        line1, = plt.plot(cheetah_pos_x, cheetah_pos_y, label='cheetah')
        line2, = plt.plot(impala_pos_x, impala_pos_y, label='impala')
        plt.legend()
        plt.title('At ' + str(round(self.cheetah.clock, 2)) + ' [sec], Range is ' + str(round(rr, 2)) + '[m]')
        plt.show()

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None


def moving_average(values, window):
    """
    Smooth values by doing a moving average
    :param values: (numpy array)
    :param window: (int)
    :return: (numpy array)
    """
    weights = np.repeat(1.0, window) / window
    return np.convolve(values, weights, 'valid')


def plot_results(log_folder, title='Learning Curve'):
    """
    plot the results

    :param log_folder: (str) the save location of the results to plot
    :param title: (str) the title of the task to plot
    """
    x, y = ts2xy(load_results(log_folder), 'timesteps')
    y = moving_average(y, window=50)
    # Truncate x
    x = x[len(x) - len(y):]

    fig = plt.figure(title)
    plt.plot(x, y)
    plt.xlabel('Number of Timesteps')
    plt.ylabel('Rewards')
    plt.title(title + " Smoothed")
    plt.show()


if __name__ == '__main__':
    # env = gym.make('CartPole-v1')

    cheetah_pos = np.array([0, 0, 0])
    cheetah_vel = np.array([10, 10, 0])
    cheetah_type = ObjTypes.HUNTER_RL
    impala_pos = np.array([1000, 1000, 0])
    impala_vel = np.array([-10, 8, 0])
    impala_type = ObjTypes.RUNNER
    env = HuntingEnvironment(cheetah_pos, cheetah_vel, cheetah_type, impala_pos, impala_vel, impala_type,
                             cheetah_guid_gain=None)
    # Create log dir:
    log_dir = "/tmp/gym/"
    os.makedirs(log_dir, exist_ok=True)

    # Logs will be saved in log_dir/monitor.:
    env = Monitor(env, log_dir)

    # Create the callback: check every 1000 steps:
    callback = SaveOnBestTrainingRewardCallback(check_freq=1000, log_dir=log_dir)

    # Create RL model:
    model = PPO(MlpPolicy, env, verbose=0)

    # Use a separate environment for evaluation
    # eval_env = gym.make('CartPole-v1')

    """cheetah_pos = np.array([0, 0, 0])
    cheetah_vel = np.array([10, 10, 0])
    cheetah_type = ObjTypes.HUNTER_RL
    impala_pos = np.array([1000, 1000, 0])
    impala_vel = np.array([-10, 8, 0])
    impala_type = ObjTypes.RUNNER
    eval_env = HuntingEnvironment(cheetah_pos, cheetah_vel, cheetah_type, impala_pos, impala_vel, impala_type,
                                  cheetah_guid_gain=None)"""

    """# Random Agent, before training
    mean_reward, std_reward = evaluate_policy(model, eval_env, n_eval_episodes=100)
    print(f"mean_reward:{mean_reward:.2f} +/- {std_reward:.2f}")"""

    # Train the agent for 20000 steps
    model.learn(total_timesteps=5000000, callback=callback)

    from stable_baselines3.common import results_plotter

    # Helper from the library
    results_plotter.plot_results([log_dir], 5000000, results_plotter.X_TIMESTEPS, "Hunting game")

    plot_results(log_dir)

    """# Evaluate the trained agent
    mean_reward, std_reward = evaluate_policy(model, eval_env, n_eval_episodes=100)
    print(f"mean_reward:{mean_reward:.2f} +/- {std_reward:.2f}")"""

    obs = env.reset()
    for i in range(1200):
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
        #if i % 100 == 0:
        env.render()
        if done:
            obs = env.reset()

    env.close()
    #eval_env.close()
