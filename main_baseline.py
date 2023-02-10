import argparse
import numpy as np
import gym
import highway_env
import time
from utils.env_config import env_kwargs
from gym.wrappers import RecordVideo

parser = argparse.ArgumentParser()
parser.add_argument("--baseline", type=str, choices=['vanilla', 'grid', 'random', 'batch'], default="vanilla", help="MPC baselines")
parser.add_argument("--episodes", type=int, default=50, help="select number of episodes")
parser.add_argument("--density",  type=float, choices=[1.0, 1.5, 2.5, 3.0], default=3.0, help="Vehicle density")
parser.add_argument("--four_lane", type=bool, default=True, help="Use 4 or 2 lanes")
parser.add_argument("--record", type=bool, default=False, help="record environment")
parser.add_argument("--render", type=bool, default=False, help="render the environment")

args = parser.parse_args()
baseline = args.baseline
n_episodes = args.episodes
four_lane_bool = args.four_lane
if four_lane_bool: lane_count = 4
else: lane_count = 2
env_density = args.density
record_bool = args.record
render_bool = args.render

# Deisred Velocity
v_des = 20

# Initializing the baseline policy
if baseline == "vanilla":
    from expert_mpc.policy_vanilla import ExpertPolicy
    expert = ExpertPolicy()
elif baseline == "grid":
    from expert_mpc.policy_grid import ExpertPolicy
    expert = ExpertPolicy()
elif baseline == "random":
    from expert_mpc.policy_random import ExpertPolicy
    expert = ExpertPolicy()
else:
    from expert_mpc.policy_batch import ExpertPolicy
    expert = ExpertPolicy()

# Environment name
env_name = 'highway-v0'

# Obstacle Velocity
params = [15]

# Environment Density
density_dict = {params[0] : env_density}

rec_video = record_bool
if __name__ == "__main__":
    start = time.time()
    render = render_bool
    for param in params:
        num_episodes = n_episodes
        collisions = 0
        speeds = []
        avg_speed = []
        env_kwargs['config']['lanes_count'] = lane_count
        env_kwargs['config']['speed_limit'] = param
        env_kwargs['config']['vehicles_density'] = density_dict[param]
        env = gym.make(env_name, **env_kwargs)
        if rec_video:
            env = RecordVideo(env, video_folder=f"./videos/{baseline}/",
                          episode_trigger=lambda e: True)
            env.unwrapped.set_record_video_wrapper(env)        
        env.seed(42)
        obs = env.reset()
        cnt = 0
        while cnt < num_episodes:
            speeds.append(obs[2])
            ax = env.vehicle.ax
            ay = env.vehicle.ay
            action = expert.predict(obs, ax, ay, v_des)
            if rec_video:
                env.env.env.viewer.set_agent_action_sequence([action])
            obs, rew, done, info = env.step(action)
            if render:
                env.render()
            if done == True:
                cnt += 1  
                if not info['crashed']:
                    avg_speed.append(np.sum(speeds) / len(speeds))
                else:
                    collisions += 1
                    avg_speed.append(0.)                
                print("-" * 100)
                print(f"Episode: {cnt}")
                print(f"Collisions: {collisions}")
                if not info["crashed"]:
                    print(f"Speed: {np.sum(speeds) / len(speeds)}")
                else:
                    print(f"Speed: {0.}")
                print("-" * 100)
                speeds = []
                obs = env.reset()
        collision_rate = collisions / num_episodes      
        print('Average Collision Rate: ' + str(collision_rate))
        print(f"Average Speed: {np.average(avg_speed)}")
        np.savez(f'./results/{baseline}_stat.npz', collisions=np.array([collision_rate]), avg_speeds=np.array(avg_speed))
    print('Elapsed' + str(time.time() - start))