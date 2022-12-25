
from q_learning import QLearning
import numpy as np
import gym

def main(map_name = "4x4",is_slippery=False):
    # Training parameters
    n_training_episodes = 10000  # Total training episodes
    learning_rate = 0.6  # Learning rate

    # Evaluation parameters
    n_eval_episodes = 250  # Total number of test episodes

    # Environment parameters
    env_id = "FrozenLake-v1"  # Name of the environment
    max_steps = 999  # Max steps per episode
    gamma = 0.95  # Discounting rate
    eval_seed = []  # The evaluation seed of the environment

    # Exploration parameters
    max_epsilon = 1.0  # Exploration probability at start
    min_epsilon = 0.05  # Minimum exploration probability
    decay_rate = 0.00005  # Exponential decay rate for exploration prob
    
    
    ##Env
    
    env = gym.make("FrozenLake-v1", map_name=map_name, is_slippery=is_slippery)
    
    state_space = env.observation_space.n
    action_space = env.action_space.n
    
    model = QLearning(state_space,action_space,env)
    
    model.train(n_training_episodes, learning_rate, max_steps, gamma, max_epsilon, min_epsilon, decay_rate)
    
    mean_reward,std_reward = model.evaluate(max_steps,n_eval_episodes)
    print(mean_reward,std_reward)
    
    return model

if __name__ == "main":
    
    main()
    
    
    
