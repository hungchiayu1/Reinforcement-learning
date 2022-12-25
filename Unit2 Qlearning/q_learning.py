import numpy as np
import gym
import random
from tqdm.notebook import tqdm


class QLearning:

    def __init__(self, state_space, action_space,env):

        self.state_space = state_space
        self.action_space = action_space
        self.env = env
        # Create q table of shape = (state_space,action_sapce)
        self.Qtable = np.zeros(shape=(state_space, action_space))

    def greedy_policy(self, state):

        action = np.argmax(self.Qtable[state])

        return action

    def random_action(self):

        action = np.random.randint(self.action_space)

        return action

    def epilson_greedy_policy(self, state, epsilon):

        random_num = np.random.uniform(0, 1)

        if random_num > epsilon:  # Greedy policy

            action = self.greedy_policy(state)

        else:
            action = self.random_action()

        return action

    def train(self, n_training_episodes, learning_rate, max_steps, gamma, max_epsilon, min_epsilon, decay_rate):
        '''
        
        Args:
                n_training_episodes: Number of training episodes
                learning_rate: Learning rate alpha 
                max_steps: Maximum number of steps per episode
                gamma: Discount factor
                max_epsilon: Max epsilon value
                min_epsilon: Min epsilon value
                decay_rate: Epsilon decay rate
        '''
        for episode in tqdm(range(n_training_episodes)):
            epsilon = min_epsilon + (max_epsilon - min_epsilon)*np.exp(-decay_rate*episode)
            # epsilon decay
            state = self.env.reset()

            for step in range(max_steps):

                action = self.epilson_greedy_policy(state,epsilon)

                new_state, reward, done, info = self.env.step(action)

                self.Qtable[state][action] = self.Qtable[state][action] + learning_rate * \
                    (reward+gamma*np.max(self.Qtable[new_state]
                                   ) - self.Qtable[state][action])            
                if done:
                    break

                state = new_state

    def evaluate(self,max_steps,n_eval_episodes):
        episode_rewards = []
        
        for episode in range(n_eval_episodes):
            
            state = self.env.reset()
            
            step = 0
            done = False
            total_reward = 0
            
            for step in range(max_steps):
                
                action = self.greedy_policy(state)
                
                new_state,reward,done,info = self.env.step(action)
                
                total_reward += reward
                
                if done:
                    break
                    
                state = new_state
            
            episode_rewards.append(total_reward)
            
        mean_reward = np.mean(episode_rewards)
        std_reward = np.std(episode_rewards)
        
        return mean_reward,std_reward
                
                
                
            
