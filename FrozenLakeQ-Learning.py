import numpy as np
import random
from support import plotting
from gym.envs.toy_text.frozen_lake import FrozenLakeEnv
import matplotlib.style

matplotlib.style.use("seaborn")

# env.action_space.n - number of actions on each state
# env.observation_space.n - number of states
# env.action_space.sample() - returns a random state (for "move" action)
# env.nA - number of actions
# env.nS - number of states

"""Using Bellman equation: 
        NewQ(s, a) = Q(s, a) + alpha(R(s,a) + gamma * maxQ'(s', a') - Q(s, a))
    Where:
    NewQ(s, a) - New Q value for that state and action,
    Q(s, a) - Current Q value,
    R(s, a) - Reward for taking that action at that state,
    max Q'(s', a') - maximum expected future reward with given new and all possible actions at that new state, 
    s - Current State of the agent,
    a - Current Action Picked according to some policy,
    s' - Next State where the agent ends up,
    a' - Next best action to be picked using current Q-value estimation, i.e. pick the action with the maximum Q-value 
    in the next state,
    R - Current Reward observed from the environment in Response of current action,
    gama (>0 and <=1) - Discounting Factor for Future Rewards. Future rewards are less valuable than current rewards so 
    they must be discounted. Since Q-value is an estimation of expected rewards from a state, discounting rule applies 
    here as well,
    alpha - Step length taken to update the estimation of Q(S, A).

    From given above equation we can make:
        New Q(s, a) = (1 - alpha) * Q(s, a) + alpha(R + gamma * max Q(s', a')"""

"""Exploration parameters
exploit vs explore to find action
Start with 70% random actions to explore the environment
And with time, using decay to shift to more optimal actions learned from experience
decay - decay rate for exploration probability"""


"""Choosing the Action to take using 𝜖 - greedy policy:
𝜖 - greedy policy of is a very simple policy of choosing actions using the current Q-value estimations. 
It goes as follows:
With probability (1 - ϵ) choose the action which has the highest Q-value.
With probability (ϵ) choose any action at random."""


class FrozenLakeQLearning:
    def __init__(self):
        self.mapSize = "4x4"
        self.slippery = True
        self.env = FrozenLakeEnv(map_name=self.mapSize, is_slippery=self.slippery)
        self.qTable = np.zeros((self.env.observation_space.n, self.env.action_space.n))
        self.episodesAmount = 1000
        self.alpha = 0.9
        self.gamma = 0.9
        self.epsilon = 0.9
        self.epsilonDecay = 0.99
        self.epsilonMin = 0.01
        self.epsilonMax = 1.0
        self.decay = 0.01
        self.stepsMax = 100
        self.action = self.env.action_space.sample()
        self.rewards = []
        self.steps = []
        self.stats = plotting.EpisodeStats(episode_lengths=np.zeros(self.episodesAmount),
                                           episode_rewards=np.zeros(self.episodesAmount))
        self.qLearning()

    def qLearning(self):
        for i in range(self.episodesAmount):
            state = self.env.reset()
            totalReward = 0
            step = 0

            for step in range(self.stepsMax):
                randomEpsilon = random.uniform(0, 1)

                if randomEpsilon > self.epsilon:
                    action = np.argmax(self.qTable[state])
                else:
                    action = self.env.action_space.sample()

                # Step format is: (1, 0.0, False, {'prob': 0.3333333333333333})
                newState, reward, done, probability = self.env.step(action)

                self.qTable[state, action] = self.qTable[state, action] + self.alpha * (reward + self.gamma * np.max(self.qTable[newState, :]) - self.qTable[state, action])

                self.stats.episode_rewards[i] += reward
                totalReward += reward
                self.stats.episode_lengths[i] = step

                state = newState

                if done:
                    if i % 10 == 0:
                        print(f"Episode: {i} Reward: {reward} Steps Taken: {step}")
                    break

            self.epsilon = self.epsilonMin + self.epsilonDecay * np.exp(-self.decay * i)

            self.rewards.append(totalReward)
            self.steps.append(step)
        print("Score over time: " + str(sum(self.rewards) / self.episodesAmount))
        plotting.plot_episode_stats(self.stats)


FrozenLakeQLearning()
