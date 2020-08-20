import numpy as np
from support import plotting
from gym.envs.toy_text.cliffwalking import CliffWalkingEnv
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


class CliffWalkingQLearning:
    def __init__(self):
        self.env = CliffWalkingEnv()
        self.qTable = np.zeros((self.env.nS, self.env.nA))
        self.epsilon = 0.7
        self.epsilonDecay = 0.005
        self.epsilonMax = 1.0
        self.epsilonMin = 0.01
        self.state = 0
        self.action = 0
        self.stepsAmount = 200
        self.episodesAmount = 1000
        self.alpha = 0.95
        self.gamma = 0.99
        self.decay = 0.01
        self.rewards = []
        self.stats = plotting.EpisodeStats(episode_lengths=np.zeros(self.episodesAmount),
                                           episode_rewards=np.zeros(self.episodesAmount))
        self.qLearning()
        plotting.plot_episode_stats(self.stats)

    def epsilonGreedyPolicy(self):
        if np.random.uniform(0, 1) > self.epsilon:
            self.action = np.argmax(self.qTable[self.state])
        else:
            self.action = self.env.action_space.sample()

    def qLearning(self):
        for episode in range(self.episodesAmount):
            self.state = self.env.reset()

            for step in range(self.stepsAmount):
                self.epsilonGreedyPolicy()
                newState, reward, done, _ = self.env.step(self.action)
                self.qTable[self.state, self.action] = (1 - self.alpha) * self.qTable[self.state, self.action] + self.alpha * (reward + self.gamma * np.max(self.qTable[newState]))
                self.state = newState
                self.stats.episode_rewards[episode] += reward
                self.rewards.append(reward)
                if done:
                    break

            self.epsilon = self.epsilonMin + self.epsilonDecay * np.exp(-self.decay * episode)
        print("Score over time: " + str(sum(self.rewards) / self.episodesAmount))


CliffWalkingQLearning()
