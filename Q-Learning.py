import itertools
from collections import defaultdict
import numpy as np
from support import plotting
from support.windy_gridworld import WindyGridworldEnv
import matplotlib.style
matplotlib.style.use("seaborn")

""" Q-Learning is a basic form of Reinforcement Learning which uses Q-values (also called action values) to iteratively 
    improve the behavior of the learning agent.
1.  Q-Values or Action-Values: Q-values are defined for states and actions. Q(S, A) is an estimation of how good is it 
    to take the action A at the state S. This estimation of Q(S, A) will be iteratively computed using the TD- Update 
    rule which we will see in the upcoming sections.
2.  Rewards and Episodes: An agent over the course of its lifetime starts from a start state, makes a number of 
    transitions from its current state to a next state based on its choice of action and also the environment the agent 
    is interacting in. At every step of transition, the agent from a state takes an action, observes a reward from the 
    environment, and then transits to another state. If at any point of time the agent ends up in one of the terminating 
    states that means there are no further transition possible. This is said to be the completion of an episode.
3.  Temporal Difference or TD-Update:
        The Temporal Difference or TD-Update rule can be represented as follows :
        Q(S, A) <- Q(S, A) + alpha(R + gamma * Q(S', A') - Q(S, A)
        
        
This update rule to estimate the value of Q is applied at every time step of the agents interaction with the 
environment. The terms used are:

S:  Current State of the agent.
A:  Current Action Picked according to some policy.
S': Next State where the agent ends up.
A': Next best action to be picked using current Q-value estimation, i.e. pick the action with the maximum Q-value in 
        the next state.
R:  Current Reward observed from the environment in Response of current action.
        gamma(>0 and <=1) : Discounting Factor for Future Rewards. Future rewards are less valuable than current rewards 
        so they must be discounted. Since Q-value is an estimation of expected rewards from a state, discounting rule 
        applies here as well.
alpha: Step length taken to update the estimation of Q(S, A).

Choosing the Action to take using epsilon-greedy policy:
epsilon - greedy policy of is a very simple policy of choosing actions using the current Q-value estimations. 
It goes as follows : ~With probability (1-epsilon) choose the action which has the highest Q-value.
                     ~With probability (epsilon) choose any action at random."""

"""Directions:  UP = 0
                RIGHT = 1
                DOWN = 2
                LEFT = 3"""

env = WindyGridworldEnv()
# env.seed(100)
# print(env.render())


def createEpsilonGreedyPolicy(QFunction, epsilon, actionsAmount):
    # Creates an epsilon-greedy policy based on a given Q-function and epsilon
    def policyFunction(state):
        # state - represent current position in array (linear)
        actionProbabilities = np.ones(actionsAmount, dtype=float) * epsilon / actionsAmount
        # QFunction[state] - represents the next move direction probabilities for each direction
        bestAction = np.argmax(QFunction[state])
        # bestAction - direction to move,
        actionProbabilities[bestAction] += (1.0 - epsilon)
        return actionProbabilities
    return policyFunction


def qLearning(environment, episodesAmount, discountFactor=1.0, alpha=0.6, epsilon=0.1):
    QFunction = defaultdict(lambda: np.zeros(environment.action_space.n))
    stats = plotting.EpisodeStats(episode_lengths=np.zeros(episodesAmount), episode_rewards=np.zeros(episodesAmount))
    policy = createEpsilonGreedyPolicy(QFunction, epsilon, environment.action_space.n)

    for episode in range(episodesAmount):
        state = environment.reset()
        for i in itertools.count():
            actionProbabilities = policy(state)
            print(actionProbabilities)
    #         # actionProbabilities - represents the next move direction probabilities for each direction
    #         action = np.random.choice(np.arange(len(actionProbabilities)), p=actionProbabilities)
    #         # action - random choice from actionProbabilities 1D array
    #         nextState, reward, done, _ = environment.step(action)
    #         # print(env.render())
    #         # nextState - current position on the grid,
    #         # reward - ,
    #         # done - boolean, to show if agent reached destination(won)
    #         stats.episode_rewards[episode] += reward
    #         stats.episode_lengths[episode] = i
    #         bestNextAction = np.argmax(QFunction[nextState])
    #         # print("rew", reward)
    #         temporalDifferenceTarget = reward + discountFactor * QFunction[nextState][bestNextAction]
    #         print(temporalDifferenceTarget)
    #         temporalDifferenceDelta = temporalDifferenceTarget - QFunction[state][action]
    #         QFunction[state][action] += alpha * temporalDifferenceDelta
    #         if done:
    #             break
    #         # print(env.render())
    #         state = nextState
    #
    # return QFunction, stats


Q, stats = qLearning(env, 1000)
plotting.plot_episode_stats(stats)
