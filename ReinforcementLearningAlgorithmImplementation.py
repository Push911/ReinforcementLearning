import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from support import plotting


class ReinforcementLearningAlgorithmImplementation:
    def __init__(self):
        self.edges = [(0, 1), (1, 5), (5, 6), (5, 4), (1, 2), (1, 3), (9, 10), (2, 4), (0, 6), (6, 7), (8, 9), (7, 8),
                      (1, 7), (3, 9)]
        self.goal = 10
        self.matrixSize = 11
        self.matrix = np.ones((self.matrixSize, self.matrixSize)) * -1
        self.qTable = np.zeros((self.matrixSize, self.matrixSize))
        self.alpha = 0.85
        self.gamma = 0.9
        self.state = 1
        self.scores = 0
        self.availableAction = ()
        self.nextAction = 0
        self.episodesAmount = 1000
        self.stepsMaxAmount = 100
        self.createRewardSystem()
        self.createGraph()
        self.stats = plotting.EpisodeStats(episode_lengths=np.zeros(self.episodesAmount),
                                           episode_rewards=np.zeros(self.episodesAmount))
        self.qLearning()
        plotting.plot_episode_stats(self.stats)

    def createGraph(self):
        graph = nx.Graph()
        graph.add_edges_from(self.edges)
        pos = nx.spring_layout(graph)
        nx.draw_networkx_nodes(graph, pos)
        nx.draw_networkx_edges(graph, pos)
        nx.draw_networkx_labels(graph, pos)
        plt.savefig("ReinforcementLearningGraph")
        plt.show()

    def createRewardSystem(self):
        for point in self.edges:
            if point[0] == self.goal:
                self.matrix[point[::-1]] = 100
            else:
                self.matrix[point[::-1]] = 0

            if point[1] == self.goal:
                self.matrix[point] = 100
            else:
                self.matrix[point] = 0

    def availableActions(self):
        self.availableAction = np.where(self.matrix[self.state] >= 0)[0]

    def createNextAction(self):
        self.nextAction = np.random.choice(self.availableAction)

    def update(self):
        bestAction = np.argmax(self.qTable[self.nextAction])
        self.qTable[self.state, self.nextAction] = self.qTable[self.state, self.nextAction] + self.alpha * (self.matrix[self.state, self.nextAction] + self.gamma * self.qTable[self.nextAction, bestAction])
        self.scores += self.matrix[self.state, self.nextAction]

    def qLearning(self):
        for i in range(self.episodesAmount):
            self.state = np.random.randint(0, self.qTable.shape[0])
            self.availableActions()
            self.createNextAction()
            self.update()
            self.stats.episode_rewards[i] += self.matrix[self.state, self.nextAction]
        print(f"Reached destination {self.scores/100} times")


if __name__ == "__main__":
    ReinforcementLearningAlgorithmImplementation()
