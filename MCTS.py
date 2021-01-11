import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import random


class db:

    def __init__(self, rootn, m):
        self.root = rootn
        self.leaf_nodes = m.leaf_nodes
        self.mc_end = m.mc_end
        self.dist = m.dist

    def unif_dist(self, c):
        l = len(self.dist)
        percent = int(l * 0.05)

        plt.figure()
        sns.distplot(self.dist)
        plt.xlim(0, 100)
        plt.ylabel('Density')
        plt.legend()
        plt.savefig('dist_plot_' + str(c) + '.png')

        return

    def leafs(self):
        l = len(self.leaf_nodes)
        node_b = self.leaf_nodes[0]
        node_w = self.leaf_nodes[0]
        ranking = 0

        for n in self.leaf_nodes:
            if node_b.reward < n.reward:
                node_b = n
            if node_w.reward > n.reward:
                node_w = n

        leafs = self.leaf_nodes
        leafs = sorted(leafs, key=lambda k: k.reward)

        for i in range(0, l):
            if self.mc_end.reward == leafs[i].reward:
                ranking = i + 1

        perc_rank = round((ranking / l) * 100, 2)

        print(f"Ranking of leaf node: reward is {self.mc_end.reward} with ranking: {ranking}/{l} ({perc_rank}%)")

        return perc_rank

    def create_box(self, box):
        box.boxplot(column = ['0', '0.01', '0.05', '0.1', '0.5', '1', '2', '5'])
        plt.show()

    def visualizing_rewards(self, stats):

        df = pd.DataFrame()
        for c in stats:
            df = df.append({'c': c, 'max': stats[c]['max'], 'min': stats[c]['min'], 'avg': stats[c]['avg']},
                           ignore_index=True)

        df_melted = df.melt(id_vars='c')

        plt.figure(figsize=(16, 10))
        sns.barplot(x='c', y='value', hue='variable', data=df_melted, palette="Blues_d")
        plt.ylim(0, 100)
        plt.ylabel('Reward')
        plt.xlabel('c-value')
        plt.legend()
        plt.savefig('c_vals.png')
        plt.show()





class Node:
    node_count = 0
    def __init__(self, reward=0, depth=0, left=None, right=None):
        self.reward = reward
        self.ucb_value_init = np.inf
        self.visits = 0
        self.depth = depth
        self.left = left
        self.right = right
        Node.node_count += 1

    def __str__(self):
        return f"({self.reward}) - UCB: {self.ucb_value_init}"

class Binary_Tree:
    def __init__(self, depth=12, root=None, c_value=0.05):
        self.node_count = 0
        self.leaf_count = 0
        self.depth = depth
        self.root = Node(0)
        self.c = c_value
        self.dist = np.random.uniform(0, 100, 2 ** (self.depth))
        self.dist_size = len(self.dist)
        self.leaf_nodes = []
        self.mcts_result = None
        print(f"node: {self.node_count} ({self.root.reward})")



    def MCTS(self, root):
        best_reward_node = self.select(root)
        return best_reward_node

    def tree(self, node, cd=0, node_count=0):
        if cd < self.depth:
            if node.left is None:
                self.node_count += 1
                if cd + 1 == self.depth:
                    node.left = Node(self.dist[self.leaf_count])
                    self.leaf_count += 1
                    self.leaf_nodes.append(node.left)
                else:
                    node.left = Node(0)
                self.tree(node.left, cd + 1)
            if node.right is None:
                if cd + 1 == self.depth:
                    node.right = Node(self.dist[self.leaf_count])
                    self.leaf_count += 1
                    self.leaf_nodes.append(node.right)
                else:
                    node.right = Node(0)
                self.tree(node.right, cd + 1)

        return node


    def snow(self, node, parent_node=None, expanded=False):

        node.visits += 1

        if not expanded:
            if node.left.ucb_value > node.right.ucb_value:
                if self.is_terminal(node):
                    return None
                self.snow(node.left, node)
            elif node.left.ucb_value < node.right.ucb_value:
                if self.is_terminal(node):
                    return None
                self.snow(node.right, node)
            # if both are unexplored (or equal), random choice (rollout)
            elif node.left.ucb_value == node.right.ucb_value:
                self.snow(self.expand(node), node, expanded=True)

        # check if leaf node
        if self.is_terminal(node):
            # if self.mcts_result is None or self.mcts_result.reward < node.reward:
            self.mcts_result = node
            print(f"BEST REWARD ---> {node.reward} <--- (all time champion: {self.mcts_result.reward})")

            return None

        leaf_reward = self.sim(node)
        node = self.back_up(node, parent_node, leaf_reward)

        return node

    def select(self, root):
        iter = 0
        while not self.is_terminal(root):
            iter += 1
            print('it:', iter)
            for i in range(0, 25):
                root = self.snow(root)

            if root.left.ucb_value > root.right.ucb_value:
                root = root.left
            else:
                root = root.right

        return root

    def expand(self, node):
        if random.random() < 0.5:
            return node.left
        else:
            return node.right

    def UCB(self, node, parent_visits):
        n = node.visits
        N = parent_visits
        r = node.reward / n
        c = self.c

        # UCB1 formula value
        node.ucb_value = r + c * (np.sqrt(np.log(N) / n))
        return node

    def sim(self, node):
        while node.left is not None or node.right is not None:
            if random.random() < 0.5:
                node = node.left
            else:
                node = node.right
        return node.reward

    def back_up(self, node, parent_node, leaf_reward):
        if node.left is not None or node.right is not None:
            node.reward += leaf_reward

        if parent_node is None:
            return node

        node = self.UCB(node, parent_node.visits)
        return node

    def is_terminal(self, node):
        if node.left is None and node.right is None:
            return True
        else:
            return False



class Main:

depth = 12
c_values = [0, 0.01, 0.05, 0.1, 0.5, 1, 2, 5]
repeats = 30 # number of iterations

rwrds = {}
box = pd.DataFrame(np.zeros((repeats, len(c_values))), columns=['0', '0.01', '0.05', '0.1', '0.5', '1', '2', '5'])

for c in c_values:
    max_rank = 0
    avg_rank = 0
    min_rank = 100
    rwrds[c] = {}

    for i in range(0, repeats):
        m = Binary_Tree(depth, c)
        rootn = m.tree(m.root)
        m.dist.sort()

        rootn = m.MCTS(rootn)

        rewards = db(rootn, m)
        rewards.unif_dist(c)
        rankingp = rewards.leafs()

        rwrds[c][i] = rankingp
        box[str(c)][i]=rankingp
        if max_rank < rankingp:
            max_rank = rankingp
        if min_rank > rankingp:
            min_rank = rankingp

        avg_rank += rankingp

    rwrds[c]['max'] = max_rank
    rwrds[c]['min'] = min_rank
    rwrds[c]['avg'] = avg_rank/repeats+1


rewards.visualizing_rewards(rwrds)
rewards.create_box(box)
