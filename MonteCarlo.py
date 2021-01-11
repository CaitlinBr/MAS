import numpy as np
import matplotlib.pyplot as plt
import random
from tqdm import tqdm

random.seed(4)
gamma = 0.1

actions = [[-1, 0], [1, 0], [0, 1], [0, -1]]
num_it = 1000

gridSize = 9
term_state = [[6,5], [gridSize-1, gridSize-1]]
walls = [[1,2], [1,3], [1,4], [1,5], [1,6], [2,6], [3,6], [4,6], [5,6], [7,1], [7,2], [7,3], [7,4]]
goal_state = [8,8]
snakepit = [6,5]


grid = np.zeros((gridSize, gridSize))

returns = {(i, j):[] for i in range(gridSize) for j in range(gridSize)}
states = [[i, j] for i in range(gridSize) for j in range(gridSize)]
deltas = {(i, j):[] for i in range(gridSize) for j in range(gridSize)}

init_states = []
for i in states:
    if (i not in term_state) and (i not in walls):
        init_states.append(i)
    

def play():
    init_s = random.choice(init_states)
    ep = []
    while True:
        if list(init_s) in term_state:
            return ep
        action = random.choice(actions)
        last_s = np.array(init_s)+np.array(action)
        if -1 in list(last_s) or gridSize in list(last_s):
            last_s = init_s
        elif list(last_s) in walls:
            last_s = init_s
        
        if list(last_s) == goal_state:
            rew = 50
        elif list(last_s) == snakepit:
            rew = -50
        else:
            rew = -1e-7
        
        ep.append([list(init_s), action, rew, list(last_s)])
        init_s = last_s

for it in tqdm(range(num_it)):
    ep = play()
    V = 0
    for i, st in enumerate(ep[::-1]):
        V = gamma * V + st[2]
        if st[0] not in [x[0] for x in ep[::-1][len(ep)-i:]]:
            id = (st[0][0], st[0][1])
            returns[id].append(V)
            updated_v = np.average(returns[id])
            deltas[id[0], id[1]].append(np.abs(V[id[0], id[1]]-updated_v))
            grid[id[0], id[1]] = updated_v

y = ["0", "1", "2", "3", "4", "5", "6", "7", "8"]
x = y

MC = np.round(grid, 2)


# Plotting the results
fig, ax = plt.subplots(figsize=(10, 10))
im = ax.imshow(MC, cmap=plt.get_cmap('plasma'))

ax.set_xticks(np.arange(len(x)))
ax.set_yticks(np.arange(len(y)))
ax.set_xticklabels(x)
ax.set_yticklabels(y)
ax.set_ylim(8.5, -0.5)

plt.setp(ax.get_xticklabels(), ha="right",
         rotation_mode="anchor")

# From stackoverflow
for i in range(len(y)):
    for j in range(len(x)):
        text = ax.text(j, i, MC[i, j],
                       ha="center", va="center", color="w")

ax.set_title("Monte-Carlo Heatmap")
fig.tight_layout()
plt.show()
fig.savefig('MC_HM10000_0_1.png')
