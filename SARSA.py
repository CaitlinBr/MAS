import numpy as np
import matplotlib.pyplot as plt
from gWorld import moving_cost, print_policy, print_values
import random

random.seed(4)


epsi = 1e-3
gamma = 1.0
pos_actions = ('N', 'S', 'E', 'W')
alfa = 0.4

def values_max(d):
  key_max = None
  val_max = float('-inf')
  for k, v in d.items():
    if v > val_max:
      val_max = v
      key_max = k
  return key_max, val_max

def rand_a(a, eps=0.1):
  p = np.random.random()
  if p < (1 - eps):
    return a
  else:
    return np.random.choice(pos_actions)


grid = moving_cost(step_cost=-1.0)

# Initializing the Q-dictionary
Q = {}
states = grid.all_states()
for s in states:
  Q[s] = {}
  for a in pos_actions:
    Q[s][a] = 0


update_counts = {}
update_counts_sa = {}
for s in states:
  update_counts_sa[s] = {}
  for action in pos_actions:
    update_counts_sa[s][action] = 1.0

time = 1.0
deltas = []
for its in range(100000):
  if its % 10 == 0:
    time += 1e-2
  if its % 10000 == 0:
    print("iteration:", its)

  start = (0, 0) # Start in this state
  grid.init_state(start)


  action = values_max(Q[s])[0]
  action = rand_a(action, eps=0.5/time)
  delta_max = 0
  while not grid.go():
    r = grid.move(action)
    state2 = grid.cs()

    action2 = values_max(Q[state2])[0]
    action2 = rand_a(action2, eps=0.5/time)
    # Updating rule SARSA
    alpha = alfa / update_counts_sa[start][action]
    update_counts_sa[start][action] += 0.005
    old_qsa = Q[start][action]
    Q[start][action] = Q[start][action] + alpha * (r + gamma * Q[state2][action2] - Q[start][action])
    delta_max = max(delta_max, np.abs(old_qsa - Q[start][action]))

    update_counts[start] = update_counts.get(start,0) + 1
    start = state2
    action = action2

  deltas.append(delta_max)


plt.figure(figsize=(10,10))
plt.plot(deltas)
plt.xlabel('Iterations')
plt.ylabel('Delta')
plt.savefig('Deltas_SARSA_100000_1.0_0.4.png')


policy = {}
P = {}
for s in grid.actions.keys():
  a, max_q = values_max(Q[s])
  policy[s] = a
  P[s] = max_q


total = np.sum(list(update_counts.values()))
for k, v in update_counts.items():
  update_counts[k] = float(v) / total

print_values(P, grid)

print_policy(policy, grid)

n_list = 8
sarsa_result = list(P.values())
sarsa_result = [round(num, 4) for num in sarsa_result]
final = [sarsa_result[i * n_list:(i + 1) * n_list] for i in range((len(sarsa_result) + n_list - 1) // n_list )]

y = ["0", "1", "2", "3", "4", "5", "6", "7", "8"]
x = y

values = sarsa_result
inserting_list = [11, 12, 13, 14, 15, 24, 33, 42, 51, 59, 64, 65, 66, 67, 80]
for elem in inserting_list:
    values.insert(elem, 0.0)

fig, ax = plt.subplots(figsize=(10, 10))
sarsa_result_a = np.array(values).reshape((9, 9))

im = ax.imshow(sarsa_result_a, cmap=plt.get_cmap('plasma'))

ax.set_xticks(np.arange(9))
ax.set_yticks(np.arange(9))
ax.set_xticklabels(x)
ax.set_yticklabels(y)
ax.set_ylim(8.5, -0.5)

plt.setp(ax.get_xticklabels(), ha="right",
         rotation_mode="anchor")

# From stackoverflow
for i in range(len(y)):
    for j in range(len(x)):
        text = ax.text(j, i, sarsa_result_a[i, j], ha="center", va="center", color="w")

ax.set_title("SARSA Heatmap")
fig.tight_layout()
fig.savefig('SARSA_HM_100000_1.0_0.4.png')
