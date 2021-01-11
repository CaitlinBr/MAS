
class World:
  def __init__(self, width, height, start):
    self.width = width
    self.height = height
    self.i = start[0]
    self.j = start[1]

  def init_state(self, s):
    self.i = s[0]
    self.j = s[1]

  def cs(self):
    return (self.i, self.j)

  def set(self, rewards, actions):
    self.rewards = rewards
    self.actions = actions

  def move(self, action):
    if action in self.actions[(self.i, self.j)]:
      if action == 'N':
        self.i -= 1
      elif action == 'S':
        self.i += 1
      elif action == 'W':
        self.j += 1
      elif action == 'E':
        self.j -= 1
    return self.rewards.get((self.i, self.j), 0)

  def undo(self, action):
    if action == 'N':
      self.i += 1
    elif action == 'S':
      self.i -= 1
    elif action == 'W':
      self.j -= 1
    elif action == 'E':
      self.j += 1
    assert(self.cs() in self.all_states())


  def all_states(self):
    return set(self.actions.keys()) | set(self.rewards.keys())


  def go(self):
    return (self.i, self.j) not in self.actions


def basic_g():
  grid = World(9, 9, (0, 0))
  rewards = {(8, 8): 50, (6, 5): -50}
  actions = {
    (0, 0): ('S', 'W'),
    (0, 1): ('E', 'W', 'S'),
    (0, 2): ('E', 'W'),
    (0, 3): ('E', 'W'),
    (0, 4): ('E', 'W'),
    (0, 5): ('E', 'W'),
    (0, 6): ('E', 'W'),
    (0, 7): ('E', 'S', 'W'),
    (0, 8): ('E', 'S'),

    (1, 0): ('N', 'S', 'W'),
    (1, 1): ('N', 'S', 'E'),
    (1, 7): ('W', 'N', 'S'),
    (1, 8): ('E', 'N', 'S'),

    (2, 0): ('N', 'W', 'S'),
    (2, 1): ('E', 'W', 'N', 'S'),
    (2, 2): ('E', 'W', 'S'),
    (2, 3): ('E', 'W', 'S'),
    (2, 4): ('E', 'W', 'S'),
    (2, 5): ('E', 'S'),
    (2, 7): ('N', 'W', 'S'),
    (2, 8): ('N', 'E', 'S'),

    (3, 0): ('N', 'W', 'S'),
    (3, 1): ('N', 'W', 'E', 'S'),
    (3, 2): ('N', 'W', 'E', 'S'),
    (3, 3): ('N', 'W', 'E', 'S'),
    (3, 4): ('N', 'W', 'E', 'S'),
    (3, 5): ('N','E', 'S'),
    (3, 7): ('N', 'W', 'S'),
    (3, 8): ('N', 'E', 'S'),

    (4, 0): ('N', 'W', 'S'),
    (4, 1): ('N', 'W', 'E', 'S'),
    (4, 2): ('N', 'W', 'E', 'S'),
    (4, 3): ('N', 'W', 'E', 'S'),
    (4, 4): ('N', 'W', 'E', 'S'),
    (4, 5): ('N', 'E', 'S'),
    (4, 7): ('N', 'W', 'S'),
    (4, 8): ('N', 'E', 'S'),

    (5, 0): ('N', 'W', 'S'),
    (5, 1): ('N', 'W', 'E', 'S'),
    (5, 2): ('N', 'W', 'E', 'S'),
    (5, 3): ('N', 'W', 'E', 'S'),
    (5, 4): ('N', 'W', 'E', 'S'),
    (5, 5): ('N', 'E', 'S'),
    (5, 7): ('N', 'W', 'S'),
    (5, 8): ('N', 'E', 'S'),

    (6, 0): ('N', 'W', 'S'),
    (6, 1): ('N', 'W', 'E'),
    (6, 2): ('N', 'W', 'E'),
    (6, 3): ('N', 'W', 'E'),
    (6, 4): ('N', 'W', 'E'),
    (6, 6): ('W', 'E', 'S'),
    (6, 7): ('N', 'W', 'E', 'S'),
    (6, 8): ('N', 'E', 'S'),

    (7, 0): ('N', 'S'),

    (7, 5): ('N', 'S', 'W'),
    (7, 6): ('N', 'S', 'E', 'W'),
    (7, 7): ('N', 'S', 'E', 'W'),
    (7, 8): ('N', 'S', 'E'),

    (8, 0): ('N', 'W'),
    (8, 1): ('W', 'E'),
    (8, 2): ('W', 'E'),
    (8, 3): ('W', 'E'),
    (8, 4): ('W', 'E'),
    (8, 5): ('N', 'E', 'W'),
    (8, 6): ('W', 'E', 'N'),
    (8, 7): ('N', 'W', 'E'),
  }
  grid.set(rewards, actions)
  return grid

def moving_cost(step_cost=-1.0):
  grid = basic_g()
  grid.rewards.update({
    (0, 0): step_cost,
    (0, 1): step_cost,
    (0, 2): step_cost,
    (0, 3): step_cost,
    (0, 4): step_cost,
    (0, 5): step_cost,
    (0, 6): step_cost,
    (0, 7): step_cost,
    (0, 8): step_cost,

    (1, 0): step_cost,
    (1, 1): step_cost,
    (1, 7): step_cost,
    (1, 8): step_cost,

    (2, 0): step_cost,
    (2, 1): step_cost,
    (2, 2): step_cost,
    (2, 3): step_cost,
    (2, 4): step_cost,
    (2, 5): step_cost,
    (2, 7): step_cost,
    (2, 8): step_cost,

    (3, 0): step_cost,
    (3, 1): step_cost,
    (3, 2): step_cost,
    (3, 3): step_cost,
    (3, 4): step_cost,
    (3, 5): step_cost,
    (3, 7): step_cost,
    (3, 8): step_cost,

    (4, 0): step_cost,
    (4, 1): step_cost,
    (4, 2): step_cost,
    (4, 3): step_cost,
    (4, 4): step_cost,
    (4, 5): step_cost,
    (4, 7): step_cost,
    (4, 8): step_cost,

    (5, 0): step_cost,
    (5, 1): step_cost,
    (5, 2): step_cost,
    (5, 3): step_cost,
    (5, 4): step_cost,
    (5, 5): step_cost,
    (5, 7): step_cost,
    (5, 8): step_cost,

    (6, 0): step_cost,
    (6, 1): step_cost,
    (6, 2): step_cost,
    (6, 3): step_cost,
    (6, 4): step_cost,
    (6, 6): step_cost,
    (6, 7): step_cost,
    (6, 8): step_cost,

    (7, 0): step_cost,
    (7, 5): step_cost,
    (7, 6): step_cost,
    (7, 7): step_cost,
    (7, 8): step_cost,

    (8, 0): step_cost,
    (8, 1): step_cost,
    (8, 2): step_cost,
    (8, 3): step_cost,
    (8, 4): step_cost,
    (8, 5): step_cost,
    (8, 6): step_cost,
    (8, 7): step_cost,
  })

  return grid

def print_values(V, g):
  for i in range(g.width):
    print("------------------------------------------------------")
    for j in range(g.height):
      v = V.get((i,j), 0)
      if v >= 0:
        print(" %.2f|" % v, end="")
      else:
        print("%.2f|" % v, end="")
    print(" ")


def print_policy(P, g):
  for i in range(g.width):
    print("------------------------------------------------------")
    for j in range(g.height):
      a = P.get((i,j), ' ')
      if a=='W':
        a='E'
      elif a == 'E':
        a = 'W'
      print("  %s  |" % a, end="")
    print(" ")


def values_max(d):
  key_max = None
  val_max = float('-inf')
  for k, v in d.items():
    if v > val_max:
      val_max = v
      key_max = k
  return key_max, val_max


