import numpy as np
import random
# Selection Algorithm. A set of weights represents the probability of selection of each
# chromosome in a group of choices. It returns the index of the chosen chromosome.
# ---------------------------------------------------------
def selection(weights):
  accumulation = np.cumsum(weights)
  p = random.uniform(0.0, 1.0)
  chosen_index = -1
  for i in range(len(accumulation)):
    if (accumulation[i] > p):
      chosen_index = i
      break

  return chosen_index