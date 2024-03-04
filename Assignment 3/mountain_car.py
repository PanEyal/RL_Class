import numpy as np
import gymnasium as gym
import matplotlib.pyplot as plt

env = gym.make('MountainCar-v0')
from tile_coding import *

a = create_tiling_grid([-1.2, -0.07], [0.6, 0.07], (4, 4), (0, 0))
print(a)