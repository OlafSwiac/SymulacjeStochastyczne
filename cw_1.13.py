import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

t_std = np.random.standard_t(1, size=(10000,2))
cauchy = np.random.standard_cauchy(size=(10000, 2))


