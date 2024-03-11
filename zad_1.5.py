import numpy as np
import matplotlib.pyplot as plt

X = []

for i in range(1000000):
    u = np.random.rand()
    x = np.random.rand()
    while x <= u:
        #u = np.random.rand()
        x = np.random.rand()
    X.append(x)
plt.hist(X, bins=1000, density=True)
plt.show()

# za kazdym razem losujemy u --> P(x > t) = 1/2 * t^2
# raz losujemy u --> P(x > t) = 1/2 * t^2

