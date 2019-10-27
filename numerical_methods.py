import matplotlib.pyplot as plt
import math
import numpy as np

# Initial value problem:
# y' = 2x(x^2 + y)
# y(x0) = y0
# x is in (x0, X)
x0 = 0
y0 = 0
X = 2
N = 50

x_a = np.linspace(x0, X, N)
y = [(y0 + x0**2 + 1) / math.exp(x0**2) * math.exp(x**2) - x**2 - 1 for x in x_a]
plt.title("Exact solution")
plt.xlabel("x")
plt.ylabel("y")
plt.grid(True)
plt.plot(x_a, y)

plt.show()
