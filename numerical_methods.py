import matplotlib.pyplot as plt
import math
import numpy as np

# Initial value problem:
# y' = f(x, y) = 2x(x^2 + y)
# y(x0) = y0
# x is in (x0, X)
#
# y' = 2x(x^2 + y)
# y(0) = 0
# x is in (0, 10)
#
# solution: y = (y0 + x0^2 + 1) / e^(x0^2) * e^(x^2) - x^2 - 1
x0 = 0
y0 = 1
X = 1
N = 10


# def f(x, y):
#     return 2 * x * (x**2 + y)
def f(x, y):
    return x**3 * math.exp(-2 * x) - 2 * y

x = np.linspace(x0, X, N + 1)


def euler_method():
    y = np.empty(N + 1)
    y[0] = y0
    for i in range(1, N + 1):
        y[i] = y[i - 1] + (x[i] - x[i - 1]) * f(x[i - 1], y[i - 1])
    return y


def improved_euler_method():
    y = np.empty(N + 1)
    y[0] = y0
    for i in range(1, N + 1):
        y[i] = y[i - 1] + (x[i] - x[i - 1]) * f(x[i - 1], y[i - 1])
        y[i] = y[i - 1] + (x[i] - x[i - 1]) * (f(x[i - 1], y[i - 1]) + f(x[i], y[i])) / 2
    return y


def runge_kutta_method():
    y = np.empty(N + 1)
    y[0] = y0
    for i in range(1, N + 1):
        h = (x[i] - x[i - 1])
        k1 = f(x[i - 1], y[i - 1])
        k2 = f(x[i - 1] + h / 2, y[i - 1] + h * k1 / 2)
        k3 = f(x[i - 1] + h / 2, y[i - 1] + h * k2 / 2)
        k4 = f(x[i - 1] + h, y[i - 1] + h * k3)
        y[i] = y[i - 1] + h * (k1 + 2*k2 + 2*k3 + k4) / 6
    return y


# y_exact = (y0 + x0**2 + 1) / math.exp(x0**2) * np.exp(x**2) - x**2 - 1
y_exact = np.exp(-2 * x) / 4 * (x**4 + 4)
y_euler = euler_method()
y_improved_euler = improved_euler_method()
y_runge_kutta = runge_kutta_method()

e_euler = y_exact - y_euler
e_improved = y_exact - y_improved_euler
e_runge_kutta = y_exact - y_runge_kutta
np.set_printoptions(formatter={'float': '{: 0.5f}'.format})
print("x:\n", x)
print("Euler's method:\n", y_euler)
print("errors: ", e_euler)
print("Improved Euler's method:\n", y_improved_euler)
print("errors: ", e_improved)
print("Runge-Kutta method:\n", y_runge_kutta)
print("errors: ", e_improved)

# plt.figure(figsize=(8, 10))

plt.subplot(2, 1, 1)
plt.title("Solutions of the initial value problem")
plt.xlabel("x")
plt.ylabel("y")
plt.grid(True)
plt.plot(x, y_euler, 'o-', label="Euler's method")
plt.plot(x, y_improved_euler, 'o-', label="Improved Euler's method")
plt.plot(x, y_runge_kutta, 'o-', label="Runge-Kutta method")
plt.plot(x, y_exact, 'o-', label="Exact solution")
plt.legend()

plt.subplot(2, 1, 2)
plt.title("Errors in approximate solutions")
plt.xlabel("x")
plt.ylabel("error")
plt.grid(True)
plt.plot(x, e_euler, 'o-', label="Euler's method")
plt.plot(x, e_improved, 'o-', label="Improved Euler's method")
plt.plot(x, e_runge_kutta, 'o-', label="Runge-Kutta method")
plt.legend()

plt.show()
