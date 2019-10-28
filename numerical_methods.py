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


# def f(x, y):
#     return 2 * x * (x**2 + y)

# y_exact = (y0 + x0**2 + 1) / math.exp(x0**2) * np.exp(x**2) - x**2 - 1


class NumericalMethods:
    def __init__(self, function, solution, x0, y0, X, N):
        self.f = function
        self.s = solution
        self.x0 = x0
        self.y0 = y0
        self.X = X
        self.N = N
        self.h = (X - x0) / N
        self.x = np.linspace(x0, X, N + 1)
        self.calculate()

    def calculate(self):
        # Function values
        self.y_exact = self.calculate_exact_solution()
        self.y_euler = self.calculate_euler_method()
        self.y_improved_euler = self.calculate_improved_euler_method()
        self.y_runge_kutta = self.calculate_runge_kutta_method()
        # Error values
        self.e_euler = self.y_exact - self.y_euler
        self.e_improved = self.y_exact - self.y_improved_euler
        self.e_runge_kutta = self.y_exact - self.y_runge_kutta

    def calculate_exact_solution(self):
        vfunc = np.vectorize(lambda x: self.s(x, self.x0, self.y0))
        return vfunc(self.x)

    def calculate_euler_method(self):
        y = np.empty(self.N + 1)
        y[0] = self.y0
        for i in range(1, self.N + 1):
            y[i] = y[i - 1] + self.h * self.f(self.x[i - 1], y[i - 1])
        return y

    def calculate_improved_euler_method(self):
        y = np.empty(self.N + 1)
        y[0] = self.y0
        for i in range(1, self.N + 1):
            y[i] = y[i - 1] + self.h * self.f(self.x[i - 1], y[i - 1])
            y[i] = y[i - 1] + self.h * (self.f(self.x[i - 1], y[i - 1]) + self.f(self.x[i], y[i])) / 2
        return y

    def calculate_runge_kutta_method(self):
        y = np.empty(self.N + 1)
        y[0] = self.y0
        for i in range(1, self.N + 1):
            k1 = self.f(self.x[i - 1], y[i - 1])
            k2 = self.f(self.x[i - 1] + self.h / 2, y[i - 1] + self.h * k1 / 2)
            k3 = self.f(self.x[i - 1] + self.h / 2, y[i - 1] + self.h * k2 / 2)
            k4 = self.f(self.x[i - 1] + self.h, y[i - 1] + self.h * k3)
            y[i] = y[i - 1] + self.h * (k1 + 2 * k2 + 2 * k3 + k4) / 6
        return y

    def print(self):
        np.set_printoptions(formatter={'float': '{: 0.5f}'.format})
        print("x:\n", self.x)
        print("Euler's method:\n", self.y_euler)
        print("errors: ", self.e_euler)
        print("Improved Euler's method:\n", self.y_improved_euler)
        print("errors: ", self.e_improved)
        print("Runge-Kutta method:\n", self.y_runge_kutta)
        print("errors: ", self.e_improved)


class Application:
    def __init__(self, function, solution, x0, y0, X, N):
        self.figure = plt.figure()
        self.function = function
        self.solution = solution
        self.x0 = x0
        self.y0 = y0
        self.X = X
        self.N = N
        self.nm = NumericalMethods(function, solution, x0, y0, X, N)
        self.axes_solutions = self.figure.add_subplot(2, 1, 1)
        self.axes_errors = self.figure.add_subplot(2, 1, 2)
        self.nm.print()
        self.plot_solutions()
        self.plot_errors()

    def plot_solutions(self):
        self.axes_solutions.set_title("Solutions of the initial value problem")
        self.axes_solutions.set_xlabel("x")
        self.axes_solutions.set_ylabel("y")
        self.axes_solutions.grid(True)
        self.axes_solutions.plot(self.nm.x, self.nm.y_euler, 'o-', label="Euler's method")
        self.axes_solutions.plot(self.nm.x, self.nm.y_improved_euler, 'o-', label="Improved Euler's method")
        self.axes_solutions.plot(self.nm.x, self.nm.y_runge_kutta, 'o-', label="Runge-Kutta method")
        self.axes_solutions.plot(self.nm.x, self.nm.y_exact, 'o-', label="Exact solution")
        self.axes_solutions.legend()

    def plot_errors(self):
        self.axes_errors.set_title("Errors in approximate solutions")
        self.axes_errors.set_xlabel("x")
        self.axes_errors.set_ylabel("error")
        self.axes_errors.grid(True)
        self.axes_errors.plot(self.nm.x, self.nm.e_euler, 'o-', label="Euler's method")
        self.axes_errors.plot(self.nm.x, self.nm.e_improved, 'o-', label="Improved Euler's method")
        self.axes_errors.plot(self.nm.x, self.nm.e_runge_kutta, 'o-', label="Runge-Kutta method")
        self.axes_errors.legend()

    def show(self):
        plt.tight_layout()
        plt.show()


def f1(x, y):
    return x**3 * math.exp(-2 * x) - 2 * y


def solution1(x, x0, y0):
    return math.exp(-2 * x) / 4 * (x ** 4 + 4)


def f2(x, y):
    return 2 * x * (x**2 + y)


def solution2(x, x0, y0):
    return (y0 + x0**2 + 1) / math.exp(x0**2) * math.exp(x**2) - x**2 - 1


app = Application(f2, solution2, 0, 0, 2, 20)
app.show()