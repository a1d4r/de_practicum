import matplotlib.pyplot as plt
import matplotlib.widgets as mwidgets
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
        self.y_improved = self.calculate_improved_euler_method()
        self.y_runge_kutta = self.calculate_runge_kutta_method()
        # Error values
        self.e_euler = self.y_exact - self.y_euler
        self.e_improved = self.y_exact - self.y_improved
        self.e_runge_kutta = self.y_exact - self.y_runge_kutta
        # Max error values
        self.max_e_euler = np.amax(self.e_euler)
        self.max_e_improved = np.amax(self.e_improved)
        self.max_e_runge_kutta = np.amax(self.e_runge_kutta)

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

    def calculate_max_error(self, N):
        self.N = N
        self.h = (self.X - self.x0) / N
        self.x = np.linspace(self.x0, self.X, N + 1)
        self.calculate()
        return np.amax(self.e_euler), np.amax(self.e_improved), np.amax(self.e_runge_kutta)

    def calculate_max_errors(self, N_values):
        max_errors_euler = np.empty(len(N_values))
        max_errors_improved = np.empty(len(N_values))
        max_errors_runge_kutta = np.empty(len(N_values))
        for i, N in enumerate(N_values):
            max_errors_euler[i],  max_errors_improved[i], max_errors_runge_kutta[i] = \
                self.calculate_max_error(N)
        return max_errors_euler, max_errors_improved, max_errors_runge_kutta

    def print(self):
        np.set_printoptions(formatter={'float': '{: 0.5f}'.format})
        print("x:\n", self.x)
        print("Euler's method:\n", self.y_euler)
        print("errors: ", self.e_euler)
        print("Improved Euler's method:\n", self.y_improved)
        print("errors: ", self.e_improved)
        print("Runge-Kutta method:\n", self.y_runge_kutta)
        print("errors: ", self.e_improved)


class Application:
    min_N = 20
    max_N = 100

    def __init__(self, function, solution, x0, y0, X, N):
        # Numerical Methods
        self.function = function
        self.solution = solution
        self.x0 = x0
        self.y0 = y0
        self.X = X
        self.N = N
        # for displaying solutions
        self._nm = NumericalMethods(function, solution, x0, y0, X, N)
        # for calculating max errors
        self._nm2 = NumericalMethods(function, solution, x0, y0, X, self.min_N)
        # for recalculating max errors (recalculate only if x0, y0, or X change)
        self._recalculate_max_errors = False
        # GUI
        self._figure = plt.figure(figsize=(9, 8))
        # self._nm.print()
        self.plot_solutions()
        self.plot_errors()
        self.plot_max_errors()
        self.draw_text_box_N()
        self.draw_text_box_x0()
        self.draw_text_box_y0()
        self.draw_text_box_X()
        self.draw_button()

    def plot_solutions(self):
        self._axes_solutions = self._figure.add_subplot(3, 1, 1)
        self._axes_solutions.set_title("Solutions of the initial value problem")
        self._axes_solutions.set_xlabel("x")
        self._axes_solutions.set_ylabel("y")
        self._axes_solutions.grid(True)
        self._solution_euler, = self._axes_solutions.plot(self._nm.x, self._nm.y_euler, 'o-', label="Euler's method")
        self._solution_impoved, = self._axes_solutions.plot(self._nm.x, self._nm.y_improved, 'o-', label="Improved Euler's method")
        self._solution_runge_kutta, = self._axes_solutions.plot(self._nm.x, self._nm.y_runge_kutta, 'o-', label="Runge-Kutta method")
        self._solution_exact, = self._axes_solutions.plot(self._nm.x, self._nm.y_exact, 'o-', label="Exact solution")
        self._axes_solutions.legend()

    def plot_errors(self):
        self._axes_errors = self._figure.add_subplot(3, 1, 2)
        self._axes_errors.set_title("Errors in approximate solutions")
        self._axes_errors.set_xlabel("x")
        self._axes_errors.set_ylabel("error")
        self._axes_errors.grid(True)
        self._errors_euler, = self._axes_errors.plot(self._nm.x, self._nm.e_euler, 'o-', label="Euler's method")
        self._errors_impoved, = self._axes_errors.plot(self._nm.x, self._nm.e_improved, 'o-', label="Improved Euler's method")
        self._errors_runge_kutta, = self._axes_errors.plot(self._nm.x, self._nm.e_runge_kutta, 'o-', label="Runge-Kutta method")
        self._axes_errors.legend()

    def plot_max_errors(self):
        self._axes_max_errors = self._figure.add_subplot(3, 1, 3)
        self._axes_max_errors.set_title("Max errors in approximate solutions")
        self._axes_max_errors.set_xlabel("N")
        self._axes_max_errors.set_ylabel("error")
        self._axes_max_errors.grid(True)
        N_values = range(self.min_N, self.max_N + 1)
        max_errors_euler, max_errors_improved, max_errors_runge_kutta = \
            self._nm2.calculate_max_errors(N_values)
        self._max_errors_euler, = self._axes_max_errors.plot(N_values, max_errors_euler, 'o-', label="Euler's method")
        self._max_errors_impoved, = self._axes_max_errors.plot(N_values, max_errors_improved, 'o-', label="Improved Euler's method")
        self._max_errors_runge_kutta, = self._axes_max_errors.plot(N_values, max_errors_runge_kutta, 'o-', label="Runge-Kutta method")
        self._axes_max_errors.legend()

    def draw_text_box_N(self):
        self._axes_text_box_n = plt.axes([0.8, 0.86, 0.15, 0.04])
        self._text_box_n = mwidgets.TextBox(self._axes_text_box_n, "N:", initial=str(self.N))
        self._text_box_n.on_submit(self._submit_n)

    def draw_text_box_x0(self):
        self._axes_text_box_x0 = plt.axes([0.8, 0.80, 0.15, 0.04])
        self._text_box_x0 = mwidgets.TextBox(self._axes_text_box_x0, "x0:", initial=str(self.x0))
        self._text_box_x0.on_submit(self._submit_x0)

    def draw_text_box_y0(self):
        self._axes_text_box_y0 = plt.axes([0.8, 0.74, 0.15, 0.04])
        self._text_box_y0 = mwidgets.TextBox(self._axes_text_box_y0, "y0:", initial=str(self.y0))
        self._text_box_y0.on_submit(self._submit_y0)

    def draw_text_box_X(self):
        self._axes_text_box_X = plt.axes([0.8, 0.68, 0.15, 0.04])
        self._text_box_X = mwidgets.TextBox(self._axes_text_box_X, "X:", initial=str(self.X))
        self._text_box_X.on_submit(self._submit_X)

    def draw_button(self):
        self._axes_button= plt.axes([0.8, 0.62, 0.15, 0.04])
        self._button = mwidgets.Button(self._axes_button, "Update")
        self._button.on_clicked(self._press_button)

    def _submit_n(self, text):
        self.N = int(text)

    def _submit_x0(self, text):
        self.x0 = int(text)
        self._recalculate_max_errors = True

    def _submit_y0(self, text):
        self.y0 = int(text)
        self._recalculate_max_errors = True

    def _submit_X(self, text):
        self.X = int(text)
        self._recalculate_max_errors = True

    def _press_button(self, event):
        self._recalculate()
        self._redraw()

    def _recalculate(self):
        self._nm = NumericalMethods(self.function, self.solution, self.x0, self.y0, self.X, self.N)

        self._solution_euler.set_data(self._nm.x, self._nm.y_euler)
        self._solution_impoved.set_data(self._nm.x, self._nm.y_improved)
        self._solution_runge_kutta.set_data(self._nm.x, self._nm.y_runge_kutta)
        self._solution_exact.set_data(self._nm.x, self._nm.y_exact)

        self._errors_euler.set_data(self._nm.x, self._nm.e_euler)
        self._errors_impoved.set_data(self._nm.x, self._nm.e_improved)
        self._errors_runge_kutta.set_data(self._nm.x, self._nm.e_runge_kutta)

        if self._recalculate_max_errors:
            self._nm2 = NumericalMethods(self.function, self.solution, self.x0, self.y0, self.X, self.min_N)
            N_values = range(self.min_N, self.max_N + 1)
            max_errors_euler, max_errors_improved, max_errors_runge_kutta = \
                self._nm2.calculate_max_errors(N_values)
            self._max_errors_euler.set_data(N_values, max_errors_euler)
            self._max_errors_impoved.set_data(N_values, max_errors_improved)
            self._max_errors_runge_kutta.set_data(N_values, max_errors_runge_kutta)

    def _redraw(self):
        self._axes_solutions.relim()
        self._axes_solutions.autoscale_view(True,True,True)
        self._axes_errors.relim()
        self._axes_errors.autoscale_view(True,True,True)

        if self._recalculate_max_errors:
            self._axes_max_errors.relim()
            self._axes_max_errors.autoscale_view(True,True,True)
            self._recalculate_max_errors = False
        plt.draw()

    def show(self):
        plt.subplots_adjust(left=0.1, bottom=0.1, right=0.7, top=0.9, wspace=None, hspace=1.0)
        # plt.tight_layout()
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