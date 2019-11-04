import matplotlib.pyplot as plt
import matplotlib.widgets as mwidgets
import math
import numpy as np


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
        return max(np.amax(self.e_euler), abs(np.amin(self.e_euler))), \
               max(np.amax(self.e_improved), abs(np.amin(self.e_improved))), \
               max(np.amax(self.e_runge_kutta), abs(np.amin(self.e_runge_kutta)))

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
        print("Exact solution:\n", self.y_exact)
        print("Euler's method:\n", self.y_euler)
        print("errors: ", self.e_euler)
        print("Improved Euler's method:\n", self.y_improved)
        print("errors: ", self.e_improved)
        print("Runge-Kutta method:\n", self.y_runge_kutta)
        print("errors: ", self.e_improved)


class Application:
    labels = ["Euler's method", "Improved Euler's method", "Runge-Kutta method", "Exact solution"]
    styles = ["Segments", "Segments with dots"]

    def __init__(self, function, solution, x0, y0, X, N=20, min_N=20, max_N=100):
        # Numerical Methods
        self.function = function
        self.solution = solution
        self.x0 = x0
        self.y0 = y0
        self.X = X
        self.N = N
        self.min_N = min_N
        self.max_N = max_N
        # for displaying solutions
        self._nm = NumericalMethods(function, solution, x0, y0, X, N)
        # for calculating max errors
        self._nm2 = NumericalMethods(function, solution, x0, y0, X, self.min_N)
        # for recalculating max errors (recalculate only if x0, y0, or X change)
        self._recalculate_solutions = False
        self._recalculate_max_errors = False
        # self._nm.print()
        # GUI
        self._figure = plt.figure(figsize=(9, 8), num="Differential Equations: Numerical Methods")
        self.plot_solutions()
        self.plot_errors()
        self.plot_max_errors()
        self.draw_text_box_N()
        self.draw_text_box_x0()
        self.draw_text_box_y0()
        self.draw_text_box_X()
        self.draw_text_box_min_N()
        self.draw_text_box_max_N()
        self.draw_button()
        self.draw_radio_buttons()

    def plot_solutions(self):
        self._axes_solutions = self._figure.add_subplot(3, 1, 1)
        self._axes_solutions.set_title("Solutions of the initial value problem")
        self._axes_solutions.set_xlabel("x")
        self._axes_solutions.set_ylabel("y")
        self._axes_solutions.grid(True)
        self._solution_euler, = self._axes_solutions.plot(self._nm.x, self._nm.y_euler, 'o-', markersize=3, label=self.labels[0])
        self._solution_improved, = self._axes_solutions.plot(self._nm.x, self._nm.y_improved, 'o-', markersize=3, label=self.labels[1])
        self._solution_runge_kutta, = self._axes_solutions.plot(self._nm.x, self._nm.y_runge_kutta, 'o-', markersize=3, label=self.labels[2])
        self._solution_exact, = self._axes_solutions.plot(self._nm.x, self._nm.y_exact, 'o-', markersize=3, label=self.labels[3])
        self._legend_solution = self._axes_solutions.legend()
        for line in self._legend_solution.get_lines():
            line.set_picker(5)

    def plot_errors(self):
        self._axes_errors = self._figure.add_subplot(3, 1, 2)
        self._axes_errors.set_title("Errors in approximate solutions")
        self._axes_errors.set_xlabel("x")
        self._axes_errors.set_ylabel("error")
        self._axes_errors.grid(True)
        self._errors_euler, = self._axes_errors.plot(self._nm.x, self._nm.e_euler, 'o-', markersize=3, label=self.labels[0])
        self._errors_improved, = self._axes_errors.plot(self._nm.x, self._nm.e_improved, 'o-', markersize=3, label=self.labels[1])
        self._errors_runge_kutta, = self._axes_errors.plot(self._nm.x, self._nm.e_runge_kutta, 'o-', markersize=3, label=self.labels[2])
        self._legend_errors = self._axes_errors.legend()
        for line in self._legend_errors.get_lines():
            line.set_picker(5)

    def plot_max_errors(self):
        self._axes_max_errors = self._figure.add_subplot(3, 1, 3)
        self._axes_max_errors.set_title("Max errors in approximate solutions")
        self._axes_max_errors.set_xlabel("N")
        self._axes_max_errors.set_ylabel("error")
        self._axes_max_errors.grid(True)
        N_values = range(self.min_N, self.max_N + 1)
        max_errors_euler, max_errors_improved, max_errors_runge_kutta = \
            self._nm2.calculate_max_errors(N_values)
        self._max_errors_euler, = self._axes_max_errors.plot(N_values, max_errors_euler, 'o-', markersize=3, label=self.labels[0])
        self._max_errors_improved, = self._axes_max_errors.plot(N_values, max_errors_improved, 'o-', markersize=3, label=self.labels[1])
        self._max_errors_runge_kutta, = self._axes_max_errors.plot(N_values, max_errors_runge_kutta, 'o-', markersize=3, label=self.labels[2])
        self._legend_max_errors = self._axes_max_errors.legend()
        for line in self._legend_max_errors.get_lines():
            line.set_picker(5)

    def draw_text_box_N(self):
        self._axes_text_box_n = plt.axes([0.8, 0.86, 0.15, 0.04])
        self._text_box_n = mwidgets.TextBox(self._axes_text_box_n, "N:", initial=str(self.N))
        self._text_box_n.on_submit(self._submit_N)

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

    def draw_text_box_min_N(self):
        self._axes_text_box_min_N = plt.axes([0.8, 0.62, 0.15, 0.04])
        self._text_box_min_N = mwidgets.TextBox(self._axes_text_box_min_N, "min N:", initial=str(self.min_N))
        self._text_box_min_N.on_submit(self._submit_min_N)

    def draw_text_box_max_N(self):
        self._axes_text_box_max_N = plt.axes([0.8, 0.56, 0.15, 0.04])
        self._text_box_max_N = mwidgets.TextBox(self._axes_text_box_max_N, "max N:", initial=str(self.max_N))
        self._text_box_max_N.on_submit(self._submit_max_N)

    def draw_button(self):
        self._axes_button= plt.axes([0.8, 0.50, 0.15, 0.04])
        self._button = mwidgets.Button(self._axes_button, "Update")
        self._button.on_clicked(self._on_press_button)

    def draw_radio_buttons(self):
        self._axes_radio_buttons = plt.axes([0.75, 0.35, 0.22, 0.1])
        self._axes_radio_buttons.set_title("Line style")
        self._radio_buttons = mwidgets.RadioButtons(self._axes_radio_buttons, self.styles, 1)
        self._radio_buttons.on_clicked(self._on_click_radio_button)

    def _submit_N(self, text):
        self.N = int(text)
        self._recalculate_solutions = True

    def _submit_x0(self, text):
        self.x0 = int(text)
        self._recalculate_solutions = True
        self._recalculate_max_errors = True

    def _submit_y0(self, text):
        self.y0 = int(text)
        self._recalculate_solutions = True
        self._recalculate_max_errors = True

    def _submit_X(self, text):
        self.X = int(text)
        self._recalculate_solutions = True
        self._recalculate_max_errors = True

    def _submit_min_N(self, text):
        self.min_N = int(text)
        self._recalculate_max_errors = True

    def _submit_max_N(self, text):
        self.max_N = int(text)
        self._recalculate_max_errors = True

    def _on_press_button(self, event):
        self._recalculate()
        self._redraw()

    def _on_click_radio_button(self, style):
        solutions = [self._solution_euler, self._solution_improved, self._solution_runge_kutta, self._solution_exact]
        errors = [self._errors_euler, self._errors_improved, self._errors_runge_kutta]
        max_errors = [self._max_errors_euler, self._max_errors_improved, self._max_errors_runge_kutta]

        if style == self.styles[0]:
            for solution in solutions:
                solution.set_marker(None)
            for error in errors:
                error.set_marker(None)
            for max_error in max_errors:
                max_error.set_marker(None)

        if style == self.styles[1]:
            for solution in solutions:
                solution.set_marker("o")
            for error in errors:
                error.set_marker("o")
            for max_error in max_errors:
                max_error.set_marker("o")

        self._figure.canvas.draw()

    def _on_pick(self, event):
        solutions = [self._solution_euler, self._solution_improved, self._solution_runge_kutta, self._solution_exact]
        errors = [self._errors_euler, self._errors_improved, self._errors_runge_kutta]
        max_errors = [self._max_errors_euler, self._max_errors_improved, self._max_errors_runge_kutta]

        for i in range(4):
            if event.artist.get_label() == self.labels[i]:
                solutions[i].set_visible(not solutions[i].get_visible())
                if solutions[i].get_visible():
                    self._legend_solution.legendHandles[i].set_alpha(1)
                    self._legend_solution.legendHandles[i]._legmarker.set_alpha(1)
                else:
                    self._legend_solution.legendHandles[i].set_alpha(0.2)
                    self._legend_solution.legendHandles[i]._legmarker.set_alpha(0.2)
                if i < 3:
                    errors[i].set_visible(not errors[i].get_visible())
                    if errors[i].get_visible():
                        self._legend_errors.legendHandles[i].set_alpha(1)
                        self._legend_errors.legendHandles[i]._legmarker.set_alpha(1)
                    else:
                        self._legend_errors.legendHandles[i].set_alpha(0.2)
                        self._legend_errors.legendHandles[i]._legmarker.set_alpha(0.2)
                    max_errors[i].set_visible(not max_errors[i].get_visible())
                    if max_errors[i].get_visible():
                        self._legend_max_errors.legendHandles[i].set_alpha(1)
                        self._legend_max_errors.legendHandles[i]._legmarker.set_alpha(1)
                    else:
                        self._legend_max_errors.legendHandles[i].set_alpha(0.2)
                        self._legend_max_errors.legendHandles[i]._legmarker.set_alpha(0.2)
        self._figure.canvas.draw()

    def _recalculate(self):
        if self._recalculate_solutions:
            self._nm = NumericalMethods(self.function, self.solution, self.x0, self.y0, self.X, self.N)
            self._solution_euler.set_data(self._nm.x, self._nm.y_euler)
            self._solution_improved.set_data(self._nm.x, self._nm.y_improved)
            self._solution_runge_kutta.set_data(self._nm.x, self._nm.y_runge_kutta)
            self._solution_exact.set_data(self._nm.x, self._nm.y_exact)

            self._errors_euler.set_data(self._nm.x, self._nm.e_euler)
            self._errors_improved.set_data(self._nm.x, self._nm.e_improved)
            self._errors_runge_kutta.set_data(self._nm.x, self._nm.e_runge_kutta)

        if self._recalculate_max_errors:
            self._nm2 = NumericalMethods(self.function, self.solution, self.x0, self.y0, self.X, self.min_N)
            N_values = range(self.min_N, self.max_N + 1)
            max_errors_euler, max_errors_improved, max_errors_runge_kutta = \
                self._nm2.calculate_max_errors(N_values)
            # print(max_errors_euler, max_errors_improved, max_errors_runge_kutta)
            self._max_errors_euler.set_data(N_values, max_errors_euler)
            self._max_errors_improved.set_data(N_values, max_errors_improved)
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
        self._figure.canvas.mpl_connect('pick_event', self._on_pick)
        # plt.tight_layout()
        plt.show()


# Initial value problem:
# y' = f(x, y) = 2x(x^2 + y)
# y(x0) = y0
# x is in (x0, X)

def f(x, y):
    return 2 * x * (x ** 2 + y)


def solution(x, x0, y0):
    return (y0 + x0**2 + 1) / math.exp(x0**2) * np.exp(x**2) - x**2 - 1


app = Application(f, solution, 0, 0, 10, 20)
app.show()
