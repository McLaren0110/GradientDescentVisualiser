import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.animation import FuncAnimation
from matplotlib.lines import Line2D
from matplotlib.backends.backend_qt5agg import FigureCanvas
from PyQt5.QtWidgets import *
from PyQt5.QtCore import QTimer
matplotlib.use('Qt5Agg')


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.init_variables()
        self.init_ui()

    # Functions and Gradients
    def rosenbrock(self, x, y):
        return (1 - x) ** 2 + 100 * (y - x ** 2) ** 2

    def rosenbrock_gradient(self, x, y):
        dx = -2 * (1 - x) - 400 * x * (y - x ** 2)
        dy = 200 * (y - x ** 2)
        return np.array([dx, dy])

    def ackley(self, x, y):
        return -20 * np.exp(-0.2 * np.sqrt(0.5 * (x ** 2 + y ** 2))) - np.exp(0.5 * (np.cos(2 * np.pi * x) + np.cos(2 * np.pi * y))) + np.e + 20

    def ackley_gradient(self, x, y):
        dx = (x * np.exp(-0.2 * np.sqrt(0.5 * (x ** 2 + y ** 2))) / np.sqrt(0.5 * (x ** 2 + y ** 2))) + (np.sin(2 * np.pi * x) * np.exp(0.5 * (np.cos(2 * np.pi * x) + np.cos(2 * np.pi * y))))
        dy = (y * np.exp(-0.2 * np.sqrt(0.5 * (x ** 2 + y ** 2))) / np.sqrt(0.5 * (x ** 2 + y ** 2))) + (np.sin(2 * np.pi * y) * np.exp(0.5 * (np.cos(2 * np.pi * x) + np.cos(2 * np.pi * y))))
        return np.array([dx, dy])

    def himmelblau(self, x, y):
        return (x ** 2 + y - 11) ** 2 + (x + y ** 2 - 7) ** 2

    def himmelblau_gradient(self, x, y):
        dx = 4 * x * (x ** 2 + y - 11) + 2 * (x + y ** 2 - 7)
        dy = 2 * (x ** 2 + y - 11) + 4 * y * (x + y ** 2 - 7)
        return np.array([dx, dy])

    def init_variables(self):
        self.ani = None
        self.ani_3d = None
        self.animation_running = False
        self.path_lines = []
        self.selected_point = None
        self.function = None
        self.num_iterations = 100
        self.adagrad_learning_rate = 0.01
        self.adagrad_epsilon = 1e-8
        self.adam_learning_rate = 0.01
        self.adam_beta1 = 0.9
        self.adam_beta2 = 0.999
        self.adam_epsilon = 1e-8
        self.nesterov_learning_rate = 0.0001
        self.nesterov_momentum = 0.9

        self.function_details = {
            "Rosenbrock Function": {
                'function': self.rosenbrock,
                'gradient': self.rosenbrock_gradient,
                'bounds': {'xmin': -2, 'xmax': 2, 'ymin': -1, 'ymax': 3}
            },
            "Ackley Function": {
                'function': self.ackley,
                'gradient': self.ackley_gradient,
                'bounds': {'xmin': -5, 'xmax': 5, 'ymin': -5, 'ymax': 5}
            },
            "Himmelblau's Function": {
                'function': self.himmelblau,
                'gradient': self.himmelblau_gradient,
                'bounds': {'xmin': -5, 'xmax': 5, 'ymin': -5, 'ymax': 5}
            }
        }

    def init_ui(self):
        centralWidget = QWidget()
        layout = QHBoxLayout()

        QTimer.singleShot(0, self.plot_function)

        buttonLayout = QVBoxLayout()
        self.create_function_selector(buttonLayout)
        self.create_gradient_descent_inputs(buttonLayout)
        self.create_gradient_descent_button(buttonLayout)
        self.create_no_point_selected_label(buttonLayout)
        self.create_optimizers_checkbox(buttonLayout)
        self.create_optimizer_parameter_inputs(buttonLayout)

        layout.addLayout(buttonLayout)
        self.create_3d_plot(layout)
        self.create_2d_plot(layout)

        centralWidget.setLayout(layout)
        self.setCentralWidget(centralWidget)
    
    def create_optimizer_parameter_inputs(self, layout):
        # Adagrad
        self.adagrad_learning_rate_input = QLineEdit(str(self.adagrad_learning_rate))
        self.adagrad_epsilon_input = QLineEdit(str(self.adagrad_epsilon))
        layout.addWidget(QLabel("Adagrad learning rate:"))
        layout.addWidget(self.adagrad_learning_rate_input)
        layout.addWidget(QLabel("Adagrad epsilon:"))
        layout.addWidget(self.adagrad_epsilon_input)

        # Adam
        self.adam_learning_rate_input = QLineEdit(str(self.adam_learning_rate))
        self.adam_beta1_input = QLineEdit(str(self.adam_beta1))
        self.adam_beta2_input = QLineEdit(str(self.adam_beta2))
        self.adam_epsilon_input = QLineEdit(str(self.adam_epsilon))
        layout.addWidget(QLabel("Adam learning rate:"))
        layout.addWidget(self.adam_learning_rate_input)
        layout.addWidget(QLabel("Adam beta1:"))
        layout.addWidget(self.adam_beta1_input)
        layout.addWidget(QLabel("Adam beta2:"))
        layout.addWidget(self.adam_beta2_input)
        layout.addWidget(QLabel("Adam epsilon:"))
        layout.addWidget(self.adam_epsilon_input)

        # Nesterov
        self.nesterov_learning_rate_input = QLineEdit(str(self.nesterov_learning_rate))
        self.nesterov_momentum_input = QLineEdit(str(self.nesterov_momentum))
        layout.addWidget(QLabel("Nesterov learning rate:"))
        layout.addWidget(self.nesterov_learning_rate_input)
        layout.addWidget(QLabel("Nesterov momentum:"))
        layout.addWidget(self.nesterov_momentum_input)

    def create_function_selector(self, layout):
        self.functionSelector = QComboBox()
        for function_name in self.function_details:
            self.functionSelector.addItem(function_name)
        self.functionSelector.currentIndexChanged.connect(self.plot_function)
        layout.addWidget(self.functionSelector)

    def create_gradient_descent_inputs(self, layout):
        self.num_iterations_input = QLineEdit(str(self.num_iterations))
        layout.addWidget(QLabel("Number of iterations:"))
        layout.addWidget(self.num_iterations_input)

    def create_gradient_descent_button(self, layout):
        self.gradientDescentButton = QPushButton("Plot Descent paths")
        self.gradientDescentButton.clicked.connect(self.plot_algorithms)
        layout.addWidget(self.gradientDescentButton)

    def create_no_point_selected_label(self, layout):
        self.no_point_selected_label = QLabel()
        layout.addWidget(self.no_point_selected_label)

    def create_optimizers_checkbox(self, layout):
        self.adagrad_checkbox = QCheckBox("Adagrad")
        self.adam_checkbox = QCheckBox("Adam")
        self.nesterov_checkbox = QCheckBox("Nesterov")
        self.adagrad_checkbox.setChecked(True)
        self.adam_checkbox.setChecked(True)
        self.nesterov_checkbox.setChecked(True)
        layout.addWidget(self.adagrad_checkbox)
        layout.addWidget(self.adam_checkbox)
        layout.addWidget(self.nesterov_checkbox)

    def create_3d_plot(self, layout):
        self.fig3d = plt.figure()
        self.ax3d = self.fig3d.add_subplot(111, projection='3d')
        self.canvas3d = FigureCanvas(self.fig3d)
        layout.addWidget(self.canvas3d)

    def create_2d_plot(self, layout):
        self.fig2d = plt.figure()
        self.ax2d = self.fig2d.add_subplot(111)
        self.canvas2d = FigureCanvas(self.fig2d)
        layout.addWidget(self.canvas2d)

    def plot_function(self):
        func_name = self.functionSelector.currentText()
        func_data = self.function_details[func_name]
        self.function = func_data['function']
        self.gradient = func_data['gradient']
        bounds = func_data['bounds']

        self.ax3d.clear()
        self.ax2d.clear()

        x = np.linspace(bounds['xmin'], bounds['xmax'], 100)
        y = np.linspace(bounds['ymin'], bounds['ymax'], 100)
        X, Y = np.meshgrid(x, y)
        Z = self.function(X, Y)

        self.ax3d.plot_surface(X, Y, Z, cmap='viridis', alpha=0.7)
        self.ax2d.contourf(X, Y, Z, 15, cmap='viridis', alpha=0.7)
        self.canvas3d.draw()
        self.canvas2d.draw()

        self.canvas2d.mpl_connect('button_press_event', self.on_2d_plot_click)

    def on_2d_plot_click(self, event):
        if self.animation_running:
            self.ani.event_source.stop()
            self.animation_running = False
            self.gradientDescentButton.setText("Plot Descent paths")
        if self.selected_point:
            self.selected_point.remove()

        z_shift = 0
        self.selected_point, = self.ax2d.plot(event.xdata, event.ydata, 'ro')
        self.canvas2d.draw()
        z = (self.function(event.xdata, event.ydata)) + z_shift

        if hasattr(self, 'selected_point_3d') and self.selected_point_3d:
            self.selected_point_3d.remove()

        self.selected_point_3d, = self.ax3d.plot([event.xdata], [event.ydata], [z], 'ro')

        self.canvas3d.draw()
        self.starting_point = np.array([event.xdata, event.ydata])

    def plot_algorithms(self):
        if self.animation_running:
            self.ani.event_source.stop()
            self.animation_running = False
            self.gradientDescentButton.setText("Plot Descent paths")
            return
        else:
            self.gradientDescentButton.setText("Stop animation")
            self.animation_running = True

        if self.selected_point is None:
            self.no_point_selected_label.setText("Please select a starting point.")
            return
        else:
            self.no_point_selected_label.setText("")

        # Update num_iterations and learning_rate from the input fields
        self.num_iterations = int(self.num_iterations_input.text())

        # Run the Adagrad and Adam algorithms
        if self.adagrad_checkbox.isChecked():
            adagrad_epsilon = float(self.adagrad_epsilon_input.text())
            adagrad_learning_rate = float(self.adagrad_learning_rate_input.text())
            adagrad_x_history, adagrad_y_history, adagrad_z_history = self.adagrad(adagrad_epsilon=adagrad_epsilon, adagrad_learning_rate=adagrad_learning_rate)
        if self.adam_checkbox.isChecked():
            adam_beta1 = float(self.adam_beta1_input.text())
            adam_beta2 = float(self.adam_beta2_input.text())
            adam_epsilon = float(self.adam_epsilon_input.text())
            adam_learning_rate = float(self.adam_learning_rate_input.text())
            adam_x_history, adam_y_history, adam_z_history = self.adam(adam_beta1=adam_beta1, adam_beta2=adam_beta2, adam_epsilon=adam_epsilon, adam_learning_rate=adam_learning_rate)
        if self.nesterov_checkbox.isChecked():
            nesterov_momentum = float(self.nesterov_momentum_input.text())
            nesterov_learning_rate = float(self.nesterov_learning_rate_input.text())
            nesterov_x_history, nesterov_y_history, nesterov_z_history = self.nesterov(momentum=nesterov_momentum, nesterov_learning_rate=nesterov_learning_rate)

        # Remove the previous path_lines
        for line in self.path_lines:
            line.remove()
        self.path_lines.clear()

        # Create and animate the new path_lines
        if self.adagrad_checkbox.isChecked():
            adagrad_line_2d = Line2D([], [], color='blue', label='Adagrad', animated=True, linewidth=3, alpha=0.7)
            adagrad_line_3d, = self.ax3d.plot([], [], [], color='blue', label='Adagrad', animated=True, linewidth=3, alpha=0.7)
            self.path_lines.extend([adagrad_line_2d])
            self.path_lines.extend([adagrad_line_3d])
            self.ax2d.add_line(adagrad_line_2d)

        if self.adam_checkbox.isChecked():
            adam_line_2d = Line2D([], [], color='red', label='Adam', animated=True, linewidth=3, alpha=0.7)
            adam_line_3d, = self.ax3d.plot([], [], [], color='red', label='Adam', animated=True, linewidth=3, alpha=0.7)
            self.path_lines.extend([adam_line_2d])
            self.path_lines.extend([adam_line_3d])
            self.ax2d.add_line(adam_line_2d)

        if self.nesterov_checkbox.isChecked():
            nesterov_line_2d = Line2D([], [], color='green', label='Nesterov', animated=True, linewidth=3, alpha=0.7)
            nesterov_line_3d, = self.ax3d.plot([], [], [], color='green', label='Nesterov', animated=True, linewidth=3, alpha=0.7)
            self.path_lines.extend([nesterov_line_2d])
            self.path_lines.extend([nesterov_line_3d])
            self.ax2d.add_line(nesterov_line_2d)


        self.ax2d.legend()
        self.ax3d.legend()

        def animate(i):
            artists = []
            if self.adagrad_checkbox.isChecked():
                adagrad_line_2d.set_data(adagrad_x_history[:i + 1], adagrad_y_history[:i + 1])
                adagrad_line_3d.set_data(adagrad_x_history[:i + 1], adagrad_y_history[:i + 1])
                adagrad_line_3d.set_3d_properties(adagrad_z_history[:i + 1])
                artists.extend([adagrad_line_2d, adagrad_line_3d])
            if self.adam_checkbox.isChecked():
                adam_line_2d.set_data(adam_x_history[:i + 1], adam_y_history[:i + 1])
                adam_line_3d.set_data(adam_x_history[:i + 1], adam_y_history[:i + 1])
                adam_line_3d.set_3d_properties(adam_z_history[:i + 1])
                artists.extend([adam_line_2d, adam_line_3d])
            if self.nesterov_checkbox.isChecked():
                nesterov_line_2d.set_data(nesterov_x_history[:i + 1], nesterov_y_history[:i + 1])
                nesterov_line_3d.set_data(nesterov_x_history[:i + 1], nesterov_y_history[:i + 1])
                nesterov_line_3d.set_3d_properties(nesterov_z_history[:i + 1])
                artists.extend([nesterov_line_2d, nesterov_line_3d])
            self.canvas2d.draw()
            self.canvas3d.draw()
            return artists

        self.ani = FuncAnimation(self.fig2d, animate, frames=self.num_iterations, interval=10, repeat=True, blit=True)



    # Add the adagrad method to the MainWindow class
    def adagrad(self, adagrad_epsilon, adagrad_learning_rate):
        num_iterations = self.num_iterations
        learning_rate = adagrad_learning_rate
        x, y = self.starting_point[0], self.starting_point[1]
        x_history, y_history = [x], [y]
        grad_x2_sum, grad_y2_sum = 0, 0

        for i in range(num_iterations):
            dx, dy = self.gradient(x, y)
            grad_x2_sum += dx ** 2
            grad_y2_sum += dy ** 2
            x -= learning_rate * dx / (np.sqrt(grad_x2_sum) + adagrad_epsilon)
            y -= learning_rate * dy / (np.sqrt(grad_y2_sum) + adagrad_epsilon)
            x_history.append(x)
            y_history.append(y)

        return x_history, y_history, [self.function(x, y) for x, y in zip(x_history, y_history)]

    # Add the adam method to the MainWindow class
    def adam(self, adam_beta1, adam_beta2, adam_epsilon, adam_learning_rate):
        num_iterations = self.num_iterations
        learning_rate = adam_learning_rate
        beta1, beta2 = adam_beta1, adam_beta2
        epsilon = adam_epsilon
        x, y = self.starting_point[0], self.starting_point[1]
        x_history, y_history = [x], [y]
        m_x, m_y, v_x, v_y = 0, 0, 0, 0

        for i in range(1, num_iterations + 1):
            dx, dy = self.gradient(x, y)
            m_x = beta1 * m_x + (1 - beta1) * dx
            m_y = beta1 * m_y + (1 - beta1) * dy
            v_x = beta2 * v_x + (1 - beta2) * (dx ** 2)
            v_y = beta2 * v_y + (1 - beta2) * (dy ** 2)
            m_x_corr = m_x / (1 - beta1 ** i)
            m_y_corr = m_y / (1 - beta1 ** i)
            v_x_corr = v_x / (1 - beta2 ** i)
            v_y_corr = v_y / (1 - beta2 ** i)

            x -= learning_rate * m_x_corr / (np.sqrt(v_x_corr) + epsilon)
            y -= learning_rate * m_y_corr / (np.sqrt(v_y_corr) + epsilon)
            x_history.append(x)
            y_history.append(y)

        return x_history, y_history, [self.function(x, y) for x, y in zip(x_history, y_history)]

    # Add the Nesterov Accelerated Gradient method to the MainWindow class
    def nesterov(self, momentum, nesterov_learning_rate):
        num_iterations = self.num_iterations
        mu = momentum
        learning_rate = nesterov_learning_rate
        x, y = self.starting_point[0], self.starting_point[1]
        x_history, y_history = [x], [y]
        v_x, v_y = 0, 0

        for i in range(1, num_iterations + 1):
            x_lookahead = x - mu * v_x
            y_lookahead = y - mu * v_y
            dx_lookahead, dy_lookahead = self.gradient(x_lookahead, y_lookahead)
            v_x = mu * v_x + learning_rate * dx_lookahead
            v_y = mu * v_y + learning_rate * dy_lookahead
            x -= v_x
            y -= v_y
            x_history.append(x)
            y_history.append(y)

        return x_history, y_history, [self.function(x, y) for x, y in zip(x_history, y_history)]


if __name__ == '__main__':
    app = QApplication(sys.argv)
    mainWindow = MainWindow()
    mainWindow.show()
    sys.exit(app.exec_())
