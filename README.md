# Gradient Descent Visualiser

Small project to allow the user to be able to visualise how different gradient descent optimisers perform on a selection of objective functions (Rosenbrock, Ackley and Himmelbrau). Currently allows the user to visualise the Adam, Nesterov Accelerated Gradient and Adagrad optimisers.

The user can select a point on the 2D contour plot of the function in the GUI, which serves as the starting point for the optimisers. Parameters for these optimisers can be changed by typing appropriate values into the input boxes on the left. Clicking the "Plot Descent paths" button starts an animation on the 2D and 3D contour plots that shows the paths of the descent. 

## Installation

Create an environment with the environment.yml file:

```python
conda env create -f environment.yml
```

Activate the environment, which I have named gd, and run the gradient_descent_visualiser.py 

```python
conda activate gd
python gradient_descent_visualiser.py
```

