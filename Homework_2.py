# Code for Gradient descent
import numpy as np
import scipy
from scipy.optimize import optimize
import matplotlib.pyplot as plt


def f(x):
    return 5 * x[0] * 2 + 12 * x[0] * x[1] - 8 * x[1] + 10 * x[1] * 2 - 14 * x[1] + 4


def grad(x):
    return np.array([10 * x[0] + 12 * x[1] - 8, 12 * x[0] + 20 * x[1] - 14])


# return [10 * x[0] + 12 * x[1] - 8, 12 * x[0] + 20 * x[1] - 14]
# def grad_d(grad, start, learn_r, tol):
#     pass

# grad_d(
#     grad=lambda x: np.array([10 * x[0] + 12 * x[1] - 8, 12 * x[0] + 20 * x[1] - 14]),
#     start=np.array([0, 0]), learn_r=0.2, tol=1e-3
# )

eps = 1e-3
x0 = (0, 0)
# x0 = 0
k = 0
soln = [x0]
x = soln[k]
# x_1 = []
# error = np.linalg.norm([10 * x[0] + 12 * x[1] - 8, 12 * x[0] + 20 * x[1] - 14])
# error = np.linalg.norm(grad(x))
error = scipy.linalg.norm(grad(x))
a = 0.01
xs = [-0.14244893, 0.78544215]
py = []
# print(error)


# line search code starts here

# def line_search(x):
#     a = 0.01
#     d = -1 * grad(x)
#     def phi(a, x):
#         return f(x) - a * 0.8 * np.matmul(np.transpose(grad(x)), d)
#     # print(phi(x))
#     while phi(a, x) < f(x - a * grad(x)):
#         a = 0.5 * a
#     return a

# while phi(a, x) > f(x - a * np.array(grad(x))):
#     a = 0.5 * a


while error >= eps:
    #     a = line_search(x)
    x = x - a * grad(x)
    #     x = x - np.matmul(grad(x), a)
    soln.append(x)
    #     x1 = 1 - 2*soln[0] - 3*soln[1]
    #     x_1.append(x1)
    error = scipy.linalg.norm(grad(x))
    ans = f(x)
    y = abs(f(x) - f(xs))
    py.append(float(y))
    # error = np.linalg.norm(x_0 - x_k)
    # conv = np.linalg.norm(grad(x_0))
    # soln.append(float(error))

    # soln.append(x)
    # error = np.linalg.norm(g(x))
# x = np.array([-2 * x_0[0] - 3 * x_0[1] + 1, x_0[0], x_0[0]])
# print(x)
# print(soln)
# soln
# p = log()
# x1 = 1 - 2*x[0] - 3*x[1]
y = [np.log(np.linalg.norm(s - f(x))) for s in soln]
plt.plot(py)
plt.title("Gradient Descent Convergence Plot")
plt.xlabel("Iteration")
plt.ylabel("Convergence")
plt.yscale("log")
soln