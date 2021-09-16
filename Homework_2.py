import numpy as np
from matplotlib import pyplot as plt

obj = lambda x: (1 - 2 * x[0] - 3 * x[1] + 1) ** 2 + x[0] ** 2 + (x[1] - 1) ** 2


def grad(x):
    return [10 * x[0] + 12 * x[1] - 8, 12 * x[0] + 20 * x[1] - 14]


eps = 1e-3
x0 = [0., 0.]
k = 0
soln = [x0]
x = soln[k]
error = np.linalg.norm(grad(x))
a = 0.01

while error >= eps:
    x = np.array(x)
    soln.append(x)
    error = np.linalg.norm(grad(x))

error2_c1 = np.asarray([])
error2_c2 = np.asarray([])
for lineup in range(1, np.size(soln, 0)):
    sols = np.array(soln[lineup]) - np.array(soln[lineup - 1])
    error2_c1 = np.append(error2_c1, sols[0])
    error2_c2 = np.append(error2_c2, sols[1])

error2 = np.column_stack((error2_c1, error2_c2))
print(error2)

plt.title("Final Graph")
plt.xlabel("x")
plt.ylabel("y")
plt.plot(error2_c1, error2_c2)
plt.show()