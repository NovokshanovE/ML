import numpy as np
import matplotlib.pyplot as plt
from funktions import *

A1 = np.array([[1, 1, 0, 3],
               [2, 1, -1, 1],
               [3, -1, -1, 2],
               [-1, 2, 3, -1]], dtype=np.float16)

b1 = np.array([[4],
               [1],
               [-3],
               [4]], dtype=np.float16)

A2 = np.array([[3, 1, -3],
               [6, 2, 5],
               [1, 4, -3]], dtype=np.float64)

b2 = np.array([[-16],
               [12],
               [-39]], dtype=np.float64)



if __name__ == '__main__':
    L1, U1 = lu(A1)
    X1 = sol(L1, U1, b1)
    P, L, U = lu_perm(A2, True)
    X = solve(L, U, P, b2)
    temp_A = A2[0][0]
    temp_b = b2[0][0]
    result = []
    result1 = []
    y_nodes = []
    y_nodes1 = []
    x_nodes = []
    p_ax = np.linspace(0, 12, 101)
    for i in range(101):
        A2[0][0] = temp_A + 10**(-p_ax[i])
        b2[0][0] = temp_b + 10**(-p_ax[i])
        P, L, U = lu_perm(A2, False)
        result.append(solve(L, U, P, b2))
        P, L, U = lu_perm(A2, True)
        result1.append(solve(L, U, P, b2))
        print(p_ax[i], " ", result[i])
        y_nodes.append(abs(norm(X) - norm(result[i])) / norm(X))
        y_nodes1.append(abs(norm(X) - norm(result1[i])) / norm(X))
        x_nodes.append(p_ax[i])
    for i in range(13):
        print(y_nodes[i])
    fig, ax = plt.subplots(1, 1, figsize=(12, 6))
    ax.grid()
    ax.set_xlabel('$p$', size = 18)
    ax.set_ylabel('$E$', size = 18)
    ax.semilogy(x_nodes, y_nodes, 'bo', markersize=6, label="permute = False")
    ax.semilogy(x_nodes, y_nodes1, 'go', markersize=6, label="permute = True")
    ax.legend(prop={'size': 15})

    plt.show()



